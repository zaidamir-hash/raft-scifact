#!/usr/bin/env python3
"""Mine the approved Stage 4 BM25 and Stage-3-dense hard negatives.

Only Stage 1 training queries and qrels are read. Validation and test files
are never opened by this entry point. The output is deterministic and cached
with hashes of every input that affects candidate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import (
    atomic_write_json,
    join_passage,
    load_json,
    load_jsonl,
    load_qrels,
    sha256_file,
    tokenize_bm25,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage4_hardneg_pilot.json"
EXPECTED_MODEL = "intfloat/e5-small-v2"
EXPECTED_QUERY_PREFIX = "query: "
EXPECTED_PASSAGE_PREFIX = "passage: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage3-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and artifact paths without loading mining models.",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != 4 or config.get("run_type") not in {
        "single_seed_pilot",
        "final_3_seed_member",
    }:
        raise ValueError("Expected an approved Stage 4 pilot or final-seed config")
    if config.get("method") != "full_finetune_mined_hard_negatives":
        raise ValueError("Unexpected Stage 4 method")

    dataset = config.get("dataset", {})
    expected_dataset = {
        "name": "scifact",
        "source": "BEIR",
        "train_split": "train",
        "mining_split": "train",
        "checkpoint_selection_split": "validation",
        "test_evaluation": False,
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise ValueError(f"Approved dataset setting changed: {key}")

    model = config.get("model", {})
    if model.get("name") != EXPECTED_MODEL or not model.get("revision"):
        raise ValueError("Stage 4 must train the pinned E5-small-v2 revision")
    if model.get("initialization") != "fresh_pretrained_base":
        raise ValueError("Stage 4 must start fresh rather than continue Stage 3")
    if model.get("query_prefix") != EXPECTED_QUERY_PREFIX:
        raise ValueError("E5 query prefix must be exactly 'query: '")
    if model.get("passage_prefix") != EXPECTED_PASSAGE_PREFIX:
        raise ValueError("E5 passage prefix must be exactly 'passage: '")
    if model.get("max_sequence_length") != 512:
        raise ValueError("E5-small-v2 max sequence length must remain 512")

    mining = config.get("mining", {})
    if mining.get("methods") != ["bm25", "stage3_seed42_dense"]:
        raise ValueError("Mining must use BM25 plus the Stage 3 seed-42 model")
    if mining.get("candidate_pool_size_per_method") != 5183:
        raise ValueError("Each approved candidate search must cover all 5,183 documents")
    if mining.get("candidate_search_scope") != "full_corpus":
        raise ValueError("The approved candidate search scope is the full corpus")
    if mining.get("hard_negatives_per_positive") != 2:
        raise ValueError("Exactly two hard negatives are required per positive")
    if mining.get("hard_negatives_by_source") != {
        "bm25": 1,
        "stage3_seed42_dense": 1,
    }:
        raise ValueError("Each example requires one BM25 and one dense negative")
    if mining.get("deduplicate_across_sources") is not True:
        raise ValueError("Hard negatives must be distinct across mining sources")
    dense = mining.get("dense", {})
    if dense.get("model_source") != "stage3_seed42_best_checkpoint":
        raise ValueError("Dense mining must use the Stage 3 seed-42 checkpoint")
    filtering = mining.get("false_negative_filter", {})
    expected_filter = {
        "exclude_all_query_qrels_positives": True,
        "exclude_normalized_exact_positive_duplicates": True,
        "cosine_margin": 0.05,
        "reference_positive_score": "minimum_gold_positive_cosine_similarity",
        "acceptance_rule": "candidate_score <= minimum_gold_positive_score - 0.05",
        "relax_filter_if_insufficient": False,
        "stop_if_no_full_corpus_candidate_passes": True,
    }
    for key, value in expected_filter.items():
        if filtering.get(key) != value:
            raise ValueError(f"Approved false-negative filter changed: {key}")


def require_training_files(data_dir: Path) -> dict[str, Path]:
    files = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries_train.jsonl",
        "qrels": data_dir / "qrels_train.tsv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1 training artifacts: {missing}")
    return files


def require_stage3_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "checkpoint": run_dir / "best_checkpoint",
        "result": run_dir / "results" / "stage3_final_result.json",
        "embeddings": run_dir / "embeddings" / "corpus_embeddings.npy",
        "corpus_ids": run_dir / "embeddings" / "corpus_ids.json",
        "cache_metadata": run_dir / "embeddings" / "cache_metadata.json",
        "faiss_index": run_dir / "embeddings" / "stage3_inbatch_seed42.index.faiss",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 3 seed-42 artifacts: {missing}")
    if not files["checkpoint"].is_dir():
        raise FileNotFoundError(f"Stage 3 checkpoint is not a directory: {files['checkpoint']}")
    result = load_json(files["result"])
    if (
        result.get("stage") != 3
        or result.get("mode") != "final"
        or result.get("test_split_loaded") is not False
        or result.get("config", {}).get("training", {}).get("random_seed") != 42
    ):
        raise ValueError("Stage 3 seed-42 result metadata is not valid")
    return files


def normalize_passage(text: str) -> str:
    return " ".join(text.casefold().split())


def checkpoint_fingerprint(checkpoint_dir: Path) -> dict[str, str]:
    candidates = sorted(checkpoint_dir.rglob("*.safetensors"))
    if not candidates:
        candidates = sorted(checkpoint_dir.rglob("pytorch_model*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No model weights found under {checkpoint_dir}")
    return {str(path.relative_to(checkpoint_dir)): sha256_file(path) for path in candidates}


def config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def select_candidate(
    query_id: str,
    source: str,
    ranked_ids: list[str],
    scores: dict[str, float],
    positive_ids: set[str],
    positive_texts: set[str],
    normalized_passages: dict[str, str],
    threshold: float,
    excluded_ids: set[str],
) -> tuple[dict[str, Any], Counter[str]]:
    rejected: Counter[str] = Counter()
    for rank, document_id in enumerate(ranked_ids, start=1):
        if document_id in positive_ids:
            rejected["qrels_positive"] += 1
            continue
        if document_id in excluded_ids:
            rejected["duplicate_across_sources"] += 1
            continue
        if normalized_passages[document_id] in positive_texts:
            rejected["normalized_exact_positive_duplicate"] += 1
            continue
        score = float(scores[document_id])
        if score > threshold:
            rejected["inside_cosine_margin"] += 1
            continue
        return {
            "document_id": document_id,
            "rank": rank,
            "dense_cosine_score": score,
            "filter_threshold": threshold,
        }, rejected
    raise RuntimeError(
        f"Query {query_id!r} has zero {source} candidates passing the fixed "
        "0.05 cosine-margin filter after searching the full corpus; the "
        "approved filter will not be relaxed"
    )


def mine(config: dict[str, Any], data_dir: Path, stage3_run_dir: Path, output_dir: Path) -> dict[str, Any]:
    training_files = require_training_files(data_dir)
    stage3_files = require_stage3_files(stage3_run_dir)
    corpus = load_jsonl(training_files["corpus"])
    queries = load_jsonl(training_files["queries"])
    qrels = load_qrels(training_files["qrels"])
    corpus_ids = [str(document["id"]) for document in corpus]
    corpus_id_set = set(corpus_ids)
    query_texts = {str(query["id"]): str(query["text"]) for query in queries}
    passage_texts = {str(document["id"]): join_passage(document) for document in corpus}
    normalized_passages = {
        document_id: normalize_passage(text) for document_id, text in passage_texts.items()
    }
    if set(query_texts) != set(qrels):
        raise ValueError("Training query IDs and qrels query IDs differ")
    pool_size = int(config["mining"]["candidate_pool_size_per_method"])
    if len(corpus_ids) != pool_size:
        raise ValueError(
            f"Approved full-corpus search expects {pool_size} documents, "
            f"but Stage 1 contains {len(corpus_ids)}"
        )
    if any(not set(positives) <= corpus_id_set for positives in qrels.values()):
        raise ValueError("Training qrels reference missing corpus documents")

    cached_ids = load_json(stage3_files["corpus_ids"])["corpus_ids"]
    if cached_ids != corpus_ids:
        raise ValueError("Stage 3 corpus-cache IDs differ from Stage 1 corpus order")
    cache_metadata = load_json(stage3_files["cache_metadata"])
    corpus_sha256 = sha256_file(training_files["corpus"])
    if cache_metadata.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Stage 3 dense cache was built from a different corpus")
    corpus_embeddings = np.load(stage3_files["embeddings"], allow_pickle=False)
    if corpus_embeddings.dtype != np.float32 or corpus_embeddings.shape[0] != len(corpus_ids):
        raise ValueError("Stage 3 corpus-embedding cache has an invalid shape or dtype")

    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    pattern = re.compile(config["mining"]["bm25"]["token_pattern"])
    tokenized_corpus = [tokenize_bm25(passage_texts[document_id], pattern) for document_id in corpus_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    model = SentenceTransformer(str(stage3_files["checkpoint"]), device="cuda")
    model.max_seq_length = int(config["model"]["max_sequence_length"])
    prefixed_queries = [
        config["model"]["query_prefix"] + query_texts[query_id]
        for query_id in sorted(query_texts)
    ]
    ordered_query_ids = sorted(query_texts)
    started = time.perf_counter()
    query_embeddings = model.encode(
        prefixed_queries,
        batch_size=int(config["mining"]["dense"]["query_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    query_embeddings = np.ascontiguousarray(query_embeddings)
    dense_scores_matrix = query_embeddings @ corpus_embeddings.T
    elapsed = time.perf_counter() - started

    margin = float(config["mining"]["false_negative_filter"]["cosine_margin"])
    rows: list[dict[str, Any]] = []
    rejection_totals: dict[str, Counter[str]] = {
        "bm25": Counter(),
        "stage3_seed42_dense": Counter(),
    }
    selected_documents: dict[str, Counter[str]] = {
        "bm25": Counter(),
        "stage3_seed42_dense": Counter(),
    }

    for query_position, query_id in enumerate(ordered_query_ids):
        positive_ids = set(qrels[query_id])
        positive_texts = {normalized_passages[document_id] for document_id in positive_ids}
        dense_scores = dense_scores_matrix[query_position]
        score_by_id = {
            document_id: float(dense_scores[index])
            for index, document_id in enumerate(corpus_ids)
        }
        minimum_positive_score = min(score_by_id[document_id] for document_id in positive_ids)
        threshold = minimum_positive_score - margin

        bm25_scores = bm25.get_scores(tokenize_bm25(query_texts[query_id], pattern))
        bm25_indices = np.argsort(-bm25_scores, kind="stable")[:pool_size]
        dense_indices = np.argsort(-dense_scores, kind="stable")[:pool_size]
        bm25_ids = [corpus_ids[index] for index in bm25_indices]
        dense_ids = [corpus_ids[index] for index in dense_indices]

        bm25_choice, bm25_rejections = select_candidate(
            query_id,
            "BM25",
            bm25_ids,
            score_by_id,
            positive_ids,
            positive_texts,
            normalized_passages,
            threshold,
            excluded_ids=set(),
        )
        dense_choice, dense_rejections = select_candidate(
            query_id,
            "Stage-3-seed-42 dense",
            dense_ids,
            score_by_id,
            positive_ids,
            positive_texts,
            normalized_passages,
            threshold,
            excluded_ids={bm25_choice["document_id"]},
        )
        rejection_totals["bm25"].update(bm25_rejections)
        rejection_totals["stage3_seed42_dense"].update(dense_rejections)
        selected_documents["bm25"][bm25_choice["document_id"]] += 1
        selected_documents["stage3_seed42_dense"][dense_choice["document_id"]] += 1

        for positive_id in sorted(positive_ids):
            rows.append(
                {
                    "query_id": query_id,
                    "positive_id": positive_id,
                    "negative_bm25_id": bm25_choice["document_id"],
                    "negative_bm25_rank": bm25_choice["rank"],
                    "negative_bm25_dense_cosine_score": bm25_choice["dense_cosine_score"],
                    "negative_dense_id": dense_choice["document_id"],
                    "negative_dense_rank": dense_choice["rank"],
                    "negative_dense_cosine_score": dense_choice["dense_cosine_score"],
                    "minimum_gold_positive_cosine_score": minimum_positive_score,
                    "filter_threshold": threshold,
                    "filter_passed": True,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    negatives_path = output_dir / "stage4_seed42_hard_negatives.jsonl"
    summary_path = output_dir / "stage4_seed42_mining_summary.json"
    write_jsonl(negatives_path, rows)
    summary = {
        "stage": 4,
        "method": "bm25_plus_stage3_seed42_dense",
        "config_sha256": config_digest(config),
        "config": config,
        "data": {
            "corpus_sha256": corpus_sha256,
            "queries_train_sha256": sha256_file(training_files["queries"]),
            "qrels_train_sha256": sha256_file(training_files["qrels"]),
            "corpus_documents": len(corpus),
            "training_queries": len(queries),
            "positive_pairs": len(rows),
            "validation_loaded": False,
            "test_split_loaded": False,
        },
        "dense_miner": {
            "source": "stage3_seed42_best_checkpoint",
            "checkpoint": str(stage3_files["checkpoint"]),
            "checkpoint_weight_sha256": checkpoint_fingerprint(stage3_files["checkpoint"]),
            "corpus_embeddings_sha256": sha256_file(stage3_files["embeddings"]),
            "query_embedding_seconds": round(elapsed, 3),
        },
        "selection": {
            "candidate_pool_size_per_method": pool_size,
            "candidate_search_scope": "full_corpus",
            "hard_negatives_per_positive": 2,
            "bm25_negatives_per_positive": 1,
            "dense_negatives_per_positive": 1,
            "cosine_margin": margin,
            "rejection_counts": {
                method: dict(sorted(counts.items()))
                for method, counts in rejection_totals.items()
            },
            "unique_selected_documents": {
                method: len(counts) for method, counts in selected_documents.items()
            },
            "filter_relaxed": False,
        },
        "outputs": {
            "hard_negatives": str(negatives_path),
            "hard_negatives_sha256": sha256_file(negatives_path),
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "rank-bm25", "sentence-transformers", "torch")
        },
    }
    atomic_write_json(summary_path, summary)
    print("STAGE4_HARD_NEGATIVE_MINING_COMPLETE")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validate_config(config)
    training_files = require_training_files(args.data_dir.resolve())
    stage3_files = require_stage3_files(args.stage3_run_dir.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "config": str(args.config.resolve()),
                    "training_files": {key: str(value) for key, value in training_files.items()},
                    "stage3_files": {key: str(value) for key, value in stage3_files.items()},
                    "validation_loaded": False,
                    "test_split_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    mine(config, args.data_dir.resolve(), args.stage3_run_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
