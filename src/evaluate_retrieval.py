#!/usr/bin/env python3
"""Evaluate untouched BM25 and pretrained E5-small-v2 on SciFact.

Stage 2 performs retrieval inference only. It contains no optimizer, loss,
backpropagation, weight updates, training loop, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_baselines.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "scifact"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "embeddings" / "pretrained_e5_small_v2"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
EXPECTED_QUERY_PREFIX = "query: "
EXPECTED_PASSAGE_PREFIX = "passage: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Ignore an otherwise valid pretrained corpus-embedding cache.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected qrels header in {path}: {reader.fieldnames}")
        for row in reader:
            score = float(row["score"])
            if score > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    return qrels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def portable_path(path: Path) -> str:
    """Prefer repository-relative paths in metadata intended for Git."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def join_passage(document: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(document.get("title", "")).strip(),
            str(document.get("text", "")).strip(),
        )
        if part
    )


def validate_config(config: dict[str, Any]) -> None:
    expected_metrics = {
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "precision_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    }
    if set(config.get("evaluation_splits", [])) != {"validation", "test"}:
        raise ValueError("Stage 2 must evaluate exactly validation and test")
    if set(config.get("metrics", [])) != expected_metrics:
        raise ValueError("Stage 2 metric set differs from the fixed specification")
    if int(config.get("top_k", 0)) != 10:
        raise ValueError("top_k must be 10 to compute the fixed Stage 2 metrics")
    e5_config = config.get("e5", {})
    if e5_config.get("model_name") != "intfloat/e5-small-v2":
        raise ValueError("The pretrained baseline must use intfloat/e5-small-v2")
    if e5_config.get("query_prefix") != EXPECTED_QUERY_PREFIX:
        raise ValueError("E5 query prefix must be exactly 'query: '")
    if e5_config.get("passage_prefix") != EXPECTED_PASSAGE_PREFIX:
        raise ValueError("E5 passage prefix must be exactly 'passage: '")
    if not e5_config.get("normalize_embeddings", False):
        raise ValueError("IndexFlatIP requires normalized E5 embeddings for cosine search")
    if e5_config.get("faiss_index") != "IndexFlatIP":
        raise ValueError("Stage 2 uses exact FAISS IndexFlatIP retrieval")


def validate_inputs(
    corpus: list[dict[str, Any]],
    queries_by_split: dict[str, list[dict[str, Any]]],
    qrels_by_split: dict[str, dict[str, dict[str, float]]],
) -> None:
    corpus_ids = [str(document["id"]) for document in corpus]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Duplicate corpus IDs")
    corpus_id_set = set(corpus_ids)
    for split, queries in queries_by_split.items():
        query_ids = [str(query["id"]) for query in queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Duplicate {split} query IDs")
        if set(query_ids) != set(qrels_by_split[split]):
            raise ValueError(f"{split} query IDs and qrels query IDs differ")
        referenced_documents = {
            document_id
            for relevant in qrels_by_split[split].values()
            for document_id in relevant
        }
        missing = referenced_documents - corpus_id_set
        if missing:
            raise ValueError(f"{split} qrels reference {len(missing)} missing documents")


def tokenize_bm25(text: str, pattern: re.Pattern[str]) -> list[str]:
    return pattern.findall(text.lower())


def retrieve_bm25(
    passages: list[str],
    corpus_ids: list[str],
    queries_by_split: dict[str, list[dict[str, Any]]],
    token_pattern: str,
    top_k: int,
) -> tuple[dict[str, dict[str, list[str]]], float]:
    from rank_bm25 import BM25Okapi

    pattern = re.compile(token_pattern)
    started = time.perf_counter()
    tokenized_corpus = [tokenize_bm25(passage, pattern) for passage in passages]
    index = BM25Okapi(tokenized_corpus)
    rankings: dict[str, dict[str, list[str]]] = {}
    for split, queries in queries_by_split.items():
        split_rankings: dict[str, list[str]] = {}
        for query in queries:
            scores = index.get_scores(tokenize_bm25(str(query["text"]), pattern))
            top_indices = np.argsort(-scores, kind="stable")[:top_k]
            split_rankings[str(query["id"])] = [corpus_ids[index] for index in top_indices]
        rankings[split] = split_rankings
    return rankings, time.perf_counter() - started


def resolve_model_revision(model: Any) -> str:
    try:
        revision = model._first_module().auto_model.config._commit_hash
    except (AttributeError, KeyError):
        revision = None
    return str(revision or "unknown")


def embedding_dimension(model: Any) -> int:
    """Support both current and older Sentence Transformers releases."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def load_or_create_e5_cache(
    model: Any,
    model_revision: str,
    e5_config: dict[str, Any],
    passages: list[str],
    corpus_ids: list[str],
    corpus_sha256: str,
    cache_dir: Path,
    force_reembed: bool,
) -> tuple[np.ndarray, Any, bool, float]:
    import faiss

    embeddings_path = cache_dir / "corpus_embeddings.npy"
    ids_path = cache_dir / "corpus_ids.json"
    index_path = cache_dir / "pretrained_e5.index.faiss"
    metadata_path = cache_dir / "cache_metadata.json"
    expected_metadata = {
        "corpus_sha256": corpus_sha256,
        "model_name": e5_config["model_name"],
        "model_revision": model_revision,
        "passage_prefix": e5_config["passage_prefix"],
        "max_sequence_length": int(e5_config["max_sequence_length"]),
        "normalize_embeddings": True,
        "faiss_index": "IndexFlatIP",
    }

    cache_valid = False
    if not force_reembed and all(
        path.is_file() for path in (embeddings_path, ids_path, index_path, metadata_path)
    ):
        try:
            metadata = load_json(metadata_path)
            cached_ids = load_json(ids_path)["corpus_ids"]
            embeddings = np.load(embeddings_path, allow_pickle=False)
            index = faiss.read_index(str(index_path))
            cache_valid = (
                all(metadata.get(key) == value for key, value in expected_metadata.items())
                and cached_ids == corpus_ids
                and embeddings.dtype == np.float32
                and embeddings.shape == (len(corpus_ids), embedding_dimension(model))
                and index.ntotal == len(corpus_ids)
                and index.d == embeddings.shape[1]
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            cache_valid = False

    if cache_valid:
        print(f"Using cached pretrained E5 corpus embeddings: {embeddings_path}")
        return embeddings, index, True, 0.0

    prefixed_passages = [e5_config["passage_prefix"] + passage for passage in passages]
    print(f"Embedding {len(prefixed_passages):,} passages with untouched E5-small-v2")
    started = time.perf_counter()
    embeddings = model.encode(
        prefixed_passages,
        batch_size=int(e5_config["corpus_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    elapsed = time.perf_counter() - started
    embeddings = np.ascontiguousarray(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(embeddings_path, embeddings)
    atomic_write_json(ids_path, {"corpus_ids": corpus_ids})
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_path)
    metadata = {
        **expected_metadata,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_count": int(embeddings.shape[0]),
    }
    atomic_write_json(metadata_path, metadata)
    return embeddings, index, False, elapsed


def retrieve_e5(
    model: Any,
    index: Any,
    corpus_ids: list[str],
    queries_by_split: dict[str, list[dict[str, Any]]],
    e5_config: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, dict[str, list[str]]], float]:
    rankings: dict[str, dict[str, list[str]]] = {}
    started = time.perf_counter()
    for split, queries in queries_by_split.items():
        prefixed_queries = [
            e5_config["query_prefix"] + str(query["text"]) for query in queries
        ]
        query_embeddings = model.encode(
            prefixed_queries,
            batch_size=int(e5_config["query_batch_size"]),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        _, indices = index.search(np.ascontiguousarray(query_embeddings), top_k)
        rankings[split] = {
            str(query["id"]): [corpus_ids[index] for index in row]
            for query, row in zip(queries, indices)
        }
    return rankings, time.perf_counter() - started


def dcg_at_k(relevances: Iterable[float], k: int) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 2)
        for rank, relevance in enumerate(list(relevances)[:k])
    )


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    per_query: dict[str, list[float]] = {
        "recall_at_5": [],
        "recall_at_10": [],
        "precision_at_5": [],
        "precision_at_10": [],
        "mrr_at_10": [],
        "ndcg_at_10": [],
    }
    if set(rankings) != set(qrels):
        raise ValueError("Ranking query IDs differ from qrels query IDs")

    for query_id, relevant in qrels.items():
        ranked_ids = rankings[query_id]
        relevant_ids = set(relevant)
        for k in (5, 10):
            hits = sum(document_id in relevant_ids for document_id in ranked_ids[:k])
            per_query[f"recall_at_{k}"].append(hits / len(relevant_ids))
            per_query[f"precision_at_{k}"].append(hits / k)

        reciprocal_rank = 0.0
        for rank, document_id in enumerate(ranked_ids[:10], start=1):
            if document_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        per_query["mrr_at_10"].append(reciprocal_rank)

        observed_relevance = [relevant.get(document_id, 0.0) for document_id in ranked_ids[:10]]
        ideal_relevance = sorted(relevant.values(), reverse=True)[:10]
        ideal_dcg = dcg_at_k(ideal_relevance, 10)
        per_query["ndcg_at_10"].append(
            dcg_at_k(observed_relevance, 10) / ideal_dcg if ideal_dcg else 0.0
        )

    return {metric: float(np.mean(values)) for metric, values in per_query.items()}


def package_versions() -> dict[str, str]:
    names = ("numpy", "torch", "transformers", "sentence-transformers", "faiss-cpu", "rank-bm25")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "split",
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "precision_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "query_count",
        "corpus_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validate_config(config)
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    results_dir = args.results_dir.resolve()
    splits = list(config["evaluation_splits"])

    corpus_path = data_dir / "corpus.jsonl"
    corpus = load_jsonl(corpus_path)
    queries_by_split = {
        split: load_jsonl(data_dir / f"queries_{split}.jsonl") for split in splits
    }
    qrels_by_split = {
        split: load_qrels(data_dir / f"qrels_{split}.tsv") for split in splits
    }
    validate_inputs(corpus, queries_by_split, qrels_by_split)

    corpus_ids = [str(document["id"]) for document in corpus]
    passages = [join_passage(document) for document in corpus]
    top_k = int(config["top_k"])
    bm25_rankings, bm25_seconds = retrieve_bm25(
        passages,
        corpus_ids,
        queries_by_split,
        config["bm25"]["token_pattern"],
        top_k,
    )

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    e5_config = config["e5"]
    device = "cuda" if e5_config["device"] == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Loading untouched {e5_config['model_name']} on {device}")
    model = SentenceTransformer(e5_config["model_name"], device=device)
    model.max_seq_length = int(e5_config["max_sequence_length"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_revision = resolve_model_revision(model)

    embeddings, index, cache_hit, corpus_embedding_seconds = load_or_create_e5_cache(
        model=model,
        model_revision=model_revision,
        e5_config=e5_config,
        passages=passages,
        corpus_ids=corpus_ids,
        corpus_sha256=sha256_file(corpus_path),
        cache_dir=cache_dir,
        force_reembed=args.force_reembed,
    )
    e5_rankings, e5_query_seconds = retrieve_e5(
        model, index, corpus_ids, queries_by_split, e5_config, top_k
    )

    results_rows: list[dict[str, Any]] = []
    for method, rankings_by_split in (
        ("bm25", bm25_rankings),
        ("pretrained_e5_small_v2", e5_rankings),
    ):
        for split in splits:
            metrics = evaluate_rankings(rankings_by_split[split], qrels_by_split[split])
            results_rows.append(
                {
                    "method": method,
                    "split": split,
                    **{metric: f"{value:.6f}" for metric, value in metrics.items()},
                    "query_count": len(queries_by_split[split]),
                    "corpus_size": len(corpus),
                }
            )

    csv_path = results_dir / "stage2_baselines.csv"
    json_path = results_dir / "stage2_baselines.json"
    write_results_csv(csv_path, results_rows)
    query_example = e5_config["query_prefix"] + str(queries_by_split["validation"][0]["text"])
    passage_example = e5_config["passage_prefix"] + passages[0]
    index_path = cache_dir / "pretrained_e5.index.faiss"
    metadata = {
        "stage": 2,
        "training_performed": False,
        "config": config,
        "data_directory": portable_path(data_dir),
        "cache_directory": portable_path(cache_dir),
        "results": results_rows,
        "e5": {
            "model_name": e5_config["model_name"],
            "resolved_model_revision": model_revision,
            "device": device,
            "embedding_dimension": int(embeddings.shape[1]),
            "embedding_count": int(embeddings.shape[0]),
            "embedding_dtype": str(embeddings.dtype),
            "embeddings_normalized": True,
            "query_prefix": e5_config["query_prefix"],
            "passage_prefix": e5_config["passage_prefix"],
            "query_example": query_example,
            "passage_example": passage_example,
            "faiss_index_type": type(index).__name__,
            "faiss_index_vectors": int(index.ntotal),
            "faiss_index_dimension": int(index.d),
            "faiss_index_file_bytes": index_path.stat().st_size,
            "corpus_embedding_cache_hit": cache_hit,
        },
        "timing_seconds": {
            "bm25_index_and_all_queries": round(bm25_seconds, 3),
            "e5_corpus_embedding": round(corpus_embedding_seconds, 3),
            "e5_all_queries": round(e5_query_seconds, 3),
        },
        "package_versions": package_versions(),
    }
    atomic_write_json(json_path, metadata)

    print("\nStage 2 results")
    with csv_path.open("r", encoding="utf-8") as handle:
        print(handle.read().rstrip())
    print("\nE5 / FAISS confirmation")
    print(f"Model revision: {model_revision}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index: {type(index).__name__}, vectors={index.ntotal:,}, bytes={index_path.stat().st_size:,}")
    print(f"Query example: {query_example[:240]!r}")
    print(f"Passage example: {passage_example[:240]!r}")
    print(f"Results written to: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
