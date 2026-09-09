#!/usr/bin/env python3
"""Stage 4: fully fine-tune E5-small-v2 with two mined hard negatives.

The student always starts from the pinned pretrained E5 base. This entry
point reads only Stage 1 train/validation artifacts; the test split is never
loaded or evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import (
    atomic_save_numpy,
    atomic_write_json,
    evaluate_rankings,
    join_passage,
    load_json,
    load_jsonl,
    resolve_model_revision,
    sha256_file,
)
from mine_negatives import config_digest, normalize_passage, validate_config as validate_mining_config
from train import (
    REQUIRED_METRICS,
    latest_resumable_checkpoint,
    load_stage3_data,
    load_validation_baseline,
    package_versions,
    summarise_loss,
    validate_stage3_data,
    write_loss_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage4_hardneg_pilot.json"
EXPECTED_MODEL = "intfloat/e5-small-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--mining-summary", type=Path, required=True)
    parser.add_argument("--stage2-results", type=Path, required=True)
    parser.add_argument("--stage3-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot", "final"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_training_config(config: dict[str, Any]) -> None:
    validate_mining_config(config)
    training = config.get("training", {})
    expected = {
        "batch_size": 32,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "lr_scheduler": "linear",
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16_on_cuda",
        "loss": "MultipleNegativesRankingLoss",
        "batch_sampler": "NO_DUPLICATES",
        "evaluation_frequency": "every_5_steps",
        "evaluation_steps": 5,
        "save_frequency": "every_5_steps",
        "save_steps": 5,
        "best_checkpoint_metric": "validation_ndcg_at_10",
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"Approved Stage 4 training setting changed: {key}")
    seed = int(training.get("random_seed", -1))
    run_type = config.get("run_type")
    if run_type == "single_seed_pilot" and seed != 42:
        raise ValueError("The approved Stage 4 pilot seed must remain 42")
    if run_type == "final_3_seed_member" and seed not in {42, 43, 44}:
        raise ValueError("The approved Stage 4 final seeds are 42, 43, and 44")
    if run_type not in {"single_seed_pilot", "final_3_seed_member"}:
        raise ValueError("Unsupported Stage 4 run type")
    if config.get("pilot_variant") and run_type != "single_seed_pilot":
        raise ValueError("Pilot adjustments cannot be applied to final runs")
    expected_patience = 4 if config.get("pilot_variant") == "early_stopping_patience4" else 2
    early_stopping = training.get("early_stopping_rule", {})
    if early_stopping != {
        "metric": "validation_ndcg_at_10",
        "mode": "max",
        "patience_evaluations": expected_patience,
        "minimum_improvement": 0.0,
    }:
        raise ValueError("Stage 4 early stopping differs from the approved rule")
    if expected_patience == 4:
        adjustment = config.get("pilot_adjustment", {})
        if (
            adjustment.get("approved_by_user") is not True
            or adjustment.get("only_changed_setting")
            != "training.early_stopping_rule.patience_evaluations"
            or adjustment.get("from") != 2
            or adjustment.get("to") != 4
        ):
            raise ValueError("The patience-4 pilot adjustment is not documented as approved")
    smoke = config.get("smoke_test", {})
    if (
        smoke.get("max_steps") != 3
        or smoke.get("max_training_examples") != 96
        or smoke.get("run_validation") is not False
        or smoke.get("stop_on_out_of_memory") is not True
        or smoke.get("allow_automatic_batch_size_change") is not False
        or smoke.get("allow_automatic_gradient_accumulation_change") is not False
    ):
        raise ValueError("Stage 4 smoke/OOM policy changed")
    evaluation = config.get("evaluation", {})
    if set(evaluation.get("metrics", [])) != REQUIRED_METRICS:
        raise ValueError("Stage 4 retrieval metric set is incomplete")
    if evaluation.get("top_k") != 10:
        raise ValueError("Stage 4 validation retrieval must rank through k=10")
    if evaluation.get("selection_uses_validation_only") is not True:
        raise ValueError("Checkpoint selection must use validation only")
    if evaluation.get("load_or_read_test_split") is not False:
        raise ValueError("Stage 4 must not load the test split")


def validate_mined_examples(
    config: dict[str, Any],
    hard_negatives_path: Path,
    mining_summary_path: Path,
    corpus: list[dict[str, Any]],
    train_queries: list[dict[str, Any]],
    train_qrels: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    if not hard_negatives_path.is_file() or not mining_summary_path.is_file():
        raise FileNotFoundError("Mined hard-negative artifacts are missing")
    summary = load_json(mining_summary_path)
    if (
        summary.get("stage") != 4
        or summary.get("method") != "bm25_plus_stage3_seed42_dense"
        or mining_signature(summary.get("config", {})) != mining_signature(config)
        or summary.get("data", {}).get("validation_loaded") is not False
        or summary.get("data", {}).get("test_split_loaded") is not False
        or summary.get("selection", {}).get("filter_relaxed") is not False
        or summary.get("outputs", {}).get("hard_negatives_sha256")
        != sha256_file(hard_negatives_path)
    ):
        raise ValueError("Mining summary does not match the approved config/artifact")

    rows = load_jsonl(hard_negatives_path)
    expected_pairs = {
        (query_id, positive_id)
        for query_id, positives in train_qrels.items()
        for positive_id in positives
    }
    observed_pairs = [(str(row["query_id"]), str(row["positive_id"])) for row in rows]
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("Mined artifact contains duplicate query-positive rows")
    if set(observed_pairs) != expected_pairs:
        raise ValueError("Mined rows do not exactly cover the Stage 1 positive pairs")

    corpus_text = {str(document["id"]): join_passage(document) for document in corpus}
    query_ids = {str(query["id"]) for query in train_queries}
    margin = float(config["mining"]["false_negative_filter"]["cosine_margin"])
    for row in rows:
        query_id = str(row["query_id"])
        positive_id = str(row["positive_id"])
        bm25_id = str(row["negative_bm25_id"])
        dense_id = str(row["negative_dense_id"])
        if query_id not in query_ids or positive_id not in train_qrels[query_id]:
            raise ValueError("Mined row references an invalid training positive")
        if bm25_id not in corpus_text or dense_id not in corpus_text or bm25_id == dense_id:
            raise ValueError("Each mined row requires two distinct corpus negatives")
        if bm25_id in train_qrels[query_id] or dense_id in train_qrels[query_id]:
            raise ValueError("A qrels-positive document was selected as a negative")
        positive_texts = {
            normalize_passage(corpus_text[document_id])
            for document_id in train_qrels[query_id]
        }
        if normalize_passage(corpus_text[bm25_id]) in positive_texts:
            raise ValueError("BM25 negative duplicates a positive passage")
        if normalize_passage(corpus_text[dense_id]) in positive_texts:
            raise ValueError("Dense negative duplicates a positive passage")
        threshold = float(row["minimum_gold_positive_cosine_score"]) - margin
        if not math.isclose(float(row["filter_threshold"]), threshold, abs_tol=1e-7):
            raise ValueError("Recorded hard-negative filter threshold is inconsistent")
        if float(row["negative_bm25_dense_cosine_score"]) > threshold + 1e-7:
            raise ValueError("BM25 negative violates the approved cosine margin")
        if float(row["negative_dense_cosine_score"]) > threshold + 1e-7:
            raise ValueError("Dense negative violates the approved cosine margin")
        if row.get("filter_passed") is not True:
            raise ValueError("Mined example did not pass the fixed filter")
    return rows


def mining_signature(config: dict[str, Any]) -> str:
    """Hash only settings that can affect the cached mining artifact."""
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    relevant = {
        "dataset": {
            "name": dataset.get("name"),
            "source": dataset.get("source"),
            "mining_split": dataset.get("mining_split"),
        },
        "model": {
            "query_prefix": model.get("query_prefix"),
            "passage_prefix": model.get("passage_prefix"),
            "max_sequence_length": model.get("max_sequence_length"),
        },
        "mining": config.get("mining"),
    }
    return config_digest(relevant)


def build_dataset_rows(
    config: dict[str, Any],
    mined_rows: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    train_queries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    query_text = {str(query["id"]): str(query["text"]) for query in train_queries}
    passage_text = {str(document["id"]): join_passage(document) for document in corpus}
    query_prefix = config["model"]["query_prefix"]
    passage_prefix = config["model"]["passage_prefix"]
    return [
        {
            "anchor": query_prefix + query_text[str(row["query_id"])],
            "positive": passage_prefix + passage_text[str(row["positive_id"])],
            "negative_bm25": passage_prefix + passage_text[str(row["negative_bm25_id"])],
            "negative_dense": passage_prefix + passage_text[str(row["negative_dense_id"])],
        }
        for row in mined_rows
    ]


def read_stage3_references(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    summary = load_json(path)
    if (
        summary.get("stage") != 3
        or summary.get("split") != "validation"
        or summary.get("test_split_loaded") is not False
    ):
        raise ValueError("Stage 3 reference summary is invalid")
    seed42 = [row for row in summary["per_seed"] if row.get("seed") == 42]
    if len(seed42) != 1:
        raise ValueError("Expected exactly one Stage 3 seed-42 result")
    seed42_metrics = {metric: float(seed42[0][metric]) for metric in REQUIRED_METRICS}
    mean_metrics = {
        metric: float(summary["aggregate"][metric]["mean"]) for metric in REQUIRED_METRICS
    }
    return seed42_metrics, mean_metrics


def write_comparison(
    path: Path,
    stage2: dict[str, float],
    stage3_seed42: dict[str, float],
    stage3_mean: dict[str, float],
    stage4: dict[str, float],
    stage4_method: str = "stage4_hard_negative_seed42_pilot",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["method", "split", *sorted(REQUIRED_METRICS)]
    rows = (
        ("stage2_pretrained_e5", stage2),
        ("stage3_inbatch_seed42", stage3_seed42),
        ("stage3_inbatch_three_seed_mean", stage3_mean),
        (stage4_method, stage4),
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in rows:
            writer.writerow(
                {
                    "method": method,
                    "split": "validation",
                    **{metric: f"{metrics[metric]:.6f}" for metric in sorted(REQUIRED_METRICS)},
                }
            )
    temporary.replace(path)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_run_config(output_dir: Path, config: dict[str, Any]) -> None:
    path = output_dir / "stage4_run_config.json"
    if path.is_file() and load_json(path) != config:
        raise RuntimeError(f"Existing output directory has a different config: {output_dir}")
    atomic_write_json(path, config)


def run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    corpus, train_queries, train_qrels, validation_queries, validation_qrels = load_stage3_data(
        data_dir
    )
    validate_stage3_data(
        corpus, train_queries, train_qrels, validation_queries, validation_qrels
    )
    mined_rows = validate_mined_examples(
        config,
        args.hard_negatives.resolve(),
        args.mining_summary.resolve(),
        corpus,
        train_queries,
        train_qrels,
    )
    dataset_rows = build_dataset_rows(config, mined_rows, corpus, train_queries)
    seed = int(config["training"]["random_seed"])
    if args.mode == "smoke":
        random.Random(seed).shuffle(dataset_rows)
        dataset_rows = dataset_rows[: int(config["smoke_test"]["max_training_examples"])]

    stage2_metrics = load_validation_baseline(args.stage2_results.resolve())
    stage3_seed42, stage3_mean = read_stage3_references(args.stage3_summary.resolve())
    if args.validate_only:
        result = {
            "validated": True,
            "stage": 4,
            "mode": args.mode,
            "training_pairs": len(dataset_rows),
            "hard_negatives_per_pair": 2,
            "bm25_negatives_per_pair": 1,
            "dense_negatives_per_pair": 1,
            "training_queries": len(train_queries),
            "validation_queries": len(validation_queries),
            "stage2_reference_loaded": True,
            "stage3_reference_loaded": True,
            "test_split_loaded": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    import faiss
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.similarity_functions import SimilarityFunction
    from sentence_transformers.training_args import BatchSamplers
    from transformers import EarlyStoppingCallback, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 training requires the approved free Colab GPU runtime")
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    print(
        f"Loading fresh base {model_config['name']} revision {model_config['revision']} on cuda"
    )
    model = SentenceTransformer(
        model_config["name"], revision=model_config["revision"], device="cuda"
    )
    model.max_seq_length = int(model_config["max_sequence_length"])
    model.similarity_fn_name = SimilarityFunction.COSINE
    resolved_revision = resolve_model_revision(model)
    if resolved_revision != model_config["revision"]:
        raise RuntimeError("Resolved base-model revision differs from the approved config")

    train_dataset = Dataset.from_dict(
        {name: [row[name] for row in dataset_rows] for name in dataset_rows[0]}
    )
    loss = MultipleNegativesRankingLoss(model)
    is_evaluation_run = args.mode in {"pilot", "final"}
    evaluator = None
    callbacks = None
    best_metric_key = "eval_scifact_validation_cosine_ndcg@10"
    if is_evaluation_run:
        evaluator = InformationRetrievalEvaluator(
            queries={
                str(query["id"]): model_config["query_prefix"] + str(query["text"])
                for query in validation_queries
            },
            corpus={
                str(document["id"]): model_config["passage_prefix"] + join_passage(document)
                for document in corpus
            },
            relevant_docs={query_id: set(relevant) for query_id, relevant in validation_qrels.items()},
            corpus_chunk_size=len(corpus),
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1, 5, 10],
            precision_recall_at_k=[5, 10],
            map_at_k=[10],
            show_progress_bar=True,
            batch_size=int(evaluation_config["corpus_batch_size"]),
            name="scifact_validation",
            write_csv=True,
            main_score_function=SimilarityFunction.COSINE,
        )
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=int(
                    training_config["early_stopping_rule"]["patience_evaluations"]
                ),
                early_stopping_threshold=0.0,
            )
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(output_dir, config)
    checkpoints_dir = output_dir / "checkpoints"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3.0,
        max_steps=(-1 if is_evaluation_run else 3),
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        eval_strategy=("steps" if is_evaluation_run else "no"),
        eval_steps=(5 if is_evaluation_run else None),
        save_strategy=("steps" if is_evaluation_run else "no"),
        save_steps=(5 if is_evaluation_run else None),
        load_best_model_at_end=is_evaluation_run,
        metric_for_best_model=(best_metric_key if is_evaluation_run else None),
        greater_is_better=(True if is_evaluation_run else None),
        save_total_limit=(3 if is_evaluation_run else None),
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=0,
        run_name=f"stage4-hardneg-{args.mode}-seed{seed}",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )
    resume_checkpoint = (
        latest_resumable_checkpoint(checkpoints_dir) if is_evaluation_run else None
    )
    started = time.perf_counter()
    try:
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if args.mode == "smoke" and "out of memory" in str(error).casefold():
            print("STAGE4_SMOKE_OOM_STOPPED_NO_CONFIG_CHANGE", file=sys.stderr)
        raise
    elapsed = time.perf_counter() - started
    peak_gpu_memory = int(torch.cuda.max_memory_allocated())
    model = trainer.model
    loss_points = [
        {
            "step": int(entry["step"]),
            "epoch": float(entry.get("epoch", 0.0)),
            "loss": float(entry["loss"]),
            "learning_rate": float(entry.get("learning_rate", 0.0)),
        }
        for entry in trainer.state.log_history
        if "loss" in entry and "step" in entry and math.isfinite(float(entry["loss"]))
    ]
    loss_summary = summarise_loss(loss_points)
    results_dir = output_dir / "results"
    write_loss_curve(results_dir / f"stage4_{args.mode}_loss_curve.csv", loss_points)
    common = {
        "stage": 4,
        "mode": args.mode,
        "method": config["method"],
        "config": config,
        "config_sha256": config_digest(config),
        "data_directory": str(data_dir),
        "hard_negatives": str(args.hard_negatives.resolve()),
        "hard_negatives_sha256": sha256_file(args.hard_negatives.resolve()),
        "mining_config_signature": mining_signature(config),
        "resolved_base_model_revision": resolved_revision,
        "fresh_pretrained_initialization": True,
        "training_queries": len(train_queries),
        "training_pairs": len(dataset_rows),
        "hard_negatives_per_pair": 2,
        "validation_queries": len(validation_queries),
        "corpus_size": len(corpus),
        "test_split_loaded": False,
        "global_steps_completed": int(trainer.state.global_step),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "training_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "loss_summary": loss_summary,
        "package_versions": package_versions(),
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }

    if not is_evaluation_run:
        if trainer.state.global_step != 3:
            raise RuntimeError("Stage 4 smoke test did not complete exactly three steps")
        smoke_result = {
            **common,
            "passed": True,
            "validation_run": False,
            "checkpoint_saved": False,
        }
        atomic_write_json(results_dir / "stage4_smoke_result.json", smoke_result)
        print("STAGE4_HARD_NEGATIVE_SMOKE_TEST_PASSED")
        print(json.dumps(smoke_result, indent=2, sort_keys=True))
        return smoke_result

    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Stage 4 pilot did not select a validation checkpoint")
    best_checkpoint_dir = output_dir / "best_checkpoint"
    model.save_pretrained(str(best_checkpoint_dir))
    corpus_ids = [str(document["id"]) for document in corpus]
    corpus_embeddings = model.encode(
        [model_config["passage_prefix"] + join_passage(document) for document in corpus],
        batch_size=int(evaluation_config["corpus_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    cache_dir = output_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "corpus_embeddings.npy"
    ids_path = cache_dir / "corpus_ids.json"
    index_path = cache_dir / f"stage4_hardneg_seed{seed}.index.faiss"
    atomic_save_numpy(embeddings_path, corpus_embeddings)
    atomic_write_json(ids_path, {"corpus_ids": corpus_ids})
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_path)
    query_embeddings = model.encode(
        [model_config["query_prefix"] + str(query["text"]) for query in validation_queries],
        batch_size=int(evaluation_config["query_batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    _, retrieved_indices = index.search(np.ascontiguousarray(query_embeddings), 10)
    rankings = {
        str(query["id"]): [corpus_ids[position] for position in row]
        for query, row in zip(validation_queries, retrieved_indices)
    }
    stage4_metrics = evaluate_rankings(rankings, validation_qrels)
    result_prefix = "stage4_pilot" if args.mode == "pilot" else "stage4_final"
    comparison_path = results_dir / f"{result_prefix}_validation_comparison.csv"
    write_comparison(
        comparison_path,
        stage2_metrics,
        stage3_seed42,
        stage3_mean,
        stage4_metrics,
        stage4_method=f"stage4_hard_negative_seed{seed}_{args.mode}",
    )
    metric_deltas = {
        "vs_stage2_pretrained": {
            metric: stage4_metrics[metric] - stage2_metrics[metric]
            for metric in sorted(REQUIRED_METRICS)
        },
        "vs_stage3_seed42": {
            metric: stage4_metrics[metric] - stage3_seed42[metric]
            for metric in sorted(REQUIRED_METRICS)
        },
        "vs_stage3_three_seed_mean": {
            metric: stage4_metrics[metric] - stage3_mean[metric]
            for metric in sorted(REQUIRED_METRICS)
        },
    }
    cache_metadata = {
        "model_name": EXPECTED_MODEL,
        "base_model_revision": resolved_revision,
        "best_checkpoint_source": str(trainer.state.best_model_checkpoint),
        "saved_best_checkpoint": str(best_checkpoint_dir),
        "checkpoint_size_bytes": directory_size(best_checkpoint_dir),
        "corpus_sha256": sha256_file(data_dir / "corpus.jsonl"),
        "embedding_count": int(corpus_embeddings.shape[0]),
        "embedding_dimension": int(corpus_embeddings.shape[1]),
        "embedding_dtype": str(corpus_embeddings.dtype),
        "normalize_embeddings": True,
        "faiss_index": type(index).__name__,
        "faiss_index_vectors": int(index.ntotal),
        "query_prefix": model_config["query_prefix"],
        "passage_prefix": model_config["passage_prefix"],
        "max_sequence_length": int(model.max_seq_length),
    }
    atomic_write_json(cache_dir / "cache_metadata.json", cache_metadata)
    result = {
        **common,
        "best_checkpoint_metric": "validation_ndcg_at_10",
        "best_checkpoint_metric_key": best_metric_key,
        "best_checkpoint_source": str(trainer.state.best_model_checkpoint),
        "saved_best_checkpoint": str(best_checkpoint_dir),
        "validation_evaluations": [
            entry for entry in trainer.state.log_history if best_metric_key in entry
        ],
        "validation_metrics": stage4_metrics,
        "stage2_pretrained_validation_metrics": stage2_metrics,
        "stage3_seed42_validation_metrics": stage3_seed42,
        "stage3_three_seed_mean_validation_metrics": stage3_mean,
        "metric_deltas": metric_deltas,
        "corpus_cache": cache_metadata,
    }
    atomic_write_json(results_dir / f"{result_prefix}_result.json", result)
    completion_marker = (
        "STAGE4_HARD_NEGATIVE_PILOT_COMPLETE"
        if args.mode == "pilot"
        else f"STAGE4_HARD_NEGATIVE_FINAL_SEED_{seed}_COMPLETE"
    )
    print(completion_marker)
    print(comparison_path.read_text(encoding="utf-8").rstrip())
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validate_training_config(config)
    run(config, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
