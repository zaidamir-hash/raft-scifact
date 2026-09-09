#!/usr/bin/env python3
"""Stage 3: fully fine-tune E5-small-v2 with in-batch negatives.

This entry point deliberately reads only the Stage 1 train and validation
files. The SciFact test split is neither loaded nor evaluated during Stage 3.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import random
import sys
import time
from collections import defaultdict
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
    load_qrels,
    resolve_model_revision,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_inbatch_pilot.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "scifact"
DEFAULT_BASELINE_RESULTS = PROJECT_ROOT / "results" / "stage2_baselines.csv"
EXPECTED_MODEL = "intfloat/e5-small-v2"
EXPECTED_QUERY_PREFIX = "query: "
EXPECTED_PASSAGE_PREFIX = "passage: "
REQUIRED_METRICS = {
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "mrr_at_10",
    "ndcg_at_10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("smoke", "pilot", "final"), required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and train/validation inputs without loading a model.",
    )
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("stage", -1)) != 3:
        raise ValueError("This script requires a Stage 3 configuration")
    if config.get("method") != "full_finetune_inbatch_negatives":
        raise ValueError("Stage 3 must use full fine-tuning with in-batch negatives")

    dataset = config.get("dataset", {})
    if dataset.get("name") != "scifact" or dataset.get("source") != "BEIR":
        raise ValueError("Only BEIR SciFact is allowed")
    if dataset.get("train_split") != "train":
        raise ValueError("Training must use the Stage 1 train split")
    if dataset.get("checkpoint_selection_split") != "validation":
        raise ValueError("Checkpoint selection must use validation only")
    if dataset.get("test_evaluation") is not False:
        raise ValueError("Stage 3 must not evaluate on the test split")

    model = config.get("model", {})
    if model.get("name") != EXPECTED_MODEL:
        raise ValueError(f"Base model must be {EXPECTED_MODEL}")
    if not model.get("revision"):
        raise ValueError("The exact pretrained model revision must be recorded")
    if model.get("query_prefix") != EXPECTED_QUERY_PREFIX:
        raise ValueError("E5 query prefix must be exactly 'query: '")
    if model.get("passage_prefix") != EXPECTED_PASSAGE_PREFIX:
        raise ValueError("E5 passage prefix must be exactly 'passage: '")
    if int(model.get("max_sequence_length", 0)) != 512:
        raise ValueError("E5-small-v2 max sequence length must be 512")

    training = config.get("training", {})
    expected = {
        "batch_size": 32,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "optimizer": "AdamW",
        "evaluation_frequency": "every_5_steps",
        "evaluation_steps": 5,
        "save_frequency": "every_5_steps",
        "save_steps": 5,
        "best_checkpoint_metric": "validation_ndcg_at_10",
        "loss": "MultipleNegativesRankingLoss",
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f"Approved Stage 3 setting changed: {key}")
    seed = int(training.get("random_seed", -1))
    run_type = config.get("run_type")
    if run_type == "single_seed_pilot" and seed != 42:
        raise ValueError("The approved pilot seed must remain 42")
    if run_type == "final_3_seed_member" and seed not in {42, 43, 44}:
        raise ValueError("The approved final Stage 3 seeds are 42, 43, and 44")
    if run_type not in {"single_seed_pilot", "final_3_seed_member"}:
        raise ValueError("Unsupported Stage 3 run_type")
    if training.get("batch_sampler") != "NO_DUPLICATES":
        raise ValueError("MultipleNegativesRankingLoss requires NO_DUPLICATES batches")
    early_stopping = training.get("early_stopping_rule", {})
    if (
        early_stopping.get("metric") != "validation_ndcg_at_10"
        or early_stopping.get("mode") != "max"
        or int(early_stopping.get("patience_evaluations", -1)) != 2
        or float(early_stopping.get("minimum_improvement", -1.0)) != 0.0
    ):
        raise ValueError("Early stopping differs from the approved rule")

    evaluation = config.get("evaluation", {})
    if set(evaluation.get("metrics", [])) != REQUIRED_METRICS:
        raise ValueError("Stage 3 retrieval metric set is incomplete")
    if int(evaluation.get("top_k", 0)) != 10:
        raise ValueError("Stage 3 validation retrieval must rank through k=10")
    if evaluation.get("selection_uses_validation_only") is not True:
        raise ValueError("Model selection must use validation only")
    if evaluation.get("load_or_read_test_split") is not False:
        raise ValueError("Stage 3 must not read the test split")


def load_stage3_data(
    data_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, float]],
]:
    """Load only corpus, train, and validation artifacts (never test)."""
    required = (
        "corpus.jsonl",
        "queries_train.jsonl",
        "qrels_train.tsv",
        "queries_validation.jsonl",
        "qrels_validation.tsv",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1 artifacts: {missing}")

    corpus = load_jsonl(data_dir / "corpus.jsonl")
    train_queries = load_jsonl(data_dir / "queries_train.jsonl")
    train_qrels = load_qrels(data_dir / "qrels_train.tsv")
    validation_queries = load_jsonl(data_dir / "queries_validation.jsonl")
    validation_qrels = load_qrels(data_dir / "qrels_validation.tsv")
    return corpus, train_queries, train_qrels, validation_queries, validation_qrels


def validate_stage3_data(
    corpus: list[dict[str, Any]],
    train_queries: list[dict[str, Any]],
    train_qrels: dict[str, dict[str, float]],
    validation_queries: list[dict[str, Any]],
    validation_qrels: dict[str, dict[str, float]],
) -> None:
    corpus_ids = [str(document["id"]) for document in corpus]
    train_ids = [str(query["id"]) for query in train_queries]
    validation_ids = [str(query["id"]) for query in validation_queries]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Duplicate corpus IDs")
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Duplicate query IDs")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train and validation query IDs overlap")
    if set(train_ids) != set(train_qrels) or set(validation_ids) != set(validation_qrels):
        raise ValueError("Query IDs and qrels query IDs differ")

    corpus_id_set = set(corpus_ids)
    train_documents = {
        document_id for relevant in train_qrels.values() for document_id in relevant
    }
    validation_documents = {
        document_id for relevant in validation_qrels.values() for document_id in relevant
    }
    if not (train_documents | validation_documents) <= corpus_id_set:
        raise ValueError("Qrels reference missing corpus documents")
    if train_documents & validation_documents:
        raise ValueError("Train/validation relevant-document leakage detected")


def build_training_pairs(
    corpus: list[dict[str, Any]],
    train_queries: list[dict[str, Any]],
    train_qrels: dict[str, dict[str, float]],
    query_prefix: str,
    passage_prefix: str,
) -> list[dict[str, str]]:
    query_text = {str(query["id"]): str(query["text"]) for query in train_queries}
    passage_text = {str(document["id"]): join_passage(document) for document in corpus}
    pairs: list[dict[str, str]] = []
    for query_id in sorted(train_qrels):
        for document_id in sorted(train_qrels[query_id]):
            pairs.append(
                {
                    "anchor": query_prefix + query_text[query_id],
                    "positive": passage_prefix + passage_text[document_id],
                }
            )
    return pairs


def resolve_output_dir(config: dict[str, Any], mode: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    artifact_key = {
        "smoke": "smoke_output",
        "pilot": "pilot_output",
        "final": "final_output",
    }[mode]
    relative = config["artifacts"][artifact_key]
    return (PROJECT_ROOT / relative).resolve()


def latest_resumable_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the newest complete Trainer checkpoint, if one exists."""
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        if not path.is_dir() or not (path / "trainer_state.json").is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def load_validation_baseline(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row.get("method") == "pretrained_e5_small_v2"
        and row.get("split") == "validation"
    ]
    if len(matches) != 1:
        raise ValueError("Expected one Stage 2 pretrained E5 validation result row")
    return {metric: float(matches[0][metric]) for metric in sorted(REQUIRED_METRICS)}


def write_loss_curve(path: Path, loss_points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ("step", "epoch", "loss", "learning_rate")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in loss_points:
            writer.writerow({name: point.get(name, "") for name in fieldnames})
    temporary.replace(path)


def summarise_loss(loss_points: list[dict[str, Any]]) -> dict[str, Any]:
    if not loss_points:
        raise RuntimeError("Trainer produced no finite training-loss observations")
    losses = [float(point["loss"]) for point in loss_points]
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Non-finite training loss observed")

    by_epoch: dict[int, list[float]] = defaultdict(list)
    for point in loss_points:
        epoch = max(1, int(math.ceil(float(point.get("epoch", 0.0)) - 1e-9)))
        by_epoch[epoch].append(float(point["loss"]))
    return {
        "observation_count": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "mean_loss": float(np.mean(losses)),
        "per_epoch_mean_loss": {
            str(epoch): float(np.mean(values)) for epoch, values in sorted(by_epoch.items())
        },
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "sentence-transformers", "faiss-cpu", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    baseline_path = args.baseline_results.resolve()
    output_dir = resolve_output_dir(config, args.mode, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, train_queries, train_qrels, validation_queries, validation_qrels = load_stage3_data(
        data_dir
    )
    validate_stage3_data(
        corpus, train_queries, train_qrels, validation_queries, validation_qrels
    )
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    seed = int(training_config["random_seed"])
    pairs = build_training_pairs(
        corpus,
        train_queries,
        train_qrels,
        model_config["query_prefix"],
        model_config["passage_prefix"],
    )
    if args.mode == "smoke":
        random.Random(seed).shuffle(pairs)
        pairs = pairs[: int(config["smoke_test"]["max_training_examples"])]

    if args.validate_only:
        return {
            "validated": True,
            "mode": args.mode,
            "config": str(config_path),
            "data_directory": str(data_dir),
            "corpus_documents": len(corpus),
            "training_queries": len(train_queries),
            "training_pairs": len(pairs),
            "validation_queries": len(validation_queries),
            "test_split_loaded": False,
        }

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

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_config['name']} revision {model_config['revision']} on {device}")
    model = SentenceTransformer(
        model_config["name"], revision=model_config["revision"], device=device
    )
    model.max_seq_length = int(model_config["max_sequence_length"])
    model.similarity_fn_name = SimilarityFunction.COSINE
    resolved_base_revision = resolve_model_revision(model)
    if resolved_base_revision != model_config["revision"]:
        raise RuntimeError(
            f"Resolved model revision {resolved_base_revision} differs from config "
            f"{model_config['revision']}"
        )

    train_dataset = Dataset.from_dict(
        {
            "anchor": [pair["anchor"] for pair in pairs],
            "positive": [pair["positive"] for pair in pairs],
        }
    )
    loss = MultipleNegativesRankingLoss(model)
    checkpoints_dir = output_dir / "checkpoints"
    is_evaluation_run = args.mode in {"pilot", "final"}
    best_metric_key = "eval_scifact_validation_cosine_ndcg@10"

    evaluator = None
    callbacks = None
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
            relevant_docs={
                query_id: set(relevant) for query_id, relevant in validation_qrels.items()
            },
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
                early_stopping_threshold=float(
                    training_config["early_stopping_rule"]["minimum_improvement"]
                ),
            )
        ]

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=int(training_config["batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        num_train_epochs=float(training_config["epochs"]),
        max_steps=(int(config["smoke_test"]["max_steps"]) if not is_evaluation_run else -1),
        learning_rate=float(training_config["learning_rate"]),
        warmup_ratio=float(training_config["warmup_ratio"]),
        lr_scheduler_type=str(training_config["lr_scheduler"]),
        optim="adamw_torch",
        weight_decay=float(training_config["weight_decay"]),
        max_grad_norm=float(training_config["max_grad_norm"]),
        fp16=(device == "cuda" and training_config["mixed_precision"] == "fp16_on_cuda"),
        bf16=False,
        eval_strategy=("steps" if is_evaluation_run else "no"),
        eval_steps=(int(training_config["evaluation_steps"]) if is_evaluation_run else None),
        save_strategy=("steps" if is_evaluation_run else "no"),
        save_steps=(int(training_config["save_steps"]) if is_evaluation_run else None),
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
        run_name=f"stage3-inbatch-{args.mode}-seed{seed}",
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
        latest_resumable_checkpoint(checkpoints_dir) if args.mode == "final" else None
    )
    started = time.perf_counter()
    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    elapsed = time.perf_counter() - started
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
    write_loss_curve(results_dir / f"stage3_{args.mode}_loss_curve.csv", loss_points)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    common = {
        "stage": 3,
        "mode": args.mode,
        "method": config["method"],
        "config": config,
        "config_path": str(config_path),
        "data_directory": str(data_dir),
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "resolved_base_model_revision": resolved_base_revision,
        "training_queries": len(train_queries),
        "training_pairs": len(pairs),
        "validation_queries": len(validation_queries),
        "corpus_size": len(corpus),
        "test_split_loaded": False,
        "global_steps_completed": int(trainer.state.global_step),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "training_seconds": round(elapsed, 3),
        "trainable_parameters": int(trainable_parameter_count),
        "total_parameters": int(parameter_count),
        "loss_summary": loss_summary,
        "package_versions": package_versions(),
    }

    if not is_evaluation_run:
        if int(trainer.state.global_step) != int(config["smoke_test"]["max_steps"]):
            raise RuntimeError("Smoke test did not complete the configured number of steps")
        smoke_result = {
            **common,
            "passed": True,
            "validation_run": False,
            "checkpoint_saved": False,
        }
        atomic_write_json(results_dir / "stage3_smoke_result.json", smoke_result)
        print("STAGE3_SMOKE_TEST_PASSED")
        print(json.dumps(smoke_result, indent=2, sort_keys=True))
        return smoke_result

    best_checkpoint_source = trainer.state.best_model_checkpoint
    if not best_checkpoint_source:
        raise RuntimeError("Pilot training did not select a validation checkpoint")
    best_checkpoint_dir = output_dir / "best_checkpoint"
    model.save_pretrained(str(best_checkpoint_dir))

    corpus_ids = [str(document["id"]) for document in corpus]
    prefixed_passages = [
        model_config["passage_prefix"] + join_passage(document) for document in corpus
    ]
    cache_dir = output_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_embeddings = model.encode(
        prefixed_passages,
        batch_size=int(evaluation_config["corpus_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    embeddings_path = cache_dir / "corpus_embeddings.npy"
    ids_path = cache_dir / "corpus_ids.json"
    index_path = cache_dir / f"stage3_inbatch_seed{seed}.index.faiss"
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
    _, retrieved_indices = index.search(
        np.ascontiguousarray(query_embeddings), int(evaluation_config["top_k"])
    )
    rankings = {
        str(query["id"]): [corpus_ids[index_position] for index_position in row]
        for query, row in zip(validation_queries, retrieved_indices)
    }
    pilot_metrics = evaluate_rankings(rankings, validation_qrels)
    baseline_metrics = load_validation_baseline(baseline_path)
    metric_deltas = {
        metric: pilot_metrics[metric] - baseline_metrics[metric]
        for metric in sorted(REQUIRED_METRICS)
    }

    result_prefix = "stage3_pilot" if args.mode == "pilot" else "stage3_final"
    comparison_path = results_dir / f"{result_prefix}_validation_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_comparison = comparison_path.with_suffix(comparison_path.suffix + ".tmp")
    fieldnames = ["method", "split", *sorted(REQUIRED_METRICS)]
    with temporary_comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "method": "pretrained_e5_small_v2",
                "split": "validation",
                **{metric: f"{baseline_metrics[metric]:.6f}" for metric in sorted(REQUIRED_METRICS)},
            }
        )
        writer.writerow(
            {
                "method": f"full_finetuned_e5_inbatch_seed{seed}_{args.mode}",
                "split": "validation",
                **{metric: f"{pilot_metrics[metric]:.6f}" for metric in sorted(REQUIRED_METRICS)},
            }
        )
        writer.writerow(
            {
                "method": "delta_vs_pretrained",
                "split": "validation",
                **{metric: f"{metric_deltas[metric]:+.6f}" for metric in sorted(REQUIRED_METRICS)},
            }
        )
    temporary_comparison.replace(comparison_path)

    evaluation_history = [
        entry for entry in trainer.state.log_history if best_metric_key in entry
    ]
    cache_metadata = {
        "model_name": model_config["name"],
        "base_model_revision": resolved_base_revision,
        "best_checkpoint_source": str(best_checkpoint_source),
        "saved_best_checkpoint": str(best_checkpoint_dir),
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
    training_result = {
        **common,
        "best_checkpoint_metric": "validation_ndcg_at_10",
        "best_checkpoint_metric_key": best_metric_key,
        "best_checkpoint_source": str(best_checkpoint_source),
        "saved_best_checkpoint": str(best_checkpoint_dir),
        "validation_evaluations": evaluation_history,
        "validation_metrics": pilot_metrics,
        "stage2_pretrained_validation_metrics": baseline_metrics,
        "delta_vs_stage2_pretrained": metric_deltas,
        "corpus_cache": cache_metadata,
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    result_path = results_dir / f"{result_prefix}_result.json"
    atomic_write_json(result_path, training_result)
    completion_marker = "STAGE3_PILOT_COMPLETE" if args.mode == "pilot" else f"STAGE3_FINAL_SEED_{seed}_COMPLETE"
    print(completion_marker)
    print(comparison_path.read_text(encoding="utf-8").rstrip())
    print(json.dumps(training_result, indent=2, sort_keys=True))
    return training_result


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validate_config(config)
    result = train(config, args)
    if args.validate_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
