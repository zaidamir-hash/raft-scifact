#!/usr/bin/env python3
"""Stage 5 LoRA pilot and matched Stage 3 full-FT memory profile.

Only Stage 1 train/validation artifacts are loaded. The test split is never
read. ``profile-full`` is a three-step, no-validation efficiency measurement
and writes only beneath the Stage 5 Drive directory.
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
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import (
    atomic_save_numpy,
    atomic_write_json,
    evaluate_rankings,
    join_passage,
    load_json,
    resolve_model_revision,
    sha256_file,
)
from train import (
    build_training_pairs,
    latest_resumable_checkpoint,
    load_stage3_data,
    summarise_loss,
    validate_stage3_data,
    write_loss_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "intfloat/e5-small-v2"
EXPECTED_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
METRICS = (
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "mrr_at_10",
    "ndcg_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("profile-full", "smoke", "pilot", "final"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage2-validation", type=Path, required=True)
    parser.add_argument("--stage3-summary", type=Path, required=True)
    parser.add_argument("--stage4-summary", type=Path, required=True)
    parser.add_argument("--stage3-result", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != 5 or config.get("method") != "lora_inbatch_negatives":
        raise ValueError("Expected the approved Stage 5 LoRA pilot configuration")
    dataset = config.get("dataset", {})
    if dataset != {
        "name": "scifact",
        "source": "BEIR",
        "train_split": "train",
        "checkpoint_selection_split": "validation",
        "test_evaluation": False,
    }:
        raise ValueError("Stage 5 dataset/split configuration changed")
    model = config.get("model", {})
    expected_model = {
        "name": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "initialization": "fresh_pretrained_base_with_lora",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_sequence_length": 512,
        "normalize_embeddings": True,
    }
    if model != expected_model:
        raise ValueError("Approved E5 model configuration changed")
    lora = config.get("lora", {})
    expected_lora = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["query", "value"],
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
        "expected_target_module_count": 24,
        "save_adapter_only": True,
    }
    if lora != expected_lora:
        raise ValueError("Approved LoRA configuration changed")
    training = config.get("training", {})
    expected_training = {
        "batch_size": 32,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "lr_scheduler": "linear",
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16_on_cuda",
        "loss": "MultipleNegativesRankingLoss",
        "negative_sampling": "in_batch_only",
        "hard_negatives_per_positive": 0,
        "mining_method": None,
        "batch_sampler": "NO_DUPLICATES",
        "evaluation_frequency": "every_5_steps",
        "evaluation_steps": 5,
        "save_frequency": "every_5_steps",
        "save_steps": 5,
        "early_stopping_rule": {
            "metric": "validation_ndcg_at_10",
            "mode": "max",
            "patience_evaluations": 2,
            "minimum_improvement": 0.0,
        },
        "best_checkpoint_metric": "validation_ndcg_at_10",
    }
    training_without_seed = {key: value for key, value in training.items() if key != "random_seed"}
    if training_without_seed != expected_training:
        raise ValueError("Approved Stage 5 training configuration changed")
    seed = training.get("random_seed")
    run_type = config.get("run_type")
    if run_type == "single_seed_pilot" and seed != 42:
        raise ValueError("The Stage 5 pilot seed must remain 42")
    if run_type == "final_3_seed_member" and seed not in {42, 43, 44}:
        raise ValueError("Stage 5 final seeds must be 42, 43, or 44")
    if run_type not in {"single_seed_pilot", "final_3_seed_member"}:
        raise ValueError("Unsupported Stage 5 run type")
    reference = config.get("full_finetuning_reference", {})
    if (
        reference.get("stage") != 3
        or reference.get("method") != "full_finetune_inbatch_negatives"
        or reference.get("seed") != 42
        or reference.get("learning_rate") != 5e-6
        or reference.get("profiling_max_steps") != 3
    ):
        raise ValueError("Matched Stage 3 profiling reference changed")
    smoke = config.get("smoke_test", {})
    if (
        smoke.get("max_steps") != 3
        or smoke.get("max_training_examples") != 96
        or smoke.get("run_validation") is not False
        or smoke.get("stop_on_out_of_memory") is not True
        or smoke.get("allow_automatic_batch_size_change") is not False
        or smoke.get("allow_automatic_gradient_accumulation_change") is not False
    ):
        raise ValueError("Approved smoke-test constraints changed")
    evaluation = config.get("evaluation", {})
    if (
        set(evaluation.get("metrics", [])) != set(METRICS)
        or evaluation.get("top_k") != 10
        or evaluation.get("selection_uses_validation_only") is not True
        or evaluation.get("load_or_read_test_split") is not False
    ):
        raise ValueError("Stage 5 validation-only evaluation configuration changed")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "datasets",
        "sentence-transformers",
        "peft",
        "faiss-cpu",
        "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_stage2_validation(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("Stage 2 reference file must contain validation rows only")
    indexed = {row["method"]: row for row in rows}
    required = {"bm25", "pretrained_e5_small_v2"}
    if set(indexed) != required:
        raise ValueError("Stage 2 validation reference rows are incomplete")
    return {
        method: {metric: float(indexed[method][metric]) for metric in METRICS}
        for method in sorted(required)
    }


def load_summary(path: Path, stage: int) -> tuple[dict[str, float], dict[str, float]]:
    summary = load_json(path)
    if (
        summary.get("stage") != stage
        or summary.get("split") != "validation"
        or summary.get("test_split_loaded") is not False
        or summary.get("random_seeds") != [42, 43, 44]
    ):
        raise ValueError(f"Stage {stage} validation summary failed provenance checks")
    seed42 = next(row for row in summary["per_seed"] if row["seed"] == 42)
    seed_metrics = {metric: float(seed42[metric]) for metric in METRICS}
    mean_metrics = {metric: float(summary["aggregate"][metric]["mean"]) for metric in METRICS}
    return seed_metrics, mean_metrics


def lora_parameter_audit(model: Any, expected_modules: int) -> dict[str, Any]:
    named = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter is trainable or no LoRA parameters were created")
    targets = sorted({name.split(".lora_", 1)[0] for name, _ in trainable})
    suffixes = {name.rsplit(".", 1)[-1] for name in targets}
    if len(targets) != expected_modules or suffixes != {"query", "value"}:
        raise RuntimeError(
            f"Expected {expected_modules} query/value targets, found {len(targets)}: {suffixes}"
        )
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    total_count = sum(parameter.numel() for _, parameter in named)
    return {
        "target_module_count": len(targets),
        "target_module_names": targets,
        "trainable_parameters": int(trainable_count),
        "total_parameters": int(total_count),
        "trainable_percent": 100.0 * trainable_count / total_count,
    }


def checkpoint_audit(path: Path) -> dict[str, Any]:
    files = sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file())
    adapter_weights = [
        name for name in files if name.endswith(("adapter_model.safetensors", "adapter_model.bin"))
    ]
    forbidden_full_weights = [
        name for name in files if name.endswith(("model.safetensors", "pytorch_model.bin"))
        and not name.endswith(("adapter_model.safetensors", "adapter_model.bin"))
    ]
    if not adapter_weights or forbidden_full_weights:
        raise RuntimeError(
            f"Adapter-only checkpoint verification failed: adapters={adapter_weights}, "
            f"full_weights={forbidden_full_weights}"
        )
    return {
        "adapter_only": True,
        "adapter_weight_files": adapter_weights,
        "forbidden_full_weight_files": forbidden_full_weights,
        "all_files": files,
        "size_bytes": directory_size(path),
    }


def write_comparison(
    path: Path,
    stage2: dict[str, dict[str, float]],
    stage3_seed42: dict[str, float],
    stage3_mean: dict[str, float],
    stage4_seed42: dict[str, float],
    stage4_mean: dict[str, float],
    stage5: dict[str, float],
    stage5_method: str,
) -> None:
    rows = (
        ("stage2_bm25", stage2["bm25"]),
        ("stage2_pretrained_e5_small_v2", stage2["pretrained_e5_small_v2"]),
        ("stage3_full_ft_inbatch_seed42", stage3_seed42),
        ("stage3_full_ft_inbatch_3seed_mean", stage3_mean),
        ("stage4_full_ft_hardneg_seed42", stage4_seed42),
        ("stage4_full_ft_hardneg_3seed_mean", stage4_mean),
        (stage5_method, stage5),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "split", *METRICS])
        writer.writeheader()
        for method, metrics in rows:
            writer.writerow(
                {"method": method, "split": "validation", **{k: f"{metrics[k]:.6f}" for k in METRICS}}
            )
    temporary.replace(path)


def run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    corpus, train_queries, train_qrels, validation_queries, validation_qrels = load_stage3_data(data_dir)
    validate_stage3_data(corpus, train_queries, train_qrels, validation_queries, validation_qrels)
    model_config = config["model"]
    training_config = config["training"]
    seed = int(training_config["random_seed"])
    pairs = build_training_pairs(
        corpus,
        train_queries,
        train_qrels,
        model_config["query_prefix"],
        model_config["passage_prefix"],
    )
    is_evaluation_run = args.mode in {"pilot", "final"}
    is_lora = args.mode != "profile-full"
    if not is_evaluation_run:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[: int(config["smoke_test"]["max_training_examples"])]
    if args.validate_only:
        return {
            "validated": True,
            "mode": args.mode,
            "training_pairs": len(pairs),
            "validation_queries": len(validation_queries),
            "test_split_loaded": False,
        }

    import faiss
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
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
        raise RuntimeError("Stage 5 requires the approved free Colab GPU runtime")
    set_seed(seed)
    model = SentenceTransformer(
        model_config["name"], revision=model_config["revision"], device="cuda"
    )
    model.max_seq_length = int(model_config["max_sequence_length"])
    model.similarity_fn_name = SimilarityFunction.COSINE
    resolved_revision = resolve_model_revision(model)
    if resolved_revision != model_config["revision"]:
        raise RuntimeError("Resolved E5 revision differs from the approved config")

    adapter_audit = None
    if is_lora:
        lora = config["lora"]
        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias=str(lora["bias"]),
            )
        )
        adapter_audit = lora_parameter_audit(model, int(lora["expected_target_module_count"]))

    train_dataset = Dataset.from_dict(
        {"anchor": [row["anchor"] for row in pairs], "positive": [row["positive"] for row in pairs]}
    )
    evaluator = None
    callbacks = None
    best_metric_key = "eval_scifact_validation_cosine_ndcg@10"
    if is_evaluation_run:
        evaluator = InformationRetrievalEvaluator(
            queries={str(q["id"]): model_config["query_prefix"] + str(q["text"]) for q in validation_queries},
            corpus={str(d["id"]): model_config["passage_prefix"] + join_passage(d) for d in corpus},
            relevant_docs={qid: set(relevant) for qid, relevant in validation_qrels.items()},
            corpus_chunk_size=len(corpus),
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1, 5, 10],
            precision_recall_at_k=[5, 10],
            map_at_k=[10],
            show_progress_bar=True,
            batch_size=int(config["evaluation"]["corpus_batch_size"]),
            name="scifact_validation",
            write_csv=True,
            main_score_function=SimilarityFunction.COSINE,
        )
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=int(training_config["early_stopping_rule"]["patience_evaluations"]),
                early_stopping_threshold=float(training_config["early_stopping_rule"]["minimum_improvement"]),
            )
        ]

    learning_rate = (
        float(training_config["learning_rate"])
        if is_lora
        else float(config["full_finetuning_reference"]["learning_rate"])
    )
    checkpoints_dir = output_dir / "checkpoints"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3.0,
        max_steps=(-1 if is_evaluation_run else 3),
        learning_rate=learning_rate,
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
        run_name=f"stage5-{args.mode}-seed{seed}",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
        callbacks=callbacks,
    )
    resume = latest_resumable_checkpoint(checkpoints_dir) if is_evaluation_run else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if "out of memory" in str(error).casefold():
            print("STAGE5_OOM_AT_BATCH_32_STOPPED_NO_CONFIG_CHANGE", file=sys.stderr)
        raise
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
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
    result_name = {
        "profile-full": "stage5_full_ft_profile_result.json",
        "smoke": "stage5_lora_smoke_result.json",
        "pilot": "stage5_lora_pilot_result.json",
        "final": "stage5_lora_final_result.json",
    }[args.mode]
    results_dir = output_dir / "results"
    write_loss_curve(results_dir / result_name.replace("_result.json", "_loss_curve.csv"), loss_points)
    common = {
        "stage": 5,
        "mode": args.mode,
        "method": (config["method"] if is_lora else "stage3_full_ft_matched_memory_profile"),
        "config": config,
        "resolved_base_model_revision": resolved_revision,
        "training_pairs": len(pairs),
        "validation_queries": len(validation_queries),
        "corpus_size": len(corpus),
        "test_split_loaded": False,
        "global_steps_completed": int(trainer.state.global_step),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "training_seconds": round(elapsed, 3),
        "peak_gpu_memory_bytes": peak_memory,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "loss_summary": summarise_loss(loss_points),
        "adapter_parameter_audit": adapter_audit,
        "package_versions": package_versions(),
        "resumed_from_checkpoint": str(resume) if resume else None,
    }
    if not is_evaluation_run:
        if trainer.state.global_step != 3:
            raise RuntimeError("Efficiency/smoke run did not complete exactly three steps")
        result = {**common, "passed": True, "validation_run": False, "checkpoint_saved": False}
        atomic_write_json(results_dir / result_name, result)
        marker = "STAGE5_LORA_SMOKE_TEST_PASSED" if is_lora else "STAGE3_FULL_FT_PROFILE_PASSED"
        print(marker)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Stage 5 pilot did not select a validation checkpoint")
    best_adapter = output_dir / "best_adapter"
    model.save_pretrained(str(best_adapter))
    adapter_checkpoint = checkpoint_audit(best_adapter)

    corpus_ids = [str(document["id"]) for document in corpus]
    corpus_embeddings = model.encode(
        [model_config["passage_prefix"] + join_passage(document) for document in corpus],
        batch_size=int(config["evaluation"]["corpus_batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    corpus_embeddings = np.ascontiguousarray(corpus_embeddings)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    cache_dir = output_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(cache_dir / "corpus_embeddings.npy", corpus_embeddings)
    atomic_write_json(cache_dir / "corpus_ids.json", {"corpus_ids": corpus_ids})
    temporary_index = cache_dir / f"stage5_lora_seed{seed}.index.faiss.tmp"
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(cache_dir / f"stage5_lora_seed{seed}.index.faiss")
    query_embeddings = model.encode(
        [model_config["query_prefix"] + str(query["text"]) for query in validation_queries],
        batch_size=int(config["evaluation"]["query_batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    _, positions = index.search(np.ascontiguousarray(query_embeddings), 10)
    rankings = {
        str(query["id"]): [corpus_ids[position] for position in row]
        for query, row in zip(validation_queries, positions)
    }
    stage5_metrics = evaluate_rankings(rankings, validation_qrels)
    stage2 = load_stage2_validation(args.stage2_validation.resolve())
    stage3_seed42, stage3_mean = load_summary(args.stage3_summary.resolve(), 3)
    stage4_seed42, stage4_mean = load_summary(args.stage4_summary.resolve(), 4)
    comparison_path = results_dir / f"stage5_{args.mode}_validation_comparison.csv"
    write_comparison(
        comparison_path,
        stage2,
        stage3_seed42,
        stage3_mean,
        stage4_seed42,
        stage4_mean,
        stage5_metrics,
        stage5_method=f"stage5_lora_inbatch_seed{seed}_{args.mode}",
    )

    full_profile = load_json(
        output_dir.parent / "full_ft_profile_seed42" / "results" / "stage5_full_ft_profile_result.json"
    )
    lora_smoke = load_json(
        output_dir.parent / "smoke_seed42" / "results" / "stage5_lora_smoke_result.json"
    )
    stage3_result = load_json(args.stage3_result.resolve())
    if stage3_result.get("test_split_loaded") is not False:
        raise ValueError("Stage 3 seed-42 reference failed provenance checks")
    full_checkpoint = args.stage3_checkpoint.resolve()
    if not full_checkpoint.is_dir():
        raise FileNotFoundError(f"Missing Stage 3 full checkpoint: {full_checkpoint}")
    efficiency = {
        "memory_measurement": "matched separate-process three-step training smokes",
        "full_finetuning": {
            "reference": "stage3_inbatch_seed42",
            "trainable_parameters": int(full_profile["trainable_parameters"]),
            "total_parameters": int(full_profile["total_parameters"]),
            "trainable_percent": 100.0 * full_profile["trainable_parameters"] / full_profile["total_parameters"],
            "peak_gpu_memory_bytes": int(full_profile["peak_gpu_memory_bytes"]),
            "checkpoint_size_bytes": directory_size(full_checkpoint),
            "training_seconds": float(stage3_result["training_seconds"]),
        },
        "lora": {
            "trainable_parameters": int(common["trainable_parameters"]),
            "total_parameters": int(common["total_parameters"]),
            "trainable_percent": 100.0 * common["trainable_parameters"] / common["total_parameters"],
            "peak_gpu_memory_bytes": int(lora_smoke["peak_gpu_memory_bytes"]),
            "checkpoint_size_bytes": int(adapter_checkpoint["size_bytes"]),
            "training_seconds": float(common["training_seconds"]),
            "adapter_only_checkpoint": adapter_checkpoint,
        },
    }
    efficiency["ratios"] = {
        "trainable_parameters_lora_over_full": efficiency["lora"]["trainable_parameters"] / efficiency["full_finetuning"]["trainable_parameters"],
        "peak_gpu_memory_lora_over_full": efficiency["lora"]["peak_gpu_memory_bytes"] / efficiency["full_finetuning"]["peak_gpu_memory_bytes"],
        "checkpoint_size_lora_over_full": efficiency["lora"]["checkpoint_size_bytes"] / efficiency["full_finetuning"]["checkpoint_size_bytes"],
        "training_time_lora_over_full": efficiency["lora"]["training_seconds"] / efficiency["full_finetuning"]["training_seconds"],
    }
    atomic_write_json(results_dir / "stage5_efficiency_comparison.json", efficiency)
    cache_metadata = {
        "adapter_checkpoint": str(best_adapter),
        "adapter_only": True,
        "corpus_sha256": sha256_file(data_dir / "corpus.jsonl"),
        "embedding_count": int(corpus_embeddings.shape[0]),
        "embedding_dimension": int(corpus_embeddings.shape[1]),
        "embedding_dtype": str(corpus_embeddings.dtype),
        "normalize_embeddings": True,
        "faiss_index": type(index).__name__,
        "query_prefix": model_config["query_prefix"],
        "passage_prefix": model_config["passage_prefix"],
        "max_sequence_length": int(model.max_seq_length),
    }
    atomic_write_json(cache_dir / "cache_metadata.json", cache_metadata)
    result = {
        **common,
        "best_checkpoint_metric": "validation_ndcg_at_10",
        "best_checkpoint_source": str(trainer.state.best_model_checkpoint),
        "saved_best_adapter": str(best_adapter),
        "adapter_checkpoint_audit": adapter_checkpoint,
        "validation_evaluations": [entry for entry in trainer.state.log_history if best_metric_key in entry],
        "validation_metrics": stage5_metrics,
        "stage2_validation_metrics": stage2,
        "stage3_seed42_validation_metrics": stage3_seed42,
        "stage3_three_seed_mean_validation_metrics": stage3_mean,
        "stage4_seed42_validation_metrics": stage4_seed42,
        "stage4_three_seed_mean_validation_metrics": stage4_mean,
        "comparison_table": str(comparison_path),
        "efficiency_comparison": efficiency,
        "corpus_cache": cache_metadata,
    }
    atomic_write_json(results_dir / result_name, result)
    print("STAGE5_LORA_PILOT_COMPLETE")
    print(comparison_path.read_text(encoding="utf-8").rstrip())
    print(json.dumps(efficiency, indent=2, sort_keys=True))
    return result


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validate_config(config)
    result = run(config, args)
    if args.validate_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
