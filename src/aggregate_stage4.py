#!/usr/bin/env python3
"""Aggregate the frozen Stage 4 three-seed validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest.resolve())
    seeds = [int(seed) for seed in manifest["random_seeds"]]
    if seeds != [42, 43, 44]:
        raise ValueError("Frozen Stage 4 seeds must be exactly [42, 43, 44]")
    per_seed: list[dict[str, Any]] = []
    stage2: dict[str, float] | None = None
    stage3_seed42: dict[str, float] | None = None
    stage3_mean: dict[str, float] | None = None
    mining_sha256: str | None = None
    mining_signature: str | None = None
    for seed in seeds:
        result_path = args.drive_root.resolve() / manifest["result_files"][str(seed)]
        result = load_json(result_path)
        if (
            result.get("stage") != 4
            or result.get("mode") != "final"
            or result.get("test_split_loaded") is not False
            or result.get("fresh_pretrained_initialization") is not True
            or result.get("config", {}).get("training", {}).get("random_seed") != seed
            or result.get("config", {}).get("training", {}).get(
                "early_stopping_rule", {}
            ).get("patience_evaluations") != 2
        ):
            raise ValueError(f"Result/config mismatch for Stage 4 seed {seed}")
        current_mining_sha256 = result["hard_negatives_sha256"]
        current_mining_signature = result["mining_config_signature"]
        if mining_sha256 is None:
            mining_sha256 = current_mining_sha256
            mining_signature = current_mining_signature
        elif (
            current_mining_sha256 != mining_sha256
            or current_mining_signature != mining_signature
        ):
            raise ValueError("Stage 4 seeds did not use the exact same mined negatives")
        current_stage2 = {
            metric: float(result["stage2_pretrained_validation_metrics"][metric])
            for metric in METRICS
        }
        current_stage3_seed42 = {
            metric: float(result["stage3_seed42_validation_metrics"][metric])
            for metric in METRICS
        }
        current_stage3_mean = {
            metric: float(result["stage3_three_seed_mean_validation_metrics"][metric])
            for metric in METRICS
        }
        if stage2 is None:
            stage2 = current_stage2
            stage3_seed42 = current_stage3_seed42
            stage3_mean = current_stage3_mean
        elif (
            current_stage2 != stage2
            or current_stage3_seed42 != stage3_seed42
            or current_stage3_mean != stage3_mean
        ):
            raise ValueError("Reference validation metrics differ across Stage 4 seeds")
        per_seed.append(
            {
                "seed": seed,
                **{
                    metric: float(result["validation_metrics"][metric])
                    for metric in METRICS
                },
                "global_steps": int(result["global_steps_completed"]),
                "epochs_completed": float(result["epochs_completed"]),
                "best_checkpoint": result["saved_best_checkpoint"],
                "corpus_cache": str(
                    args.drive_root.resolve()
                    / manifest["output_directories"][str(seed)]
                    / "embeddings"
                ),
            }
        )

    bootstrap = manifest["confidence_interval"]
    if bootstrap["method"] != "percentile_bootstrap_over_seeds":
        raise ValueError("Unsupported Stage 4 confidence-interval method")
    confidence = float(bootstrap["confidence_level"])
    iterations = int(bootstrap["iterations"])
    rng = np.random.default_rng(int(bootstrap["random_seed"]))
    alpha = 1.0 - confidence
    aggregate: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in per_seed], dtype=np.float64)
        sampled = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        lower, upper = np.quantile(sampled, [alpha / 2.0, 1.0 - alpha / 2.0])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "ci95_half_width": float((upper - lower) / 2.0),
        }

    assert stage2 is not None and stage3_seed42 is not None and stage3_mean is not None
    output_dir = args.output_dir.resolve()
    per_seed_path = output_dir / "stage4_final_per_seed_validation.csv"
    aggregate_path = output_dir / "stage4_final_aggregate_validation.csv"
    write_csv(
        per_seed_path,
        ["seed", *METRICS, "global_steps", "epochs_completed", "best_checkpoint", "corpus_cache"],
        per_seed,
    )
    write_csv(
        aggregate_path,
        [
            "metric",
            "mean",
            "sample_standard_deviation",
            "ci95_lower",
            "ci95_upper",
            "ci95_half_width",
            "stage2_pretrained_validation",
            "stage3_seed42_validation",
            "stage3_three_seed_mean_validation",
        ],
        [
            {
                "metric": metric,
                **values,
                "stage2_pretrained_validation": stage2[metric],
                "stage3_seed42_validation": stage3_seed42[metric],
                "stage3_three_seed_mean_validation": stage3_mean[metric],
            }
            for metric, values in aggregate.items()
        ],
    )
    audit_path = output_dir / "stage4_final_seed_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Stage 4 distinctness audit must run before aggregation")
    audit = load_json(audit_path)
    if not (
        audit.get("all_loss_hashes_distinct") is True
        and audit.get("all_embedding_hashes_distinct") is True
        and audit.get("all_model_hashes_distinct") is True
        and audit.get("test_split_loaded") is False
    ):
        raise ValueError("Stage 4 distinctness audit did not pass")
    summary = {
        "stage": 4,
        "method": "full_finetune_mined_hard_negatives",
        "split": "validation",
        "random_seeds": seeds,
        "test_split_loaded": False,
        "fresh_pretrained_initialization": True,
        "early_stopping_patience_evaluations": 2,
        "hard_negatives_sha256": mining_sha256,
        "mining_config_signature": mining_signature,
        "confidence_interval": bootstrap,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "stage2_pretrained_validation": stage2,
        "stage3_seed42_validation": stage3_seed42,
        "stage3_three_seed_mean_validation": stage3_mean,
        "distinctness_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "passed": True,
        },
        "files": {
            "per_seed_csv": str(per_seed_path),
            "aggregate_csv": str(aggregate_path),
        },
    }
    summary_path = output_dir / "stage4_final_summary.json"
    atomic_write_json(summary_path, summary)
    print("STAGE4_FINAL_AGGREGATION_COMPLETE")
    print(per_seed_path.read_text(encoding="utf-8").rstrip())
    print(aggregate_path.read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
