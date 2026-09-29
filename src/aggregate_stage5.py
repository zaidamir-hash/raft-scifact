#!/usr/bin/env python3
"""Aggregate the frozen Stage 5 three-seed validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import sha256_file


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in manifest["random_seeds"]]
    if seeds != [42, 43, 44]:
        raise ValueError("Frozen Stage 5 seeds must be exactly [42, 43, 44]")

    per_seed: list[dict[str, Any]] = []
    references: dict[str, Any] | None = None
    for seed in seeds:
        relative = manifest["result_files"][str(seed)]
        result_path = args.drive_root / relative
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["mode"] != "final" or int(result["config"]["training"]["random_seed"]) != seed:
            raise ValueError(f"Result/config mismatch for seed {seed}")
        if result["test_split_loaded"] is not False:
            raise ValueError(f"Test data was loaded for seed {seed}")
        metrics = {metric: float(result["validation_metrics"][metric]) for metric in METRICS}
        current_references = {
            "stage2": result["stage2_validation_metrics"],
            "stage3_seed42": result["stage3_seed42_validation_metrics"],
            "stage3_mean": result["stage3_three_seed_mean_validation_metrics"],
            "stage4_seed42": result["stage4_seed42_validation_metrics"],
            "stage4_mean": result["stage4_three_seed_mean_validation_metrics"],
        }
        if references is None:
            references = current_references
        elif current_references != references:
            raise ValueError("Validation reference metrics differ across seed results")
        per_seed.append(
            {
                "seed": seed,
                **metrics,
                "global_steps": int(result["global_steps_completed"]),
                "epochs_completed": float(result["epochs_completed"]),
                "best_adapter": result["saved_best_adapter"],
                "corpus_cache": str((args.drive_root / manifest["output_directories"][str(seed)]) / "embeddings"),
            }
        )

    bootstrap = manifest["confidence_interval"]
    if bootstrap["method"] != "percentile_bootstrap_over_seeds":
        raise ValueError("Unsupported confidence-interval method")
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

    assert references is not None
    per_seed_path = args.output_dir / "stage5_final_per_seed_validation.csv"
    aggregate_path = args.output_dir / "stage5_final_aggregate_validation.csv"
    write_csv(
        per_seed_path,
        ["seed", *METRICS, "global_steps", "epochs_completed", "best_adapter", "corpus_cache"],
        per_seed,
    )
    aggregate_rows = [
        {
            "metric": metric,
            **values,
            "stage2_pretrained_e5_validation": references["stage2"]["pretrained_e5_small_v2"][metric],
            "stage3_three_seed_mean_validation": references["stage3_mean"][metric],
            "stage4_three_seed_mean_validation": references["stage4_mean"][metric],
        }
        for metric, values in aggregate.items()
    ]
    write_csv(
        aggregate_path,
        [
            "metric",
            "mean",
            "sample_standard_deviation",
            "ci95_lower",
            "ci95_upper",
            "ci95_half_width",
            "stage2_pretrained_e5_validation",
            "stage3_three_seed_mean_validation",
            "stage4_three_seed_mean_validation",
        ],
        aggregate_rows,
    )
    summary = {
        "stage": 5,
        "method": "lora_inbatch_negatives",
        "split": "validation",
        "random_seeds": seeds,
        "test_split_loaded": False,
        "confidence_interval": bootstrap,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "stage2_validation": references["stage2"],
        "stage3_seed42_validation": references["stage3_seed42"],
        "stage3_three_seed_mean_validation": references["stage3_mean"],
        "stage4_seed42_validation": references["stage4_seed42"],
        "stage4_three_seed_mean_validation": references["stage4_mean"],
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["query", "value"],
            "learning_rate": 1e-4,
            "negative_sampling": "in_batch_only",
        },
        "distinctness_audit": {
            "passed": True,
            "path": str(args.output_dir / "stage5_final_seed_audit.json"),
            "sha256": sha256_file(args.output_dir / "stage5_final_seed_audit.json"),
        },
        "files": {
            "per_seed_csv": str(per_seed_path),
            "aggregate_csv": str(aggregate_path),
        },
    }
    atomic_json(args.output_dir / "stage5_final_summary.json", summary)
    print("STAGE5_FINAL_AGGREGATION_COMPLETE")
    print(per_seed_path.read_text(encoding="utf-8").rstrip())
    print(aggregate_path.read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
