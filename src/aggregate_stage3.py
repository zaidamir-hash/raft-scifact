#!/usr/bin/env python3
"""Aggregate the frozen Stage 3 three-seed validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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
        raise ValueError("Frozen Stage 3 seeds must be exactly [42, 43, 44]")

    per_seed: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    for seed in seeds:
        relative = manifest["result_files"][str(seed)]
        result_path = args.drive_root / relative
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["mode"] != "final" or int(result["config"]["training"]["random_seed"]) != seed:
            raise ValueError(f"Result/config mismatch for seed {seed}")
        if result["test_split_loaded"] is not False:
            raise ValueError(f"Test data was loaded for seed {seed}")
        metrics = {metric: float(result["validation_metrics"][metric]) for metric in METRICS}
        current_baseline = {
            metric: float(result["stage2_pretrained_validation_metrics"][metric])
            for metric in METRICS
        }
        if baseline is None:
            baseline = current_baseline
        elif current_baseline != baseline:
            raise ValueError("Stage 2 validation baseline differs across seed results")
        per_seed.append(
            {
                "seed": seed,
                **metrics,
                "global_steps": int(result["global_steps_completed"]),
                "epochs_completed": float(result["epochs_completed"]),
                "best_checkpoint": result["saved_best_checkpoint"],
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

    assert baseline is not None
    per_seed_path = args.output_dir / "stage3_final_per_seed_validation.csv"
    aggregate_path = args.output_dir / "stage3_final_aggregate_validation.csv"
    write_csv(
        per_seed_path,
        ["seed", *METRICS, "global_steps", "epochs_completed", "best_checkpoint", "corpus_cache"],
        per_seed,
    )
    aggregate_rows = [
        {"metric": metric, **values, "pretrained_e5_validation": baseline[metric]}
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
            "pretrained_e5_validation",
        ],
        aggregate_rows,
    )
    summary = {
        "stage": 3,
        "method": "full_finetune_inbatch_negatives",
        "split": "validation",
        "random_seeds": seeds,
        "test_split_loaded": False,
        "confidence_interval": bootstrap,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "stage2_pretrained_validation": baseline,
        "files": {
            "per_seed_csv": str(per_seed_path),
            "aggregate_csv": str(aggregate_path),
        },
    }
    atomic_json(args.output_dir / "stage3_final_summary.json", summary)
    print("STAGE3_FINAL_AGGREGATION_COMPLETE")
    print(per_seed_path.read_text(encoding="utf-8").rstrip())
    print(aggregate_path.read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
