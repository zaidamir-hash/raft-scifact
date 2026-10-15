#!/usr/bin/env python3
"""Aggregate Stage 6 seed-42 fraction pilots and the verified 100% reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_retrieval import atomic_write_json, load_json


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
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metric_values(values: dict[str, Any]) -> dict[str, float]:
    return {metric: float(values[metric]) for metric in METRICS}


def main() -> int:
    args = parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    stage2_rows = list(
        csv.DictReader((repo / "results/stage2_validation_baselines.csv").open(
            "r", encoding="utf-8", newline=""
        ))
    )
    comparison: list[dict[str, Any]] = []
    for method in ("bm25", "pretrained_e5_small_v2"):
        matches = [row for row in stage2_rows if row["method"] == method]
        if len(matches) != 1 or matches[0]["split"] != "validation":
            raise ValueError(f"Missing Stage 2 validation baseline: {method}")
        comparison.append(
            {"method": f"stage2_{method}", "split": "validation", **metric_values(matches[0])}
        )

    stage_summaries = {
        3: load_json(repo / "results/stage3_final_summary.json"),
        4: load_json(repo / "results/stage4_final_summary.json"),
        5: load_json(repo / "results/stage5_final_summary.json"),
    }
    names = {
        3: "stage3_full_ft_inbatch_3seed_mean",
        4: "stage4_full_ft_hardneg_3seed_mean",
        5: "stage5_lora_inbatch_3seed_mean",
    }
    for stage, summary in stage_summaries.items():
        if summary.get("test_split_loaded") is not False:
            raise ValueError(f"Stage {stage} reference loaded test data")
        comparison.append(
            {
                "method": names[stage],
                "split": "validation",
                **{
                    metric: float(summary["aggregate"][metric]["mean"])
                    for metric in METRICS
                },
            }
        )

    fraction_rows: list[dict[str, Any]] = []
    for percentage in (10, 25, 50):
        run_dir = drive / "stage6_data_efficiency" / f"pilot_fraction{percentage}_seed42"
        result = load_json(
            run_dir / "results" / f"stage6_fraction{percentage}_pilot_result.json"
        )
        if (
            result.get("stage") != 6
            or result.get("mode") != "pilot"
            or result.get("test_split_loaded") is not False
            or int(result.get("target_fraction_percent", -1)) != percentage
            or int(result["config"]["training"]["random_seed"]) != 42
        ):
            raise ValueError(f"Invalid Stage 6 pilot result for {percentage}%")
        fraction_rows.append(
            {
                "fraction_percent": percentage,
                "realized_fraction": float(result["realized_fraction"]),
                "training_queries": int(result["training_queries"]),
                "training_pairs": int(result["training_pairs"]),
                **metric_values(result["validation_metrics"]),
                "global_steps": int(result["global_steps_completed"]),
                "epochs_completed": float(result["epochs_completed"]),
                "reused_stage3_result": False,
                "best_checkpoint": result["saved_best_checkpoint"],
                "corpus_cache": str(run_dir / "embeddings"),
            }
        )

    equivalence = load_json(output / "stage6_100_percent_equivalence.json")
    if equivalence.get("passed") is not True or equivalence.get("test_split_loaded") is not False:
        raise ValueError("The Stage 3 100% equivalence proof did not pass")
    stage3_seed42 = [
        row for row in stage_summaries[3]["per_seed"] if int(row["seed"]) == 42
    ]
    if len(stage3_seed42) != 1:
        raise ValueError("Missing Stage 3 seed-42 result")
    source = stage3_seed42[0]
    fraction_rows.append(
        {
            "fraction_percent": 100,
            "realized_fraction": 1.0,
            "training_queries": int(equivalence["training_queries"]),
            "training_pairs": int(equivalence["training_pairs"]),
            **metric_values(source),
            "global_steps": int(source["global_steps"]),
            "epochs_completed": float(source["epochs_completed"]),
            "reused_stage3_result": True,
            "best_checkpoint": source["best_checkpoint"],
            "corpus_cache": source["corpus_cache"],
        }
    )
    fraction_rows.sort(key=lambda row: int(row["fraction_percent"]))
    for row in fraction_rows:
        comparison.append(
            {
                "method": f"stage6_fraction{row['fraction_percent']}_seed42_pilot",
                "split": "validation",
                **{metric: row[metric] for metric in METRICS},
            }
        )

    per_fraction_path = output / "stage6_pilot_per_fraction_validation.csv"
    comparison_path = output / "stage6_pilot_comparison_validation.csv"
    write_csv(per_fraction_path, fraction_rows)
    write_csv(comparison_path, comparison)
    summary = {
        "stage": 6,
        "phase": "single_seed_pilots",
        "base_method": "stage3_full_finetune_inbatch_negatives",
        "training_seed": 42,
        "subset_seed": 42,
        "fractions_percent": [10, 25, 50, 100],
        "per_fraction": fraction_rows,
        "stage2_validation_baselines": comparison[:2],
        "stage3_stage4_stage5_context": comparison[2:5],
        "hundred_percent_equivalence": equivalence,
        "test_split_loaded": False,
        "final_three_seed_sweep_run": False,
        "files": {
            "per_fraction_csv": str(per_fraction_path),
            "comparison_csv": str(comparison_path),
        },
    }
    atomic_write_json(output / "stage6_pilot_summary.json", summary)
    print("STAGE6_SEED42_PILOTS_AGGREGATED")
    print(per_fraction_path.read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
