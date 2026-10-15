#!/usr/bin/env python3
"""Aggregate the Stage 6 validation-only three-seed data-efficiency sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


FRACTIONS = (10, 25, 50, 100)
SEEDS = (42, 43, 44)
METRICS = (
    "recall_at_5", "recall_at_10", "precision_at_5", "precision_at_10",
    "mrr_at_10", "ndcg_at_10",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    output = args.output_dir.resolve()
    manifest = load_json(repo / "configs/stage6_data_efficiency_final_manifest.json")
    audit_path = output / "stage6_final_seed_audit.json"
    audit = load_json(audit_path)
    equivalence = load_json(output / "stage6_100_percent_three_seed_equivalence.json")
    embedding_audit_ok = bool(
        audit.get("all_embedding_hashes_distinct_within_fraction")
        or audit.get("known_identical_embedding_seeds") == {"10": [[42, 44]]}
    )
    if (
        audit.get("test_split_loaded") is not False
        or not audit.get("all_loss_hashes_distinct_within_fraction")
        or not embedding_audit_ok
        or not audit.get("all_model_hashes_distinct_within_fraction")
        or equivalence.get("passed") is not True
        or equivalence.get("test_split_loaded") is not False
    ):
        raise RuntimeError("Stage 6 audits must pass before aggregation")
    stage3_summary = load_json(repo / "results/stage3_final_summary.json")
    stage3_by_seed = {int(row["seed"]): row for row in stage3_summary["per_seed"]}
    if stage3_summary.get("test_split_loaded") is not False or set(stage3_by_seed) != set(SEEDS):
        raise RuntimeError("Stage 3 three-seed reference is incomplete")

    with (repo / "results/stage2_validation_baselines.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        baseline_rows = list(csv.DictReader(handle))
    baselines = {
        row["method"]: {metric: float(row[metric]) for metric in METRICS}
        for row in baseline_rows if row["split"] == "validation"
    }
    if set(baselines) != {"bm25", "pretrained_e5_small_v2"}:
        raise RuntimeError("Stage 2 validation baselines are incomplete")

    per_seed: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        fraction_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            if fraction == 100:
                source = stage3_by_seed[seed]
                row = {
                    "fraction_percent": fraction,
                    "realized_fraction": 1.0,
                    "training_queries": int(equivalence["training_queries"]),
                    "training_pairs": int(equivalence["training_pairs"]),
                    "seed": seed,
                    **{metric: float(source[metric]) for metric in METRICS},
                    "global_steps": int(source["global_steps"]),
                    "epochs_completed": float(source["epochs_completed"]),
                    "reused_stage3_result": True,
                    "best_checkpoint": source["best_checkpoint"],
                    "corpus_cache": source["corpus_cache"],
                }
            else:
                run_dir = drive / f"stage6_data_efficiency/final_fraction{fraction}_seed{seed}"
                result = load_json(
                    run_dir / "results" / f"stage6_fraction{fraction}_final_result.json"
                )
                frozen = load_json(
                    repo / f"configs/stage6_data_efficiency_final_fraction{fraction}_seed{seed}.json"
                )
                if (
                    result.get("mode") != "final"
                    or result.get("stage") != 6
                    or result.get("test_split_loaded") is not False
                    or result.get("config") != frozen
                ):
                    raise RuntimeError(f"Fraction {fraction} seed {seed} result invalid")
                row = {
                    "fraction_percent": fraction,
                    "realized_fraction": float(result["realized_fraction"]),
                    "training_queries": int(result["training_queries"]),
                    "training_pairs": int(result["training_pairs"]),
                    "seed": seed,
                    **{metric: float(result["validation_metrics"][metric]) for metric in METRICS},
                    "global_steps": int(result["global_steps_completed"]),
                    "epochs_completed": float(result["epochs_completed"]),
                    "reused_stage3_result": False,
                    "best_checkpoint": result["saved_best_checkpoint"],
                    "corpus_cache": str(run_dir / "embeddings"),
                }
            fraction_rows.append(row)
            per_seed.append(row)
        aggregate[str(fraction)] = {}
        for metric in METRICS:
            if fraction == 100:
                statistics = {
                    key: float(value)
                    for key, value in stage3_summary["aggregate"][metric].items()
                }
            else:
                values = np.asarray([row[metric] for row in fraction_rows], dtype=np.float64)
                bootstrap = manifest["confidence_interval"]
                if (
                    bootstrap["method"] != "percentile_bootstrap_over_seeds"
                    or bootstrap["confidence_level"] != 0.95
                    or int(bootstrap["iterations"]) != 100000
                ):
                    raise RuntimeError("Unexpected bootstrap procedure")
                rng = np.random.default_rng(int(bootstrap["random_seed"]))
                draws = rng.choice(
                    values, size=(int(bootstrap["iterations"]), len(values)), replace=True
                ).mean(axis=1)
                lower, upper = np.quantile(draws, [0.025, 0.975])
                statistics = {
                    "mean": float(values.mean()),
                    "sample_standard_deviation": float(values.std(ddof=1)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "ci95_half_width": float((upper - lower) / 2.0),
                }
            aggregate[str(fraction)][metric] = statistics
            aggregate_rows.append(
                {
                    "fraction_percent": fraction,
                    "metric": metric,
                    **statistics,
                    "stage2_pretrained_e5_validation": baselines["pretrained_e5_small_v2"][metric],
                    "stage2_bm25_validation": baselines["bm25"][metric],
                    "reused_stage3_aggregate": fraction == 100,
                }
            )

    paths = {
        "per_seed_csv": output / "stage6_final_per_seed_validation.csv",
        "aggregate_csv": output / "stage6_final_aggregate_validation.csv",
    }
    write_csv(paths["per_seed_csv"], per_seed)
    write_csv(paths["aggregate_csv"], aggregate_rows)
    reference_means = {}
    for stage in (3, 4, 5):
        summary = load_json(repo / f"results/stage{stage}_final_summary.json")
        if summary.get("test_split_loaded") is not False:
            raise RuntimeError(f"Stage {stage} reference loaded the test split")
        reference_means[str(stage)] = {
            metric: float(summary["aggregate"][metric]["mean"]) for metric in METRICS
        }
    summary = {
        "stage": 6,
        "phase": "final_three_seed_validation_only",
        "method": "full_finetune_inbatch_data_efficiency",
        "split": "validation",
        "random_seeds": list(SEEDS),
        "subset_seed": 42,
        "fractions_percent": list(FRACTIONS),
        "confidence_interval": manifest["confidence_interval"],
        "per_seed": per_seed,
        "aggregate": aggregate,
        "stage2_validation_baselines": baselines,
        "stage3_stage4_stage5_three_seed_means": reference_means,
        "hundred_percent_stage3_reuse": equivalence,
        "distinctness_audit": {
            "passed": True,
            "embedding_uniqueness_exception": audit.get(
                "known_identical_embedding_seeds", {}
            ),
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "test_split_loaded": False,
        "stage7_started": False,
        "files": {key: str(value) for key, value in paths.items()},
    }
    atomic_write_json(output / "stage6_final_summary.json", summary)
    print("STAGE6_FINAL_VALIDATION_AGGREGATED")
    print(paths["per_seed_csv"].read_text(encoding="utf-8").rstrip())
    print(paths["aggregate_csv"].read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
