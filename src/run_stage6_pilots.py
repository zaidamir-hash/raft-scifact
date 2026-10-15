#!/usr/bin/env python3
"""Run Stage 6 subset creation, smoke test, and seed-42 fraction pilots."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def verified_result(path: Path, percentage: int) -> bool:
    if not path.is_file():
        return False
    result = load_json(path)
    return (
        result.get("stage") == 6
        and result.get("mode") == "pilot"
        and result.get("test_split_loaded") is False
        and int(result.get("target_fraction_percent", -1)) == percentage
        and int(result.get("config", {}).get("training", {}).get("random_seed", -1)) == 42
        and Path(result.get("saved_best_checkpoint", "")).is_dir()
    )


def main() -> int:
    args = parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    data_dir = drive / "stage1_scifact" / "processed"
    subset_root = drive / "stage6_data_efficiency" / "subsets"
    final_results = drive / "stage6_data_efficiency" / "pilot_results"
    references_dir = repo / "results"
    baseline_results = references_dir / "stage2_validation_baselines.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo / "src/prepare_data_efficiency_subsets.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(subset_root),
            "--subset-seed", "42",
            "--batch-size", "32",
        ],
        check=True,
    )
    subset_manifest = load_json(subset_root / "stage6_subset_manifest.json")
    if subset_manifest.get("test_split_loaded") is not False:
        raise RuntimeError("Subset preparation loaded test data")

    for percentage in (10, 25, 50, 100):
        config = load_json(repo / f"configs/stage6_data_efficiency_pilot_{percentage}.json")
        subset = subset_manifest["fractions"][str(percentage)]
        for key in (
            "realized_query_count",
            "training_pair_count",
            "evaluation_steps",
            "save_steps",
            "queries_train_sha256",
            "qrels_train_sha256",
        ):
            if config["subset"][key] != subset[key]:
                raise RuntimeError(f"Fraction {percentage} config/subset mismatch: {key}")
    print("STAGE6_SUBSET_CONFIG_AUDIT_PASSED")

    stage3_config = load_json(repo / "configs/stage3_inbatch_final_seed42.json")
    config100 = load_json(repo / "configs/stage6_data_efficiency_pilot_100.json")
    training100 = copy.deepcopy(config100["training"])
    if training100 != stage3_config["training"]:
        raise RuntimeError("The 100% Stage 6 training config differs from frozen Stage 3")
    if config100["model"] != stage3_config["model"]:
        raise RuntimeError("The 100% Stage 6 model config differs from frozen Stage 3")
    full_subset = subset_manifest["fractions"]["100"]
    if full_subset["queries_train_sha256"] != sha256_file(data_dir / "queries_train.jsonl"):
        raise RuntimeError("The 100% query set differs from Stage 1 train")
    if full_subset["qrels_train_sha256"] != sha256_file(data_dir / "qrels_train.tsv"):
        raise RuntimeError("The 100% qrels differ from Stage 1 train")
    stage3_summary = load_json(repo / "results/stage3_final_summary.json")
    if stage3_summary.get("test_split_loaded") is not False:
        raise RuntimeError("Stage 3 reference loaded test data")
    stage3_result_path = (
        drive / "stage3_inbatch/final_seed42/results/stage3_final_result.json"
    )
    if not stage3_result_path.is_file():
        raise FileNotFoundError(f"Missing frozen Stage 3 seed-42 result: {stage3_result_path}")
    stage3_result = load_json(stage3_result_path)
    if (
        stage3_result.get("stage") != 3
        or stage3_result.get("mode") != "final"
        or stage3_result.get("test_split_loaded") is not False
        or stage3_result.get("config", {}).get("training") != stage3_config["training"]
        or stage3_result.get("config", {}).get("model") != stage3_config["model"]
        or int(stage3_result.get("training_queries", -1))
        != int(full_subset["realized_query_count"])
        or int(stage3_result.get("training_pairs", -1))
        != int(full_subset["training_pair_count"])
        or not Path(stage3_result.get("saved_best_checkpoint", "")).is_dir()
        or not (drive / "stage3_inbatch/final_seed42/embeddings").is_dir()
    ):
        raise RuntimeError("Frozen Stage 3 seed-42 result failed the 100% reuse audit")
    equivalence = {
        "passed": True,
        "stage6_fraction_percent": 100,
        "reused_stage": 3,
        "reused_seed": 42,
        "training_config_identical": True,
        "model_config_identical": True,
        "original_stage1_train_hashes_identical": True,
        "stage3_saved_result_config_identical": True,
        "stage3_checkpoint_and_embedding_cache_usable": True,
        "stage3_result_path": str(stage3_result_path),
        "training_queries": int(full_subset["realized_query_count"]),
        "training_pairs": int(full_subset["training_pair_count"]),
        "queries_train_sha256": full_subset["queries_train_sha256"],
        "qrels_train_sha256": full_subset["qrels_train_sha256"],
        "test_split_loaded": False,
    }
    atomic_write_json(final_results / "stage6_100_percent_equivalence.json", equivalence)
    print("STAGE6_100_PERCENT_STAGE3_EQUIVALENCE_PASSED")

    train_script = repo / "src/train_data_efficiency.py"
    config10 = repo / "configs/stage6_data_efficiency_pilot_10.json"
    smoke_dir = drive / "stage6_data_efficiency" / "smoke_fraction10_seed42"
    smoke_result_path = smoke_dir / "results/stage6_smoke_result.json"
    if smoke_result_path.is_file():
        smoke = load_json(smoke_result_path)
    else:
        print("STARTING_STAGE6_10_PERCENT_SMOKE")
        subprocess.run(
            [
                sys.executable, str(train_script),
                "--mode", "smoke",
                "--config", str(config10),
                "--data-dir", str(data_dir),
                "--subset-dir", str(subset_root / "fraction_10"),
                "--baseline-results", str(baseline_results),
                "--references-dir", str(references_dir),
                "--output-dir", str(smoke_dir),
            ],
            check=True,
        )
        smoke = load_json(smoke_result_path)
    if not (
        smoke.get("passed") is True
        and int(smoke.get("global_steps_completed", -1)) == 3
        and smoke.get("test_split_loaded") is False
        and int(smoke.get("target_fraction_percent", -1)) == 10
        and int(smoke.get("config", {}).get("training", {}).get("batch_size", -1)) == 32
    ):
        raise RuntimeError("Stage 6 10% smoke result failed verification")
    print("STAGE6_SMOKE_TEST_PASSED_AT_BATCH_32")

    for percentage in (10, 25, 50):
        config = repo / f"configs/stage6_data_efficiency_pilot_{percentage}.json"
        output = drive / "stage6_data_efficiency" / f"pilot_fraction{percentage}_seed42"
        result_path = output / "results" / f"stage6_fraction{percentage}_pilot_result.json"
        if verified_result(result_path, percentage):
            print(f"STAGE6_FRACTION_{percentage}_PILOT_ALREADY_COMPLETE")
            continue
        print(f"STARTING_STAGE6_FRACTION_{percentage}_PILOT")
        subprocess.run(
            [
                sys.executable, str(train_script),
                "--mode", "pilot",
                "--config", str(config),
                "--data-dir", str(data_dir),
                "--subset-dir", str(subset_root / f"fraction_{percentage}"),
                "--baseline-results", str(baseline_results),
                "--references-dir", str(references_dir),
                "--output-dir", str(output),
            ],
            check=True,
        )
        if not verified_result(result_path, percentage):
            raise RuntimeError(f"Fraction {percentage} pilot finished without a verified result")

    subprocess.run(
        [
            sys.executable,
            str(repo / "src/aggregate_stage6_pilots.py"),
            "--repo-dir", str(repo),
            "--drive-root", str(drive),
            "--output-dir", str(final_results),
        ],
        check=True,
    )
    summary = final_results / "stage6_pilot_summary.json"
    if not summary.is_file():
        raise RuntimeError("Stage 6 pilot summary was not created")
    print(f"STAGE6_SEED42_PILOTS_COMPLETE={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
