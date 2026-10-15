#!/usr/bin/env python3
"""Freeze Stage 6 fraction pilots as seed-42/43/44 final configurations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from evaluate_retrieval import atomic_write_json, load_json


FRACTIONS = (10, 25, 50)
SEEDS = (42, 43, 44)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_dir.resolve()
    pilot_manifest = load_json(repo / "configs/stage6_data_efficiency_pilot_manifest.json")
    if (
        pilot_manifest["phase"] != "single_seed_pilots"
        or pilot_manifest["subset_seed"] != 42
        or pilot_manifest["training_seed"] != 42
        or pilot_manifest["test_evaluation"] is not False
    ):
        raise ValueError("Unexpected Stage 6 pilot manifest")

    output_files: dict[str, dict[str, str]] = {}
    for fraction in FRACTIONS:
        pilot = load_json(repo / f"configs/stage6_data_efficiency_pilot_{fraction}.json")
        if pilot["run_type"] != "single_seed_fraction_pilot":
            raise ValueError(f"Fraction {fraction} is not the frozen pilot configuration")
        output_files[str(fraction)] = {}
        for seed in SEEDS:
            config = copy.deepcopy(pilot)
            config["run_type"] = "final_three_seed_fraction"
            config["training"]["random_seed"] = seed
            config["execution"] = {
                "phase": "final_three_seed_validation_only",
                "pilot_config": f"configs/stage6_data_efficiency_pilot_{fraction}.json",
                "only_training_setting_changed_from_pilot": "random_seed",
                "reuse_pilot_training_run": False,
            }
            run_relative = f"stage6_data_efficiency/final_fraction{fraction}_seed{seed}"
            config["artifacts"] = {
                "drive_root": "/content/drive/MyDrive/RAFT",
                "stage1_data": "stage1_scifact/processed",
                "subset_root": "stage6_data_efficiency/subsets",
                "final_output": run_relative,
                "best_checkpoint": f"{run_relative}/best_checkpoint",
                "corpus_cache": f"{run_relative}/embeddings",
                "results": f"{run_relative}/results",
            }
            name = f"stage6_data_efficiency_final_fraction{fraction}_seed{seed}.json"
            atomic_write_json(repo / "configs" / name, config)
            output_files[str(fraction)][str(seed)] = f"configs/{name}"

    manifest = {
        "stage": 6,
        "phase": "final_three_seed_validation_only",
        "approved_by_user": True,
        "pilot_manifest": "configs/stage6_data_efficiency_pilot_manifest.json",
        "subset_seed": 42,
        "random_seeds": list(SEEDS),
        "trained_fractions_percent": list(FRACTIONS),
        "reused_fraction_percent": 100,
        "reused_reference": "Stage 3 frozen seeds 42, 43, 44",
        "confidence_interval": {
            "method": "percentile_bootstrap_over_seeds",
            "confidence_level": 0.95,
            "iterations": 100000,
            "random_seed": 42,
        },
        "config_files": output_files,
        "test_evaluation": False,
        "start_stage_7": False,
    }
    atomic_write_json(repo / "configs/stage6_data_efficiency_final_manifest.json", manifest)
    print("STAGE6_FINAL_CONFIGS_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
