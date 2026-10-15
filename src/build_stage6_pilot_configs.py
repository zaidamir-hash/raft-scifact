#!/usr/bin/env python3
"""Create the approved Stage 6 pilot configs from a verified subset manifest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from evaluate_retrieval import atomic_write_json, load_json


FRACTIONS = (10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--stage3-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset_manifest = load_json(args.subset_manifest.resolve())
    stage3 = load_json(args.stage3_config.resolve())
    if subset_manifest.get("stage") != 6 or subset_manifest.get("subset_seed") != 42:
        raise ValueError("Unexpected Stage 6 subset manifest")
    if stage3.get("stage") != 3 or stage3.get("method") != "full_finetune_inbatch_negatives":
        raise ValueError("Unexpected frozen Stage 3 reference config")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths: dict[str, str] = {}
    for percentage in FRACTIONS:
        subset = subset_manifest["fractions"][str(percentage)]
        training = copy.deepcopy(stage3["training"])
        if percentage < 100:
            training["evaluation_frequency"] = "scaled_per_fraction_steps"
            training["evaluation_steps"] = int(subset["evaluation_steps"])
            training["save_frequency"] = "scaled_per_fraction_steps"
            training["save_steps"] = int(subset["save_steps"])
            run_type = "single_seed_fraction_pilot"
        else:
            run_type = "reused_100_percent_stage3_reference"

        config = {
            "stage": 6,
            "run_type": run_type,
            "method": "data_efficiency_full_finetune_inbatch",
            "approved_by_user": True,
            "approved_date": "2026-09-29",
            "base_reference": {
                "stage": 3,
                "config": "configs/stage3_inbatch_final_seed42.json",
                "method": "full_finetune_inbatch_negatives",
                "reason": "Clean labeled-data dose response without full-data dense-mining dependence",
            },
            "dataset": {
                "name": "scifact",
                "source": "BEIR",
                "train_split": "train",
                "checkpoint_selection_split": "validation",
                "test_evaluation": False,
            },
            "model": copy.deepcopy(stage3["model"]),
            "subset": {
                "subset_seed": 42,
                "grouping_rule": subset_manifest["grouping_rule"],
                "nested_fractions_percent": [10, 25, 50, 100],
                "fraction_definition": "query_count",
                "target_fraction_percent": percentage,
                "target_query_count": int(subset["target_query_count"]),
                "realized_query_count": int(subset["realized_query_count"]),
                "realized_fraction": float(subset["realized_fraction"]),
                "training_pair_count": int(subset["training_pair_count"]),
                "component_count": int(subset["component_count"]),
                "steps_per_epoch": int(subset["steps_per_epoch"]),
                "evaluation_steps": int(subset["evaluation_steps"]),
                "save_steps": int(subset["save_steps"]),
                "queries_train_sha256": subset["queries_train_sha256"],
                "qrels_train_sha256": subset["qrels_train_sha256"],
                "train_validation_relevant_document_overlap": 0,
            },
            "training": training,
            "smoke_test": {
                "fraction_percent": 10,
                "max_steps": 3,
                "max_training_examples": 96,
                "run_validation": False,
                "stop_on_out_of_memory": True,
                "allow_automatic_batch_size_change": False,
                "allow_automatic_gradient_accumulation_change": False,
                "reuse_for_larger_fraction_pilots": percentage != 10,
            },
            "evaluation": {
                **copy.deepcopy(stage3["evaluation"]),
                "comparison_rows": [
                    "stage2_bm25_validation",
                    "stage2_pretrained_e5_validation",
                    "stage3_three_seed_mean_validation",
                    "stage4_three_seed_mean_validation",
                    "stage5_three_seed_mean_validation",
                ],
            },
            "execution": {
                "train_this_pilot": percentage < 100,
                "reuse_verified_stage3_result": percentage == 100,
                "reuse_requires_exact_training_config": True,
                "reuse_requires_original_stage1_train_hashes": True,
            },
            "artifacts": {
                "drive_root": "/content/drive/MyDrive/RAFT",
                "stage1_data": "stage1_scifact/processed",
                "subset_root": "stage6_data_efficiency/subsets",
                "smoke_output": "stage6_data_efficiency/smoke_fraction10_seed42",
                "pilot_output": f"stage6_data_efficiency/pilot_fraction{percentage}_seed42",
                "best_checkpoint": f"stage6_data_efficiency/pilot_fraction{percentage}_seed42/best_checkpoint",
                "corpus_cache": f"stage6_data_efficiency/pilot_fraction{percentage}_seed42/embeddings",
                "results": f"stage6_data_efficiency/pilot_fraction{percentage}_seed42/results",
            },
        }
        path = output_dir / f"stage6_data_efficiency_pilot_{percentage}.json"
        atomic_write_json(path, config)
        config_paths[str(percentage)] = f"configs/{path.name}"

    manifest = {
        "stage": 6,
        "phase": "single_seed_pilots",
        "approved_by_user": True,
        "approved_date": "2026-09-29",
        "base_config": "configs/stage3_inbatch_final_seed42.json",
        "subset_seed": 42,
        "training_seed": 42,
        "fractions_percent": [10, 25, 50, 100],
        "trained_pilot_fractions_percent": [10, 25, 50],
        "reused_reference_fraction_percent": 100,
        "config_files": config_paths,
        "test_evaluation": False,
        "final_three_seed_sweep": False,
    }
    atomic_write_json(output_dir / "stage6_data_efficiency_pilot_manifest.json", manifest)
    print("STAGE6_PILOT_CONFIGS_CREATED")
    for percentage in FRACTIONS:
        print(output_dir / f"stage6_data_efficiency_pilot_{percentage}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
