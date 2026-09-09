#!/usr/bin/env python3
"""Run/resume the frozen Stage 4 three-seed validation experiment."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import load_json, sha256_file
from mine_negatives import config_digest
from run_stage4_patience4_pilot import verify_exact_mining_reuse, verify_smoke_reuse
from train_hard_negatives import mining_signature, validate_training_config


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def validate_frozen_configs(repo_dir: Path) -> tuple[dict, dict[int, dict]]:
    pilot = load_json(repo_dir / "configs" / "stage4_hardneg_pilot.json")
    final = {
        seed: load_json(repo_dir / "configs" / f"stage4_hardneg_final_seed{seed}.json")
        for seed in SEEDS
    }
    for seed, config in final.items():
        validate_training_config(config)
        if config.get("frozen_from") != "configs/stage4_hardneg_pilot.json":
            raise ValueError(f"Seed {seed} does not identify the frozen pilot config")
        if config["training"]["random_seed"] != seed:
            raise ValueError(f"Seed {seed} config records a different random seed")
        for section in ("dataset", "model", "mining", "smoke_test"):
            if config[section] != pilot[section]:
                raise ValueError(f"Seed {seed} changed frozen section {section}")
        pilot_eval = copy.deepcopy(pilot["evaluation"])
        final_eval = copy.deepcopy(config["evaluation"])
        pilot_eval.pop("comparison_rows", None)
        final_eval.pop("comparison_rows", None)
        if final_eval != pilot_eval:
            raise ValueError(f"Seed {seed} changed frozen evaluation settings")
        pilot_training = copy.deepcopy(pilot["training"])
        final_training = copy.deepcopy(config["training"])
        pilot_training.pop("random_seed")
        final_training.pop("random_seed")
        if final_training != pilot_training:
            raise ValueError(f"Seed {seed} changed training settings beyond random seed")
        if config["training"]["early_stopping_rule"]["patience_evaluations"] != 2:
            raise ValueError("The frozen Stage 4 final patience must remain two")
        if config.get("pilot_variant") is not None:
            raise ValueError("The patience-4 pilot adjustment leaked into a final config")
    signatures = {mining_signature(config) for config in final.values()}
    if signatures != {mining_signature(pilot)}:
        raise ValueError("Final configs do not share the frozen mining signature")
    print("STAGE4_FROZEN_CONFIG_AUDIT_PASSED")
    return pilot, final


def verified_final(config: dict, result_path: Path, negatives: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
        seed = config["training"]["random_seed"]
        return (
            result.get("stage") == 4
            and result.get("mode") == "final"
            and result.get("config_sha256") == config_digest(config)
            and result.get("mining_config_signature") == mining_signature(config)
            and result.get("hard_negatives_sha256") == sha256_file(negatives)
            and result.get("test_split_loaded") is False
            and result.get("fresh_pretrained_initialization") is True
            and result.get("config", {}).get("training", {}).get("random_seed") == seed
            and result.get("config", {}).get("training", {}).get(
                "early_stopping_rule", {}
            ).get("patience_evaluations") == 2
            and Path(result["saved_best_checkpoint"]).is_dir()
            and (result_path.parents[1] / "embeddings" / "corpus_embeddings.npy").is_file()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    drive_root = args.drive_root.resolve()
    pilot, configs = validate_frozen_configs(repo_dir)
    negatives = drive_root / pilot["artifacts"]["mined_negatives"]
    mining_summary = drive_root / pilot["artifacts"]["mining_summary"]
    smoke_path = (
        drive_root
        / pilot["artifacts"]["smoke_output"]
        / "results"
        / "stage4_smoke_result.json"
    )
    verify_exact_mining_reuse(pilot, negatives, mining_summary)
    print(f"REUSING_EXACT_MINED_NEGATIVES_SHA256={sha256_file(negatives)}")
    verify_smoke_reuse(pilot, smoke_path, negatives)
    print("REUSING_VERIFIED_STAGE4_BATCH32_SMOKE_TEST")

    data_dir = drive_root / pilot["artifacts"]["stage1_data"]
    stage2_results = drive_root / pilot["artifacts"]["stage2_results"]
    stage3_summary = drive_root / pilot["artifacts"]["stage3_summary"]
    train_script = repo_dir / "src" / "train_hard_negatives.py"
    for seed in SEEDS:
        config = configs[seed]
        config_path = repo_dir / "configs" / f"stage4_hardneg_final_seed{seed}.json"
        output_dir = drive_root / config["artifacts"]["final_output"]
        result_path = output_dir / "results" / "stage4_final_result.json"
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--hard-negatives",
            str(negatives),
            "--mining-summary",
            str(mining_summary),
            "--stage2-results",
            str(stage2_results),
            "--stage3-summary",
            str(stage3_summary),
            "--mode",
            "final",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run([*command, "--validate-only"], check=True)
        if verified_final(config, result_path, negatives):
            print(f"REUSING_VERIFIED_STAGE4_FINAL_SEED_{seed}")
            continue
        print(f"STARTING_STAGE4_FINAL_SEED_{seed}")
        subprocess.run(command, check=True)
        if not verified_final(config, result_path, negatives):
            raise RuntimeError(f"Stage 4 seed {seed} finished without verified artifacts")

    manifest = repo_dir / "configs" / "stage4_hardneg_final_manifest.json"
    output_dir = drive_root / "stage4_hardneg" / "final_results"
    subprocess.run(
        [
            sys.executable,
            str(repo_dir / "src" / "audit_stage4.py"),
            "--manifest",
            str(manifest),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_dir / "src" / "aggregate_stage4.py"),
            "--manifest",
            str(manifest),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    summary = output_dir / "stage4_final_summary.json"
    audit = output_dir / "stage4_final_seed_audit.json"
    if not summary.is_file() or not audit.is_file():
        raise RuntimeError("Stage 4 final summary/audit was not created")
    print(f"STAGE4_FINAL_THREE_SEED_VALIDATION_COMPLETE={summary}")
    print(f"STAGE4_FINAL_DISTINCTNESS_AUDIT={audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
