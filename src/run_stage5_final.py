#!/usr/bin/env python3
"""Run/resume the frozen Stage 5 three-seed validation experiment."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import load_json
from train_lora import validate_config


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def validate_frozen_configs(repo: Path) -> tuple[dict, dict[int, dict]]:
    pilot = load_json(repo / "configs" / "stage5_lora_pilot.json")
    final = {
        seed: load_json(repo / "configs" / f"stage5_lora_final_seed{seed}.json")
        for seed in SEEDS
    }
    for seed, config in final.items():
        validate_config(config)
        if config.get("frozen_from") != "configs/stage5_lora_pilot.json":
            raise ValueError(f"Seed {seed} does not identify the frozen pilot config")
        if config["training"]["random_seed"] != seed:
            raise ValueError(f"Seed {seed} config records a different random seed")
        for section in (
            "dataset",
            "model",
            "lora",
            "smoke_test",
            "efficiency",
            "full_finetuning_reference",
        ):
            if config[section] != pilot[section]:
                raise ValueError(f"Seed {seed} changed frozen section {section}")
        pilot_training = copy.deepcopy(pilot["training"])
        final_training = copy.deepcopy(config["training"])
        pilot_training.pop("random_seed")
        final_training.pop("random_seed")
        if final_training != pilot_training:
            raise ValueError(f"Seed {seed} changed training settings beyond random seed")
        pilot_eval = copy.deepcopy(pilot["evaluation"])
        final_eval = copy.deepcopy(config["evaluation"])
        pilot_eval.pop("comparison_rows", None)
        final_eval.pop("comparison_rows", None)
        if final_eval != pilot_eval:
            raise ValueError(f"Seed {seed} changed frozen evaluation settings")
    print("STAGE5_FROZEN_CONFIG_AUDIT_PASSED")
    return pilot, final


def valid_smoke(path: Path, pilot: dict) -> bool:
    if not path.is_file():
        return False
    result = load_json(path)
    return (
        result.get("stage") == 5
        and result.get("mode") == "smoke"
        and result.get("config") == pilot
        and result.get("passed") is True
        and result.get("global_steps_completed") == 3
        and result.get("test_split_loaded") is False
        and result.get("adapter_parameter_audit", {}).get("target_module_count") == 24
    )


def valid_final(path: Path, config: dict) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
        run_dir = path.parents[1]
        return (
            result.get("stage") == 5
            and result.get("mode") == "final"
            and result.get("config") == config
            and result.get("test_split_loaded") is False
            and result.get("adapter_checkpoint_audit", {}).get("adapter_only") is True
            and Path(result["saved_best_adapter"]).is_dir()
            and (run_dir / "embeddings" / "corpus_embeddings.npy").is_file()
            and (run_dir / "embeddings" / f"stage5_lora_seed{config['training']['random_seed']}.index.faiss").is_file()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    pilot, configs = validate_frozen_configs(repo)
    data = drive / "stage1_scifact" / "processed"
    stage2_validation = repo / "results" / "stage2_validation_baselines.csv"
    stage3_summary = repo / "results" / "stage3_final_summary.json"
    stage4_summary = repo / "results" / "stage4_final_summary.json"
    stage3_result = drive / "stage3_inbatch" / "final_seed42" / "results" / "stage3_final_result.json"
    stage3_checkpoint = drive / "stage3_inbatch" / "final_seed42" / "best_checkpoint"
    smoke_path = drive / "stage5_lora" / "smoke_seed42" / "results" / "stage5_lora_smoke_result.json"
    required = [
        data / "corpus.jsonl",
        data / "queries_train.jsonl",
        data / "qrels_train.tsv",
        data / "queries_validation.jsonl",
        data / "qrels_validation.tsv",
        stage2_validation,
        stage3_summary,
        stage4_summary,
        stage3_result,
        stage3_checkpoint,
        smoke_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing restartable Stage 5 final inputs: {missing}")
    if not valid_smoke(smoke_path, pilot):
        raise RuntimeError("The approved batch-32 Stage 5 LoRA smoke result is not reusable")
    print("REUSING_VERIFIED_STAGE5_BATCH32_LORA_SMOKE_TEST")

    script = repo / "src" / "train_lora.py"
    common = [
        "--data-dir", str(data),
        "--stage2-validation", str(stage2_validation),
        "--stage3-summary", str(stage3_summary),
        "--stage4-summary", str(stage4_summary),
        "--stage3-result", str(stage3_result),
        "--stage3-checkpoint", str(stage3_checkpoint),
    ]
    for seed in SEEDS:
        config = configs[seed]
        config_path = repo / "configs" / f"stage5_lora_final_seed{seed}.json"
        output = drive / config["artifacts"]["final_output"]
        result_path = output / "results" / "stage5_lora_final_result.json"
        command = [
            sys.executable,
            str(script),
            "--mode", "final",
            "--config", str(config_path),
            *common,
            "--output-dir", str(output),
        ]
        subprocess.run([*command, "--validate-only"], check=True)
        if valid_final(result_path, config):
            print(f"REUSING_VERIFIED_STAGE5_FINAL_SEED_{seed}")
            continue
        print(f"STARTING_STAGE5_FINAL_SEED_{seed}")
        subprocess.run(command, check=True)
        if not valid_final(result_path, config):
            raise RuntimeError(f"Stage 5 seed {seed} finished without verified artifacts")

    manifest = repo / "configs" / "stage5_lora_final_manifest.json"
    output = drive / "stage5_lora" / "final_results"
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "audit_stage5.py"),
            "--manifest", str(manifest),
            "--drive-root", str(drive),
            "--output-dir", str(output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "aggregate_stage5.py"),
            "--manifest", str(manifest),
            "--drive-root", str(drive),
            "--output-dir", str(output),
        ],
        check=True,
    )
    summary = output / "stage5_final_summary.json"
    audit = output / "stage5_final_seed_audit.json"
    if not summary.is_file() or not audit.is_file():
        raise RuntimeError("Stage 5 final summary/audit was not created")
    print(f"STAGE5_FINAL_THREE_SEED_VALIDATION_COMPLETE={summary}")
    print(f"STAGE5_FINAL_DISTINCTNESS_AUDIT={audit}")
    print("TEST_SPLIT_LOADED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
