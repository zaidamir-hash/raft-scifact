#!/usr/bin/env python3
"""Run the approved Stage 5 profiling smoke, LoRA smoke, and seed-42 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def valid_result(path: Path, mode: str, expected_config: dict) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return (
        result.get("stage") == 5
        and result.get("mode") == mode
        and result.get("config") == expected_config
        and result.get("test_split_loaded") is False
        and result.get("global_steps_completed", 0) > 0
        and (mode == "pilot" or result.get("passed") is True)
    )


def main() -> int:
    args = parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    config = repo / "configs" / "stage5_lora_pilot.json"
    expected_config = json.loads(config.read_text(encoding="utf-8"))
    script = repo / "src" / "train_lora.py"
    data = drive / "stage1_scifact" / "processed"
    stage2_validation = repo / "results" / "stage2_validation_baselines.csv"
    stage3_summary = repo / "results" / "stage3_final_summary.json"
    stage4_summary = repo / "results" / "stage4_final_summary.json"
    stage3_result = drive / "stage3_inbatch" / "final_seed42" / "results" / "stage3_final_result.json"
    stage3_checkpoint = drive / "stage3_inbatch" / "final_seed42" / "best_checkpoint"
    required = [
        config,
        script,
        stage2_validation,
        data / "corpus.jsonl",
        data / "queries_train.jsonl",
        data / "qrels_train.tsv",
        data / "queries_validation.jsonl",
        data / "qrels_validation.tsv",
        stage3_summary,
        stage4_summary,
        stage3_result,
        stage3_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing restartable Stage 5 inputs: {missing}")

    common = [
        "--config", str(config),
        "--data-dir", str(data),
        "--stage2-validation", str(stage2_validation),
        "--stage3-summary", str(stage3_summary),
        "--stage4-summary", str(stage4_summary),
        "--stage3-result", str(stage3_result),
        "--stage3-checkpoint", str(stage3_checkpoint),
    ]
    runs = (
        (
            "profile-full",
            drive / "stage5_lora" / "full_ft_profile_seed42",
            "stage5_full_ft_profile_result.json",
        ),
        (
            "smoke",
            drive / "stage5_lora" / "smoke_seed42",
            "stage5_lora_smoke_result.json",
        ),
        (
            "pilot",
            drive / "stage5_lora" / "pilot_seed42",
            "stage5_lora_pilot_result.json",
        ),
    )
    for mode, output, filename in runs:
        result_path = output / "results" / filename
        if valid_result(result_path, mode, expected_config):
            print(f"REUSING_VERIFIED_STAGE5_{mode.upper().replace('-', '_')}")
            continue
        print(f"STARTING_STAGE5_{mode.upper().replace('-', '_')}")
        subprocess.run(
            [sys.executable, str(script), "--mode", mode, *common, "--output-dir", str(output)],
            check=True,
        )
        if not valid_result(result_path, mode, expected_config):
            raise RuntimeError(f"Stage 5 {mode} finished without a verified result")
        if mode == "smoke":
            print("STAGE5_SMOKE_TEST_PASSED_AT_BATCH_32")

    pilot_dir = drive / "stage5_lora" / "pilot_seed42"
    print(f"STAGE5_PILOT_RESULT={pilot_dir / 'results/stage5_lora_pilot_result.json'}")
    print(f"STAGE5_COMPARISON={pilot_dir / 'results/stage5_pilot_validation_comparison.csv'}")
    print(f"STAGE5_EFFICIENCY={pilot_dir / 'results/stage5_efficiency_comparison.json'}")
    print(f"STAGE5_ADAPTER={pilot_dir / 'best_adapter'}")
    print(f"STAGE5_EMBEDDINGS={pilot_dir / 'embeddings'}")
    print("TEST_SPLIT_LOADED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
