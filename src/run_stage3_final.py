#!/usr/bin/env python3
"""Run or resume the frozen three-seed Stage 3 validation experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def verified_completed_result(path: Path, seed: int) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return (
        result.get("mode") == "final"
        and result.get("test_split_loaded") is False
        and int(result["config"]["training"]["random_seed"]) == seed
        and Path(result["saved_best_checkpoint"]).is_dir()
    )


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    drive_root = args.drive_root.resolve()
    data_dir = drive_root / "stage1_scifact" / "processed"
    baseline_results = drive_root / "stage2_baselines" / "results" / "stage2_baselines.csv"
    smoke_result_path = (
        drive_root / "stage3_inbatch" / "smoke_seed42" / "results" / "stage3_smoke_result.json"
    )
    smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    if not (
        smoke.get("passed") is True
        and int(smoke.get("global_steps_completed", -1)) == 3
        and smoke.get("test_split_loaded") is False
    ):
        raise RuntimeError("The persisted Stage 3 smoke-test result is not valid")
    print("REUSING_VERIFIED_STAGE3_SMOKE_TEST")

    train_script = repo_dir / "src" / "train.py"
    for seed in SEEDS:
        config = repo_dir / "configs" / f"stage3_inbatch_final_seed{seed}.json"
        output_dir = drive_root / "stage3_inbatch" / f"final_seed{seed}"
        result_path = output_dir / "results" / "stage3_final_result.json"
        if verified_completed_result(result_path, seed):
            print(f"SEED_{seed}_ALREADY_COMPLETE")
            continue
        print(f"STARTING_STAGE3_FINAL_SEED_{seed}")
        subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--mode",
                "final",
                "--config",
                str(config),
                "--data-dir",
                str(data_dir),
                "--baseline-results",
                str(baseline_results),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )
        if not verified_completed_result(result_path, seed):
            raise RuntimeError(f"Seed {seed} finished without a verified result")

    final_results = drive_root / "stage3_inbatch" / "final_results"
    subprocess.run(
        [
            sys.executable,
            str(repo_dir / "src" / "aggregate_stage3.py"),
            "--manifest",
            str(repo_dir / "configs" / "stage3_inbatch_final_manifest.json"),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(final_results),
        ],
        check=True,
    )
    summary = final_results / "stage3_final_summary.json"
    if not summary.is_file():
        raise RuntimeError("Final Stage 3 summary was not created")
    print(f"STAGE3_FINAL_THREE_SEED_COMPLETE={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
