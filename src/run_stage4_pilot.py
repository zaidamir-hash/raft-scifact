#!/usr/bin/env python3
"""Mine negatives, run the Stage 4 smoke test, then run the seed-42 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import load_json, sha256_file
from mine_negatives import config_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def verified_mining(config: dict, negatives: Path, summary_path: Path) -> bool:
    if not negatives.is_file() or not summary_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
        return (
            summary.get("stage") == 4
            and summary.get("config_sha256") == config_digest(config)
            and summary.get("data", {}).get("validation_loaded") is False
            and summary.get("data", {}).get("test_split_loaded") is False
            and summary.get("selection", {}).get("hard_negatives_per_positive") == 2
            and summary.get("selection", {}).get("bm25_negatives_per_positive") == 1
            and summary.get("selection", {}).get("dense_negatives_per_positive") == 1
            and summary.get("selection", {}).get("candidate_pool_size_per_method") == 5183
            and summary.get("selection", {}).get("candidate_search_scope") == "full_corpus"
            and summary.get("selection", {}).get("filter_relaxed") is False
            and summary.get("outputs", {}).get("hard_negatives_sha256")
            == sha256_file(negatives)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def verified_smoke(config: dict, result_path: Path, negatives: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
        return (
            result.get("stage") == 4
            and result.get("mode") == "smoke"
            and result.get("passed") is True
            and result.get("global_steps_completed") == 3
            and result.get("config_sha256") == config_digest(config)
            and result.get("hard_negatives_sha256") == sha256_file(negatives)
            and result.get("test_split_loaded") is False
            and result.get("config", {}).get("training", {}).get("batch_size") == 32
            and result.get("config", {}).get("training", {}).get("effective_batch_size") == 32
            and result.get("config", {}).get("training", {}).get("gradient_accumulation_steps") == 1
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def verified_pilot(config: dict, result_path: Path, negatives: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
        return (
            result.get("stage") == 4
            and result.get("mode") == "pilot"
            and result.get("config_sha256") == config_digest(config)
            and result.get("hard_negatives_sha256") == sha256_file(negatives)
            and result.get("test_split_loaded") is False
            and result.get("fresh_pretrained_initialization") is True
            and Path(result["saved_best_checkpoint"]).is_dir()
            and Path(result["corpus_cache"]["saved_best_checkpoint"]).is_dir()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    drive_root = args.drive_root.resolve()
    config_path = repo_dir / "configs" / "stage4_hardneg_pilot.json"
    config = load_json(config_path)
    data_dir = drive_root / config["artifacts"]["stage1_data"]
    stage3_run_dir = drive_root / "stage3_inbatch" / "final_seed42"
    stage2_results = drive_root / config["artifacts"]["stage2_results"]
    stage3_summary = drive_root / config["artifacts"]["stage3_summary"]
    mining_dir = drive_root / "stage4_hardneg" / "mined"
    negatives = drive_root / config["artifacts"]["mined_negatives"]
    mining_summary = drive_root / config["artifacts"]["mining_summary"]
    smoke_dir = drive_root / config["artifacts"]["smoke_output"]
    pilot_dir = drive_root / config["artifacts"]["pilot_output"]
    smoke_result = smoke_dir / "results" / "stage4_smoke_result.json"
    pilot_result = pilot_dir / "results" / "stage4_pilot_result.json"
    train_script = repo_dir / "src" / "train_hard_negatives.py"

    if verified_mining(config, negatives, mining_summary):
        print("REUSING_VERIFIED_STAGE4_MINED_NEGATIVES")
    else:
        print("STARTING_STAGE4_HARD_NEGATIVE_MINING")
        subprocess.run(
            [
                sys.executable,
                str(repo_dir / "src" / "mine_negatives.py"),
                "--config",
                str(config_path),
                "--data-dir",
                str(data_dir),
                "--stage3-run-dir",
                str(stage3_run_dir),
                "--output-dir",
                str(mining_dir),
            ],
            check=True,
        )
        if not verified_mining(config, negatives, mining_summary):
            raise RuntimeError("Mining completed without a verified Stage 4 artifact")

    shared_train_args = [
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
    ]

    if verified_smoke(config, smoke_result, negatives):
        print("REUSING_VERIFIED_STAGE4_BATCH32_SMOKE_TEST")
    else:
        print("STARTING_STAGE4_BATCH32_SMOKE_TEST")
        try:
            subprocess.run(
                [
                    *shared_train_args,
                    "--mode",
                    "smoke",
                    "--output-dir",
                    str(smoke_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                "STAGE4_STOPPED_AFTER_SMOKE_FAILURE. "
                "The pilot was not started and no batch setting was changed.",
                file=sys.stderr,
            )
            raise
        if not verified_smoke(config, smoke_result, negatives):
            raise RuntimeError("Smoke test completed without a verified pass result")
    print("STAGE4_BATCH32_SMOKE_TEST_VERIFIED_PASSED")

    if verified_pilot(config, pilot_result, negatives):
        print("REUSING_VERIFIED_STAGE4_SEED42_PILOT")
    else:
        print("STARTING_STAGE4_SEED42_PILOT")
        subprocess.run(
            [
                *shared_train_args,
                "--mode",
                "pilot",
                "--output-dir",
                str(pilot_dir),
            ],
            check=True,
        )
        if not verified_pilot(config, pilot_result, negatives):
            raise RuntimeError("Pilot completed without a verified result/checkpoint")

    result = load_json(pilot_result)
    if result.get("test_split_loaded") is not False:
        raise RuntimeError("Stage 4 pilot result does not prove test isolation")
    print("STAGE4_SEED42_PILOT_VALIDATION_ONLY_COMPLETE")
    print(pilot_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
