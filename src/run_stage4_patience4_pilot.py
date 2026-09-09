#!/usr/bin/env python3
"""Run the approved Stage 4 seed-42 pilot with early-stopping patience four."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import load_json, sha256_file
from mine_negatives import config_digest
from train_hard_negatives import mining_signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser.parse_args()


def verify_exact_mining_reuse(config: dict, negatives: Path, summary_path: Path) -> None:
    if not negatives.is_file() or not summary_path.is_file():
        raise FileNotFoundError("The completed Stage 4 mined-negative artifacts are missing")
    summary = load_json(summary_path)
    checks = (
        summary.get("stage") == 4,
        summary.get("data", {}).get("test_split_loaded") is False,
        summary.get("selection", {}).get("candidate_pool_size_per_method") == 5183,
        summary.get("selection", {}).get("candidate_search_scope") == "full_corpus",
        summary.get("selection", {}).get("cosine_margin") == 0.05,
        summary.get("selection", {}).get("hard_negatives_per_positive") == 2,
        summary.get("selection", {}).get("bm25_negatives_per_positive") == 1,
        summary.get("selection", {}).get("dense_negatives_per_positive") == 1,
        summary.get("selection", {}).get("filter_relaxed") is False,
        mining_signature(summary.get("config", {})) == mining_signature(config),
        summary.get("outputs", {}).get("hard_negatives_sha256") == sha256_file(negatives),
    )
    if not all(checks):
        raise ValueError("Existing mined negatives do not match the patience-4 mining invariants")


def verify_smoke_reuse(config: dict, smoke_path: Path, negatives: Path) -> None:
    if not smoke_path.is_file():
        raise FileNotFoundError("The completed Stage 4 batch-32 smoke result is missing")
    smoke = load_json(smoke_path)
    old_config = smoke.get("config", {})
    invariant_training_keys = (
        "random_seed",
        "batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "epochs",
        "warmup_ratio",
        "optimizer",
        "weight_decay",
        "lr_scheduler",
        "max_grad_norm",
        "mixed_precision",
        "loss",
        "batch_sampler",
        "evaluation_frequency",
        "evaluation_steps",
        "save_frequency",
        "save_steps",
        "best_checkpoint_metric",
    )
    checks = (
        smoke.get("stage") == 4,
        smoke.get("mode") == "smoke",
        smoke.get("passed") is True,
        smoke.get("global_steps_completed") == 3,
        smoke.get("test_split_loaded") is False,
        smoke.get("hard_negatives_sha256") == sha256_file(negatives),
        old_config.get("model") == config.get("model"),
        old_config.get("mining") == config.get("mining"),
        all(
            old_config.get("training", {}).get(key) == config["training"].get(key)
            for key in invariant_training_keys
        ),
    )
    if not all(checks):
        raise ValueError("Existing smoke test is not reusable for the patience-only adjustment")


def verified_pilot(config: dict, result_path: Path, negatives: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
        return (
            result.get("stage") == 4
            and result.get("mode") == "pilot"
            and result.get("config_sha256") == config_digest(config)
            and result.get("mining_config_signature") == mining_signature(config)
            and result.get("hard_negatives_sha256") == sha256_file(negatives)
            and result.get("test_split_loaded") is False
            and result.get("fresh_pretrained_initialization") is True
            and result.get("config", {}).get("training", {}).get(
                "early_stopping_rule", {}
            ).get("patience_evaluations") == 4
            and Path(result["saved_best_checkpoint"]).is_dir()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    drive_root = args.drive_root.resolve()
    config_path = repo_dir / "configs" / "stage4_hardneg_pilot_patience4.json"
    config = load_json(config_path)
    data_dir = drive_root / config["artifacts"]["stage1_data"]
    negatives = drive_root / config["artifacts"]["mined_negatives"]
    mining_summary = drive_root / config["artifacts"]["mining_summary"]
    smoke_path = (
        drive_root
        / config["artifacts"]["smoke_output"]
        / "results"
        / "stage4_smoke_result.json"
    )
    pilot_dir = drive_root / config["artifacts"]["pilot_output"]
    pilot_result = pilot_dir / "results" / "stage4_pilot_result.json"
    stage2_results = drive_root / config["artifacts"]["stage2_results"]
    stage3_summary = drive_root / config["artifacts"]["stage3_summary"]
    train_script = repo_dir / "src" / "train_hard_negatives.py"

    verify_exact_mining_reuse(config, negatives, mining_summary)
    print(f"REUSING_EXACT_MINED_NEGATIVES_SHA256={sha256_file(negatives)}")
    verify_smoke_reuse(config, smoke_path, negatives)
    print("REUSING_VERIFIED_STAGE4_BATCH32_SMOKE_TEST")

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
        "pilot",
        "--output-dir",
        str(pilot_dir),
    ]
    subprocess.run([*command, "--validate-only"], check=True)
    if verified_pilot(config, pilot_result, negatives):
        print("REUSING_VERIFIED_STAGE4_PATIENCE4_PILOT")
    else:
        print("STARTING_STAGE4_SEED42_PATIENCE4_PILOT")
        subprocess.run(command, check=True)
        if not verified_pilot(config, pilot_result, negatives):
            raise RuntimeError("Patience-4 pilot completed without a verified result")
    print(f"STAGE4_PATIENCE4_PILOT_COMPLETE={pilot_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
