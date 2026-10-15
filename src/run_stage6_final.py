#!/usr/bin/env python3
"""Run or resume approved Stage 6 final validation runs, then audit them."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


FRACTIONS = (10, 25, 50)
SEEDS = (42, 43, 44)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    data = drive / "stage1_scifact/processed"
    stage6 = drive / "stage6_data_efficiency"
    subset_root = stage6 / "subsets"
    output = stage6 / "final_results"
    manifest = load_json(repo / "configs/stage6_data_efficiency_final_manifest.json")
    if (
        manifest["phase"] != "final_three_seed_validation_only"
        or manifest["random_seeds"] != list(SEEDS)
        or manifest["trained_fractions_percent"] != list(FRACTIONS)
        or manifest["reused_fraction_percent"] != 100
        or manifest["test_evaluation"] is not False
    ):
        raise RuntimeError("Unexpected Stage 6 final manifest")

    pilot_summary = load_json(stage6 / "pilot_results/stage6_pilot_summary.json")
    if (
        pilot_summary["phase"] != "single_seed_pilots"
        or pilot_summary["test_split_loaded"] is not False
        or pilot_summary["final_three_seed_sweep_run"] is not False
        or sorted(row["fraction_percent"] for row in pilot_summary["per_fraction"])
        != [10, 25, 50, 100]
    ):
        raise RuntimeError("The approved Stage 6 pilot summary is missing or invalid")
    smoke = load_json(stage6 / "smoke_fraction10_seed42/results/stage6_smoke_result.json")
    if (
        smoke.get("passed") is not True
        or smoke.get("test_split_loaded") is not False
        or smoke.get("global_steps_completed") != 3
        or smoke.get("config", {}).get("training", {}).get("batch_size") != 32
    ):
        raise RuntimeError("The Stage 6 batch-32 smoke test failed verification")
    print("REUSING_VERIFIED_STAGE6_10_PERCENT_BATCH32_SMOKE")

    # Regenerate the same nested subsets from saved Stage 1 training artifacts.
    subprocess.run(
        [
            sys.executable, str(repo / "src/prepare_data_efficiency_subsets.py"),
            "--data-dir", str(data), "--output-dir", str(subset_root),
            "--subset-seed", "42", "--batch-size", "32",
        ],
        check=True,
    )
    subsets = load_json(subset_root / "stage6_subset_manifest.json")
    if subsets["test_split_loaded"] is not False:
        raise RuntimeError("Subset preparation loaded the test split")
    pilot_equivalence = load_json(
        stage6 / "pilot_results/stage6_100_percent_equivalence.json"
    )
    if pilot_equivalence.get("passed") is not True:
        raise RuntimeError("Pilot 100% equivalence proof is missing")
    for field, filename in (
        ("queries_train_sha256", "queries_train.jsonl"),
        ("qrels_train_sha256", "qrels_train.tsv"),
    ):
        actual = sha256_file(data / filename)
        if actual != subsets["fractions"]["100"][field] or actual != pilot_equivalence[field]:
            raise RuntimeError(f"100% Stage 1 training source changed: {filename}")

    stage3_summary = load_json(repo / "results/stage3_final_summary.json")
    if stage3_summary.get("test_split_loaded") is not False:
        raise RuntimeError("Stage 3 reference loaded test data")
    reference_by_seed = {int(row["seed"]): row for row in stage3_summary["per_seed"]}
    if set(reference_by_seed) != set(SEEDS):
        raise RuntimeError("Stage 3 summary is missing one of the three seeds")
    stage3_config42 = load_json(repo / "configs/stage3_inbatch_final_seed42.json")
    stage6_config100 = load_json(repo / "configs/stage6_data_efficiency_pilot_100.json")
    if (
        stage6_config100["training"] != stage3_config42["training"]
        or stage6_config100["model"] != stage3_config42["model"]
        or stage6_config100["subset"]["queries_train_sha256"]
        != subsets["fractions"]["100"]["queries_train_sha256"]
        or stage6_config100["subset"]["qrels_train_sha256"]
        != subsets["fractions"]["100"]["qrels_train_sha256"]
    ):
        raise RuntimeError("100% configuration or data differs from frozen Stage 3")

    equivalence_rows = []
    for seed in SEEDS:
        config = load_json(repo / f"configs/stage3_inbatch_final_seed{seed}.json")
        expected_training = copy.deepcopy(stage6_config100["training"])
        expected_training["random_seed"] = seed
        result_path = (
            drive / f"stage3_inbatch/final_seed{seed}/results/stage3_final_result.json"
        )
        result = load_json(result_path)
        checkpoint = drive / f"stage3_inbatch/final_seed{seed}/best_checkpoint"
        embeddings = drive / f"stage3_inbatch/final_seed{seed}/embeddings"
        if (
            config["training"] != expected_training
            or config["model"] != stage6_config100["model"]
            or result.get("stage") != 3
            or result.get("mode") != "final"
            or result.get("test_split_loaded") is not False
            or result.get("config", {}).get("training") != config["training"]
            or result.get("config", {}).get("model") != config["model"]
            or result.get("training_queries") != subsets["fractions"]["100"]["realized_query_count"]
            or result.get("training_pairs") != subsets["fractions"]["100"]["training_pair_count"]
            or not checkpoint.is_dir()
            or not (embeddings / "corpus_embeddings.npy").is_file()
            or not (embeddings / "corpus_ids.json").is_file()
        ):
            raise RuntimeError(f"Stage 3 seed {seed} failed 100% reuse equivalence")
        for metric in (
            "recall_at_5", "recall_at_10", "precision_at_5", "precision_at_10",
            "mrr_at_10", "ndcg_at_10",
        ):
            if abs(float(result["validation_metrics"][metric]) - float(reference_by_seed[seed][metric])) > 1e-12:
                raise RuntimeError(f"Stage 3 seed {seed} summary/result mismatch: {metric}")
        equivalence_rows.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "best_checkpoint": str(checkpoint),
                "corpus_cache": str(embeddings),
                "training_config_identical": True,
                "model_config_identical": True,
                "original_stage1_train_hashes_identical": True,
                "summary_metrics_identical": True,
                "checkpoint_and_embeddings_usable": True,
            }
        )
    atomic_write_json(
        output / "stage6_100_percent_three_seed_equivalence.json",
        {
            "stage": 6,
            "fraction_percent": 100,
            "reused_stage": 3,
            "random_seeds": list(SEEDS),
            "per_seed": equivalence_rows,
            "queries_train_sha256": subsets["fractions"]["100"]["queries_train_sha256"],
            "qrels_train_sha256": subsets["fractions"]["100"]["qrels_train_sha256"],
            "training_queries": subsets["fractions"]["100"]["realized_query_count"],
            "training_pairs": subsets["fractions"]["100"]["training_pair_count"],
            "test_split_loaded": False,
            "passed": True,
        },
    )
    print("STAGE6_100_PERCENT_ALL_THREE_STAGE3_SEEDS_EQUIVALENT")

    for fraction in FRACTIONS:
        pilot_config = load_json(repo / f"configs/stage6_data_efficiency_pilot_{fraction}.json")
        subset = subsets["fractions"][str(fraction)]
        for field in (
            "realized_query_count", "training_pair_count", "evaluation_steps",
            "save_steps", "queries_train_sha256", "qrels_train_sha256",
        ):
            if pilot_config["subset"][field] != subset[field]:
                raise RuntimeError(f"Fraction {fraction} subset changed: {field}")
        for seed in SEEDS:
            config_path = repo / manifest["config_files"][str(fraction)][str(seed)]
            config = load_json(config_path)
            expected_training = copy.deepcopy(pilot_config["training"])
            expected_training["random_seed"] = seed
            if (
                config["run_type"] != "final_three_seed_fraction"
                or config["training"] != expected_training
                or config["model"] != pilot_config["model"]
                or config["dataset"] != pilot_config["dataset"]
                or config["subset"] != pilot_config["subset"]
                or config["evaluation"] != pilot_config["evaluation"]
                or config["smoke_test"] != pilot_config["smoke_test"]
            ):
                raise RuntimeError(f"Final fraction {fraction} seed {seed} differs from pilot")
            run_dir = stage6 / f"final_fraction{fraction}_seed{seed}"
            result_path = run_dir / "results" / f"stage6_fraction{fraction}_final_result.json"
            if result_path.is_file():
                result = load_json(result_path)
                if (
                    result.get("mode") != "final"
                    or result.get("test_split_loaded") is not False
                    or result.get("config") != config
                    or not (run_dir / "best_checkpoint").is_dir()
                    or not (run_dir / "embeddings/corpus_embeddings.npy").is_file()
                ):
                    raise RuntimeError(f"Existing fraction {fraction} seed {seed} result is invalid")
                prune_completed_checkpoints(stage6, run_dir, result_path, config, fraction)
                print(f"FRACTION_{fraction}_SEED_{seed}_ALREADY_COMPLETE")
                continue
            print(f"STARTING_STAGE6_FINAL_FRACTION_{fraction}_SEED_{seed}")
            subprocess.run(
                [
                    sys.executable, str(repo / "src/train_data_efficiency.py"),
                    "--mode", "final", "--config", str(config_path),
                    "--data-dir", str(data),
                    "--subset-dir", str(subset_root / f"fraction_{fraction}"),
                    "--baseline-results", str(repo / "results/stage2_validation_baselines.csv"),
                    "--references-dir", str(repo / "results"),
                    "--output-dir", str(run_dir),
                ],
                check=True,
            )
            if not result_path.is_file():
                raise RuntimeError(f"Fraction {fraction} seed {seed} completed without results")
            prune_completed_checkpoints(stage6, run_dir, result_path, config, fraction)

    subprocess.run(
        [
            sys.executable, str(repo / "src/audit_stage6_final.py"),
            "--repo-dir", str(repo), "--drive-root", str(drive),
            "--output-dir", str(output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(repo / "src/aggregate_stage6_final.py"),
            "--repo-dir", str(repo), "--drive-root", str(drive),
            "--output-dir", str(output),
        ],
        check=True,
    )
    print(f"STAGE6_FINAL_VALIDATION_COMPLETE={output / 'stage6_final_summary.json'}")
    return 0


def prune_completed_checkpoints(stage6, run_dir, result_path, config, fraction):
    """Remove only resumable Trainer saves after final artifacts are verified."""
    import shutil

    stage6 = stage6.resolve()
    run_dir = run_dir.resolve()
    if run_dir.parent != stage6 or not run_dir.name.startswith(f"final_fraction{fraction}_seed"):
        raise RuntimeError(f"Refusing checkpoint cleanup outside a Stage 6 final run: {run_dir}")

    result = load_json(result_path)
    if (result.get("mode") != "final"
            or result.get("test_split_loaded") is not False
            or result.get("config") != config):
        raise RuntimeError(f"Refusing checkpoint cleanup: invalid final result in {run_dir}")

    best = run_dir / "best_checkpoint"
    embeddings = run_dir / "embeddings"
    loss_curve = run_dir / "results" / f"stage6_fraction{fraction}_final_loss_curve.csv"
    required = [
        result_path,
        loss_curve,
        embeddings / "corpus_embeddings.npy",
        embeddings / "corpus_ids.json",
    ]
    if (any(not path.is_file() or path.stat().st_size == 0 for path in required)
            or not best.is_dir()
            or not any(path.stat().st_size > 0 for path in best.rglob("*.safetensors"))):
        raise RuntimeError(f"Refusing checkpoint cleanup: final artifacts incomplete in {run_dir}")

    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        print(f"COMPLETED_RUN_CHECKPOINTS_ALREADY_ABSENT={run_dir}", flush=True)
        return
    if checkpoints.is_symlink() or not checkpoints.is_dir() or checkpoints.resolve().parent != run_dir:
        raise RuntimeError(f"Refusing unsafe checkpoint cleanup target: {checkpoints}")
    shutil.rmtree(checkpoints)
    print(f"PRUNED_COMPLETED_RUN_CHECKPOINTS={checkpoints}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
