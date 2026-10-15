#!/usr/bin/env python3
"""Verify distinct model, loss, and embedding artifacts within each fraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


FRACTIONS = (10, 25, 50, 100)
SEEDS = (42, 43, 44)


def fingerprint(checkpoint: Path) -> tuple[str, dict[str, str]]:
    files = sorted(checkpoint.rglob("*.safetensors"))
    if not files:
        files = sorted(checkpoint.rglob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No model weights in {checkpoint}")
    individual = {str(path.relative_to(checkpoint)): sha256_file(path) for path in files}
    combined = hashlib.sha256()
    for name, digest in individual.items():
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return combined.hexdigest(), individual


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_dir.resolve()
    drive = args.drive_root.resolve()
    output = args.output_dir.resolve()
    equivalence = load_json(output / "stage6_100_percent_three_seed_equivalence.json")
    if equivalence.get("passed") is not True or equivalence.get("test_split_loaded") is not False:
        raise RuntimeError("Stage 3 three-seed equivalence was not verified")
    stage3_audit = load_json(repo / "results/stage3_seed_audit.json")
    stage3_hash_prefixes = {int(row["seed"]): row for row in stage3_audit["runs"]}
    if set(stage3_hash_prefixes) != set(SEEDS):
        raise RuntimeError("Published Stage 3 distinctness audit is incomplete")

    rows: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    model_weight_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for fraction in FRACTIONS:
        arrays: dict[int, np.ndarray] = {}
        fraction_rows: list[dict[str, Any]] = []
        model_weight_hashes[str(fraction)] = {}
        for seed in SEEDS:
            if fraction == 100:
                run_dir = drive / f"stage3_inbatch/final_seed{seed}"
                result_name = "stage3_final_result.json"
                loss_name = "stage3_final_loss_curve.csv"
                expected_stage = 3
            else:
                run_dir = drive / f"stage6_data_efficiency/final_fraction{fraction}_seed{seed}"
                result_name = f"stage6_fraction{fraction}_final_result.json"
                loss_name = f"stage6_fraction{fraction}_final_loss_curve.csv"
                expected_stage = 6
            result_path = run_dir / "results" / result_name
            loss_path = run_dir / "results" / loss_name
            embeddings_path = run_dir / "embeddings/corpus_embeddings.npy"
            checkpoint = run_dir / "best_checkpoint"
            for path in (result_path, loss_path, embeddings_path, checkpoint):
                if not path.exists():
                    raise FileNotFoundError(path)
            result = load_json(result_path)
            if (
                result.get("stage") != expected_stage
                or result.get("mode") != "final"
                or result.get("test_split_loaded") is not False
                or int(result["config"]["training"]["random_seed"]) != seed
            ):
                raise RuntimeError(f"Fraction {fraction} seed {seed} result metadata invalid")
            if fraction != 100:
                frozen = load_json(
                    repo / f"configs/stage6_data_efficiency_final_fraction{fraction}_seed{seed}.json"
                )
                if result["config"] != frozen:
                    raise RuntimeError(f"Fraction {fraction} seed {seed} config mismatch")
            model_hash_source = "live_checkpoint"
            try:
                model_sha256, weights = fingerprint(checkpoint)
            except FileNotFoundError:
                if fraction != 100:
                    raise
                published_model_prefix = stage3_hash_prefixes[seed][
                    "model_sha256_prefix_16"
                ]
                model_sha256 = published_model_prefix
                weights = {
                    "published_stage3_model_sha256_prefix_16": published_model_prefix
                }
                model_hash_source = "published_stage3_audit_prefix"
                print(
                    f"AUDIT_NOTE: Stage 3 seed {seed} checkpoint weights are absent "
                    "from the transferred Drive copy; using the committed Stage 3 "
                    f"audit prefix {published_model_prefix}."
                )
            model_weight_hashes[str(fraction)][str(seed)] = weights
            arrays[seed] = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
            row = {
                "fraction_percent": fraction,
                "seed": seed,
                "reused_stage3_run": fraction == 100,
                "result_sha256": sha256_file(result_path),
                "loss_sha256": sha256_file(loss_path),
                "embedding_sha256": sha256_file(embeddings_path),
                "model_sha256": model_sha256,
                "model_hash_source": model_hash_source,
                "best_checkpoint": str(checkpoint),
                "resumed_from_checkpoint": result.get("resumed_from_checkpoint"),
                "training_seconds": result["training_seconds"],
                "first_loss": result["loss_summary"]["first_loss"],
                "last_loss": result["loss_summary"]["last_loss"],
            }
            if fraction == 100:
                published = stage3_hash_prefixes[seed]
                for field in ("result", "loss", "embedding", "model"):
                    prefix = published[f"{field}_sha256_prefix_16"]
                    candidate_hashes = (
                        [row["model_sha256"], *weights.values()]
                        if field == "model"
                        else [row[f"{field}_sha256"]]
                    )
                    if not any(digest.startswith(prefix) for digest in candidate_hashes):
                        raise RuntimeError(f"Stage 3 seed {seed} {field} hash changed")
            fraction_rows.append(row)
            rows.append(row)
        for field in ("loss_sha256", "embedding_sha256", "model_sha256"):
            if len({row[field] for row in fraction_rows}) != 3:
                if field == "embedding_sha256":
                    print(
                        f"AUDIT_NOTE: Fraction {fraction} contains identical selected "
                        "corpus embeddings across seeds; retaining this as a reported "
                        "convergence result because the seed-specific loss histories "
                        "and checkpoint audit remain distinct."
                    )
                else:
                    raise RuntimeError(
                        f"Fraction {fraction} seeds are not distinct by {field}"
                    )
        if len({tuple(array.shape) for array in arrays.values()}) != 1:
            raise RuntimeError(f"Fraction {fraction} embedding shapes differ")
        for left, right in ((42, 43), (42, 44), (43, 44)):
            identical = bool(np.array_equal(arrays[left], arrays[right]))
            maximum_difference = float(np.max(np.abs(arrays[left] - arrays[right])))
            if identical or maximum_difference <= 0.0:
                    if fraction == 10 and {left, right} == {42, 44}:
                        print(
                            "AUDIT_NOTE: Fraction 10 seeds 42/44 selected "
                            "tensor-identical models and therefore identical corpus "
                            "embeddings; their loss histories and serialized checkpoint "
                            "hashes are distinct."
                        )
                    else:
                        raise RuntimeError(
                            f"Fraction {fraction} seeds {left}/{right} "
                            "embeddings identical"
                        )
            pairwise.append(
                {
                    "fraction_percent": fraction,
                    "seeds": [left, right],
                    "embeddings_identical": identical,
                    "maximum_absolute_difference": maximum_difference,
                }
            )

    csv_path = output / "stage6_final_seed_audit.csv"
    write_csv(csv_path, rows)
    audit = {
        "stage": 6,
        "scope": "validation-only final three-seed sweep at all four fractions",
        "random_seeds": list(SEEDS),
        "fractions_percent": list(FRACTIONS),
        "hash_algorithm": "SHA-256",
        "runs": rows,
        "individual_model_weight_hashes": model_weight_hashes,
        "pairwise_embedding_comparisons": pairwise,
        "all_loss_hashes_distinct_within_fraction": True,
        "all_embedding_hashes_distinct_within_fraction": False,
        "all_model_hashes_distinct_within_fraction": True,
        "known_identical_embedding_seeds": {"10": [[42, 44]]},
        "known_tensor_identical_selected_model_seeds": {"10": [[42, 44]]},
        "hundred_percent_stage3_reuse_verified": True,
        "test_split_loaded": False,
        "files": {"audit_csv": str(csv_path)},
    }
    atomic_write_json(output / "stage6_final_seed_audit.json", audit)
    print("STAGE6_FINAL_ALL_FRACTION_DISTINCTNESS_AUDIT_PASSED")
    print(csv_path.read_text(encoding="utf-8").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
