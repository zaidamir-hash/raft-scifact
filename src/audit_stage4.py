#!/usr/bin/env python3
"""Audit distinctness of the frozen Stage 4 three-seed artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_retrieval import atomic_write_json, load_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def model_fingerprint(checkpoint: Path) -> tuple[str, dict[str, str]]:
    files = sorted(checkpoint.rglob("*.safetensors"))
    if not files:
        files = sorted(checkpoint.rglob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No model weights found under {checkpoint}")
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
    args = parse_args()
    manifest = load_json(args.manifest.resolve())
    seeds = [int(seed) for seed in manifest["random_seeds"]]
    if seeds != [42, 43, 44]:
        raise ValueError("Stage 4 audit requires seeds 42, 43, and 44")
    rows: list[dict[str, Any]] = []
    arrays: dict[int, np.ndarray] = {}
    individual_model_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        run_dir = args.drive_root.resolve() / manifest["output_directories"][str(seed)]
        result_path = run_dir / "results" / "stage4_final_result.json"
        loss_path = run_dir / "results" / "stage4_final_loss_curve.csv"
        embeddings_path = run_dir / "embeddings" / "corpus_embeddings.npy"
        checkpoint = run_dir / "best_checkpoint"
        required = (result_path, loss_path, embeddings_path, checkpoint)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} audit artifacts are missing: {missing}")
        result = load_json(result_path)
        if (
            result.get("stage") != 4
            or result.get("mode") != "final"
            or result.get("test_split_loaded") is not False
            or result.get("config", {}).get("training", {}).get("random_seed") != seed
            or result.get("config", {}).get("training", {}).get(
                "early_stopping_rule", {}
            ).get("patience_evaluations") != 2
        ):
            raise ValueError(f"Seed {seed} result metadata failed the audit")
        model_sha256, weight_hashes = model_fingerprint(checkpoint)
        individual_model_hashes[str(seed)] = weight_hashes
        arrays[seed] = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        rows.append(
            {
                "seed": seed,
                "result_sha256": sha256_file(result_path),
                "loss_sha256": sha256_file(loss_path),
                "embedding_sha256": sha256_file(embeddings_path),
                "model_sha256": model_sha256,
                "best_checkpoint": str(checkpoint),
                "resumed_from_checkpoint": result.get("resumed_from_checkpoint"),
                "training_seconds": result["training_seconds"],
                "first_loss": result["loss_summary"]["first_loss"],
                "last_loss": result["loss_summary"]["last_loss"],
            }
        )

    for field in ("loss_sha256", "embedding_sha256", "model_sha256"):
        values = [row[field] for row in rows]
        if len(set(values)) != len(values):
            raise RuntimeError(f"Stage 4 seed runs are not distinct by {field}")
    shapes = {seed: tuple(array.shape) for seed, array in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Seed embedding shapes differ: {shapes}")
    pairwise: list[dict[str, Any]] = []
    for left, right in ((42, 43), (42, 44), (43, 44)):
        identical = bool(np.array_equal(arrays[left], arrays[right]))
        maximum_difference = float(np.max(np.abs(arrays[left] - arrays[right])))
        if identical or maximum_difference <= 0.0:
            raise RuntimeError(f"Seed {left} and {right} embeddings are identical")
        pairwise.append(
            {
                "seeds": [left, right],
                "embeddings_identical": identical,
                "maximum_absolute_difference": maximum_difference,
            }
        )

    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "stage4_final_seed_audit.csv"
    write_csv(csv_path, rows)
    audit = {
        "stage": 4,
        "scope": "final three-seed validation runs",
        "random_seeds": seeds,
        "hash_algorithm": "SHA-256",
        "runs": rows,
        "individual_model_weight_hashes": individual_model_hashes,
        "pairwise_embedding_comparisons": pairwise,
        "all_loss_hashes_distinct": True,
        "all_embedding_hashes_distinct": True,
        "all_model_hashes_distinct": True,
        "test_split_loaded": False,
        "conclusion": "All three Stage 4 runs have distinct model, loss, and embedding artifacts.",
        "files": {"audit_csv": str(csv_path)},
    }
    json_path = output_dir / "stage4_final_seed_audit.json"
    atomic_write_json(json_path, audit)
    print("STAGE4_FINAL_THREE_SEED_DISTINCTNESS_AUDIT_PASSED")
    print(csv_path.read_text(encoding="utf-8").rstrip())
    print(json.dumps(pairwise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
