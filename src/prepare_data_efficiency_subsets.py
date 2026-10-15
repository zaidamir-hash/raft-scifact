#!/usr/bin/env python3
"""Build deterministic, nested Stage 6 training subsets without reading test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_retrieval import atomic_write_json, load_jsonl, load_qrels, sha256_file


FRACTIONS = (10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


class UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_components(
    query_ids: set[str], qrels: dict[str, dict[str, float]]
) -> list[list[str]]:
    union_find = UnionFind(query_ids)
    document_queries: dict[str, list[str]] = defaultdict(list)
    for query_id, relevant in qrels.items():
        for document_id in relevant:
            document_queries[document_id].append(query_id)
    for linked_queries in document_queries.values():
        anchor = linked_queries[0]
        for query_id in linked_queries[1:]:
            union_find.union(anchor, query_id)
    components: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        components[union_find.find(query_id)].append(query_id)
    canonical = [sorted(component) for component in components.values()]
    return sorted(canonical, key=lambda component: tuple(component))


def component_id(component: list[str]) -> str:
    payload = "\n".join(component).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_nested_components(
    components: list[list[str]], total_queries: int, seed: int
) -> dict[int, list[list[str]]]:
    ordered = [list(component) for component in components]
    random.Random(seed).shuffle(ordered)
    cumulative: list[int] = []
    running = 0
    for component in ordered:
        running += len(component)
        cumulative.append(running)

    selected: dict[int, list[list[str]]] = {}
    for percentage in FRACTIONS:
        if percentage == 100:
            prefix_length = len(ordered)
        else:
            target = round(total_queries * percentage / 100.0)
            prefix_length = min(
                range(1, len(ordered) + 1),
                key=lambda index: (abs(cumulative[index - 1] - target), index),
            )
        selected[percentage] = ordered[:prefix_length]
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_qrels(path: Path, qrels: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for query_id in sorted(qrels):
            for document_id in sorted(qrels[query_id]):
                writer.writerow([query_id, document_id, qrels[query_id][document_id]])
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.subset_seed != 42:
        raise ValueError("The approved Stage 6 subset seed is 42")
    if args.batch_size != 32:
        raise ValueError("The frozen Stage 3 batch size is 32")

    required = (
        "queries_train.jsonl",
        "qrels_train.tsv",
        "queries_validation.jsonl",
        "qrels_validation.tsv",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1 train/validation files: {missing}")

    train_queries = load_jsonl(data_dir / "queries_train.jsonl")
    train_qrels = load_qrels(data_dir / "qrels_train.tsv")
    validation_queries = load_jsonl(data_dir / "queries_validation.jsonl")
    validation_qrels = load_qrels(data_dir / "qrels_validation.tsv")
    train_ids = {str(query["id"]) for query in train_queries}
    validation_ids = {str(query["id"]) for query in validation_queries}
    if train_ids != set(train_qrels):
        raise ValueError("Stage 1 train queries and qrels differ")
    if validation_ids != set(validation_qrels):
        raise ValueError("Stage 1 validation queries and qrels differ")
    if train_ids & validation_ids:
        raise ValueError("Stage 1 train and validation query IDs overlap")
    train_documents = {
        document_id for relevant in train_qrels.values() for document_id in relevant
    }
    validation_documents = {
        document_id for relevant in validation_qrels.values() for document_id in relevant
    }
    if train_documents & validation_documents:
        raise ValueError("Stage 1 train/validation relevant-document leakage detected")

    components = build_components(train_ids, train_qrels)
    selected_components = select_nested_components(components, len(train_ids), args.subset_seed)
    query_by_id = {str(query["id"]): query for query in train_queries}
    source_hashes = {
        name: sha256_file(data_dir / name)
        for name in required
    }
    fraction_manifests: dict[str, dict[str, Any]] = {}
    previous_ids: set[str] = set()
    for percentage in FRACTIONS:
        components_for_fraction = selected_components[percentage]
        selected_ids = {
            query_id for component in components_for_fraction for query_id in component
        }
        if not previous_ids <= selected_ids:
            raise RuntimeError("Stage 6 subsets are not nested")
        previous_ids = selected_ids
        selected_queries = [
            query_by_id[str(query["id"])]
            for query in train_queries
            if str(query["id"]) in selected_ids
        ]
        selected_qrels = {
            query_id: train_qrels[query_id] for query_id in sorted(selected_ids)
        }
        selected_documents = {
            document_id
            for relevant in selected_qrels.values()
            for document_id in relevant
        }
        if selected_documents & validation_documents:
            raise RuntimeError(f"Fraction {percentage} leaks validation documents")
        fraction_dir = output_dir / f"fraction_{percentage}"
        if percentage < 100:
            write_jsonl(fraction_dir / "queries_train.jsonl", selected_queries)
            write_qrels(fraction_dir / "qrels_train.tsv", selected_qrels)
            query_path = fraction_dir / "queries_train.jsonl"
            qrels_path = fraction_dir / "qrels_train.tsv"
        else:
            query_path = data_dir / "queries_train.jsonl"
            qrels_path = data_dir / "qrels_train.tsv"

        pair_count = sum(len(relevant) for relevant in selected_qrels.values())
        steps_per_epoch = math.ceil(pair_count / args.batch_size)
        evaluation_steps = max(1, math.ceil(steps_per_epoch / 5.0))
        manifest = {
            "stage": 6,
            "target_fraction_percent": percentage,
            "target_query_count": round(len(train_ids) * percentage / 100.0),
            "realized_query_count": len(selected_ids),
            "realized_fraction": len(selected_ids) / len(train_ids),
            "training_pair_count": pair_count,
            "component_count": len(components_for_fraction),
            "selected_component_sha256": [
                component_id(component) for component in components_for_fraction
            ],
            "selected_query_ids": sorted(selected_ids),
            "queries_train_path": str(query_path),
            "qrels_train_path": str(qrels_path),
            "queries_train_sha256": sha256_file(query_path),
            "qrels_train_sha256": sha256_file(qrels_path),
            "steps_per_epoch": steps_per_epoch,
            "evaluation_steps": evaluation_steps,
            "save_steps": evaluation_steps,
            "train_validation_relevant_document_overlap": 0,
            "test_split_loaded": False,
        }
        atomic_write_json(fraction_dir / "subset_manifest.json", manifest)
        fraction_manifests[str(percentage)] = manifest

    full = fraction_manifests["100"]
    if full["realized_query_count"] != len(train_ids):
        raise RuntimeError("The 100% subset does not contain every Stage 1 training query")
    if full["queries_train_sha256"] != source_hashes["queries_train.jsonl"]:
        raise RuntimeError("The 100% query file is not the original Stage 1 train file")
    if full["qrels_train_sha256"] != source_hashes["qrels_train.tsv"]:
        raise RuntimeError("The 100% qrels file is not the original Stage 1 train file")

    top_manifest = {
        "stage": 6,
        "method": "nested_group_aware_training_subsets",
        "subset_seed": args.subset_seed,
        "fraction_definition": "query_count",
        "nested_order": list(FRACTIONS),
        "grouping_rule": "connected components of training queries linked by shared positive-relevance documents",
        "source_train_query_count": len(train_ids),
        "source_train_pair_count": sum(len(relevant) for relevant in train_qrels.values()),
        "source_component_count": len(components),
        "source_hashes": source_hashes,
        "fractions": fraction_manifests,
        "validation_query_count": len(validation_ids),
        "train_validation_relevant_document_overlap": 0,
        "test_split_loaded": False,
    }
    atomic_write_json(output_dir / "stage6_subset_manifest.json", top_manifest)
    print("STAGE6_NESTED_SUBSETS_READY")
    for percentage in FRACTIONS:
        row = fraction_manifests[str(percentage)]
        print(
            f"{percentage}%: queries={row['realized_query_count']} "
            f"pairs={row['training_pair_count']} steps/epoch={row['steps_per_epoch']} "
            f"eval_steps={row['evaluation_steps']}"
        )
    print(f"MANIFEST={output_dir / 'stage6_subset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
