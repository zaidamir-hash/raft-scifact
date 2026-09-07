#!/usr/bin/env python3
"""Download and prepare BEIR SciFact for the RAFT retrieval experiments.

This Stage 1 script uses only Python's standard library, so it runs in a
fresh Google Colab runtime without installing project dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_data_prep.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "scifact"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "scifact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace the cached BEIR archive before preparing the data.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"dataset_url", "validation_fraction", "split_seed"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    fraction = float(config["validation_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using cached archive: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "RAFT-stage1/1.0"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a zip while rejecting members that escape the target directory."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zipped.extractall(destination)


def ensure_extracted(archive: Path, destination: Path, force: bool = False) -> Path:
    """Return a complete extraction, repairing interrupted prior attempts."""
    if destination.exists() and not force:
        try:
            return locate_beir_dataset(destination)
        except RuntimeError:
            print(f"Replacing incomplete extraction: {destination}")

    temporary = destination.with_name(destination.name + ".extracting")
    for path in (temporary, destination):
        if path.exists():
            shutil.rmtree(path)
    print(f"Extracting {archive}")
    safe_extract(archive, temporary)
    dataset_relative = locate_beir_dataset(temporary).relative_to(temporary)
    temporary.replace(destination)
    return destination / dataset_relative


def locate_beir_dataset(extracted_root: Path) -> Path:
    candidates = []
    for corpus_path in extracted_root.rglob("corpus.jsonl"):
        parent = corpus_path.parent
        if (
            (parent / "queries.jsonl").is_file()
            and (parent / "qrels" / "train.tsv").is_file()
            and (parent / "qrels" / "test.tsv").is_file()
        ):
            candidates.append(parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one BEIR dataset directory, found {len(candidates)}"
        )
    return candidates[0]


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("_id", record.get("id", "")))
            if not record_id:
                raise ValueError(f"Missing ID in {path} line {line_number}")
            if record_id in records:
                raise ValueError(f"Duplicate ID {record_id!r} in {path}")
            records[record_id] = record
    return records


def parse_score(value: str) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def load_qrels(path: Path) -> list[tuple[str, str, int | float]]:
    rows: list[tuple[str, str, int | float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected qrels header in {path}: {reader.fieldnames}")
        for row in reader:
            rows.append(
                (str(row["query-id"]), str(row["corpus-id"]), parse_score(row["score"]))
            )
    return rows


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_query_components(
    query_ids: set[str], qrels: list[tuple[str, str, int | float]]
) -> list[list[str]]:
    """Return connected query groups induced by shared relevant documents."""
    union_find = UnionFind(query_ids)
    document_queries: dict[str, list[str]] = defaultdict(list)
    for query_id, document_id, score in qrels:
        if score > 0:
            document_queries[document_id].append(query_id)

    for linked_queries in document_queries.values():
        anchor = linked_queries[0]
        for query_id in linked_queries[1:]:
            union_find.union(anchor, query_id)

    components: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        components[union_find.find(query_id)].append(query_id)
    # Canonicalize before the seeded shuffle so PYTHONHASHSEED and platform
    # iteration order cannot change the split produced by split_seed.
    canonical_components = [sorted(component) for component in components.values()]
    return sorted(canonical_components, key=lambda component: tuple(component))


def choose_validation_components(
    components: list[list[str]], validation_fraction: float, seed: int
) -> tuple[set[str], int]:
    """Select whole components nearest to the requested query fraction.

    A seeded shuffle supplies deterministic tie-breaking. Dynamic programming
    then finds a component subset whose query count is closest to the target.
    """
    shuffled = [list(component) for component in components]
    random.Random(seed).shuffle(shuffled)
    query_count = sum(len(component) for component in shuffled)
    target = round(query_count * validation_fraction)

    # Maps attainable query counts to one corresponding tuple of component indices.
    attainable: dict[int, tuple[int, ...]] = {0: ()}
    for index, component in enumerate(shuffled):
        for count, selected in list(attainable.items()):
            new_count = count + len(component)
            if new_count not in attainable:
                attainable[new_count] = selected + (index,)

    eligible = [count for count in attainable if 0 < count < query_count]
    if not eligible:
        raise RuntimeError("Cannot produce non-empty train and validation splits")
    selected_count = min(
        eligible,
        key=lambda count: (abs(count - target), count > target, count),
    )
    selected_indices = attainable[selected_count]
    validation_ids = {
        query_id
        for index in selected_indices
        for query_id in shuffled[index]
    }
    return validation_ids, target


def stable_id_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value, value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_qrels(path: Path, rows: list[tuple[str, str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in sorted(rows, key=lambda item: (stable_id_key(item[0]), stable_id_key(item[1]))):
            writer.writerow(row)
    temporary.replace(path)


def relevant_documents(
    qrels: list[tuple[str, str, int | float]], query_ids: set[str]
) -> set[str]:
    return {
        document_id
        for query_id, document_id, score in qrels
        if query_id in query_ids and score > 0
    }


def prepare(config: dict[str, Any], raw_dir: Path, output_dir: Path, force: bool) -> dict[str, Any]:
    archive = raw_dir / "scifact.zip"
    extracted = raw_dir / "extracted"
    download(str(config["dataset_url"]), archive, force=force)
    dataset_dir = ensure_extracted(archive, extracted, force=force)
    corpus = load_jsonl(dataset_dir / "corpus.jsonl")
    queries = load_jsonl(dataset_dir / "queries.jsonl")
    official_train_qrels = load_qrels(dataset_dir / "qrels" / "train.tsv")
    official_test_qrels = load_qrels(dataset_dir / "qrels" / "test.tsv")

    official_train_ids = {row[0] for row in official_train_qrels}
    official_test_ids = {row[0] for row in official_test_qrels}
    if official_train_ids & official_test_ids:
        raise AssertionError("Official BEIR train and test query IDs overlap")
    missing_queries = (official_train_ids | official_test_ids) - queries.keys()
    if missing_queries:
        raise AssertionError(f"Qrels reference {len(missing_queries)} missing queries")
    missing_documents = {
        row[1] for row in official_train_qrels + official_test_qrels
    } - corpus.keys()
    if missing_documents:
        raise AssertionError(f"Qrels reference {len(missing_documents)} missing documents")

    components = build_query_components(official_train_ids, official_train_qrels)
    validation_ids, target_validation_queries = choose_validation_components(
        components,
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["split_seed"]),
    )
    train_ids = official_train_ids - validation_ids

    # Leakage prevention: claims that share any positive source document are in
    # one connected component (including transitive multi-document links), and
    # whole components—not individual query/passage pairs—are assigned to a split.
    # Consequently, a relevant source document can never occur in both train and
    # validation, avoiding document-specific leakage during checkpoint selection.
    train_documents = relevant_documents(official_train_qrels, train_ids)
    validation_documents = relevant_documents(official_train_qrels, validation_ids)
    shared_documents = train_documents & validation_documents
    if shared_documents:
        raise AssertionError(
            f"Leakage detected: {len(shared_documents)} relevant documents cross splits"
        )

    train_qrels = [row for row in official_train_qrels if row[0] in train_ids]
    validation_qrels = [row for row in official_train_qrels if row[0] in validation_ids]

    corpus_rows = (
        {
            "id": document_id,
            "title": str(corpus[document_id].get("title", "")),
            "text": str(corpus[document_id].get("text", "")),
        }
        for document_id in sorted(corpus, key=stable_id_key)
    )
    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)

    split_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "test": official_test_ids,
    }
    for split, query_ids in split_ids.items():
        query_rows = (
            {"id": query_id, "text": str(queries[query_id].get("text", ""))}
            for query_id in sorted(query_ids, key=stable_id_key)
        )
        write_jsonl(output_dir / f"queries_{split}.jsonl", query_rows)

    write_qrels(output_dir / "qrels_train.tsv", train_qrels)
    write_qrels(output_dir / "qrels_validation.tsv", validation_qrels)
    # The official test query IDs and qrels are copied as a whole and never sampled.
    write_qrels(output_dir / "qrels_test.tsv", official_test_qrels)

    written_test_queries = load_jsonl(output_dir / "queries_test.jsonl")
    written_test_qrels = load_qrels(output_dir / "qrels_test.tsv")
    official_test_preserved = (
        set(written_test_queries) == official_test_ids
        and all(
            written_test_queries[query_id].get("text", "")
            == str(queries[query_id].get("text", ""))
            for query_id in official_test_ids
        )
        and sorted(written_test_qrels) == sorted(official_test_qrels)
    )
    if not official_test_preserved:
        raise AssertionError("Written test queries or qrels differ from official BEIR test data")

    total_passage_tokens = sum(
        len(
            " ".join(
                part
                for part in (
                    str(document.get("title", "")).strip(),
                    str(document.get("text", "")).strip(),
                )
                if part
            ).split()
        )
        for document in corpus.values()
    )
    average_passage_length = total_passage_tokens / len(corpus) if corpus else math.nan
    component_sizes = sorted((len(component) for component in components), reverse=True)
    stats = {
        "average_passage_length_whitespace_tokens": round(average_passage_length, 2),
        "documents": len(corpus),
        "official_training_queries": len(official_train_ids),
        "queries": {split: len(ids) for split, ids in split_ids.items()},
        "qrels": {
            "train": len(train_qrels),
            "validation": len(validation_qrels),
            "test": len(official_test_qrels),
        },
        "requested_validation_fraction": float(config["validation_fraction"]),
        "realized_validation_fraction": round(
            len(validation_ids) / len(official_train_ids), 6
        ),
        "split_seed": int(config["split_seed"]),
        "target_validation_queries": target_validation_queries,
        "query_document_components": len(components),
        "largest_component_queries": component_sizes[0],
        "train_relevant_documents": len(train_documents),
        "validation_relevant_documents": len(validation_documents),
        "train_validation_relevant_document_overlap": len(shared_documents),
        "official_test_preserved": official_test_preserved,
    }
    manifest = {
        "config": config,
        "source_archive": str(archive),
        "source_archive_sha256": sha256_file(archive),
        "source_dataset_directory": str(dataset_dir),
        "output_directory": str(output_dir),
        "files": [
            "corpus.jsonl",
            "queries_train.jsonl",
            "queries_validation.jsonl",
            "queries_test.jsonl",
            "qrels_train.tsv",
            "qrels_validation.tsv",
            "qrels_test.tsv",
            "sanity_stats.json",
            "split_manifest.json",
            "stage1_config.snapshot.json",
        ],
    }
    write_json(output_dir / "sanity_stats.json", stats)
    write_json(output_dir / "split_manifest.json", manifest)
    write_json(output_dir / "stage1_config.snapshot.json", config)
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    queries = stats["queries"]
    print("\nStage 1 sanity stats")
    print(f"Documents: {stats['documents']:,}")
    print(
        "Queries: "
        f"train={queries['train']:,}, validation={queries['validation']:,}, "
        f"test={queries['test']:,}"
    )
    print(
        "Average passage length: "
        f"{stats['average_passage_length_whitespace_tokens']:.2f} whitespace tokens "
        "(title + text)"
    )
    print(
        "Train/validation relevant-document overlap: "
        f"{stats['train_validation_relevant_document_overlap']}"
    )
    print(f"Official test preserved unchanged: {stats['official_test_preserved']}")
    print(
        "Validation split: "
        f"requested={stats['requested_validation_fraction']:.1%}, "
        f"realized={stats['realized_validation_fraction']:.1%}, "
        f"seed={stats['split_seed']}"
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    stats = prepare(
        config=config,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        force=args.force_download,
    )
    print_stats(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
