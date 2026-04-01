from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import DATA_DIR, PROCESSED_DIR
from backend.services.qdrant_store import build_doc_id


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_faq_categories(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    records: list[dict[str, Any]] = []

    for category in data.get("categories", []):
        topic = category.get("category", "FAQ")
        for index, item in enumerate(category.get("questions", [])):
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question or not answer:
                continue
            source_file = path.name
            records.append(
                {
                    "source_type": "faq",
                    "source_file": source_file,
                    "sheet": "Mobile App",
                    "topic": topic,
                    "question": question,
                    "answer": answer,
                    "source_row_index": index,
                    "expected_doc_id": build_doc_id(source_file, index, index),
                }
            )
    return records


def _flatten_processed_records(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        return []

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        topic = (item.get("topic") or "Knowledge Base").strip()
        if not question or not answer:
            continue
        source_file = path.name
        source_row_index = int(item.get("source_row_index", index))
        records.append(
            {
                "source_type": "processed",
                "source_file": source_file,
                "sheet": (item.get("sheet") or "Knowledge Base").strip(),
                "topic": topic,
                "question": question,
                "answer": answer,
                "source_row_index": source_row_index,
                "expected_doc_id": build_doc_id(source_file, index, source_row_index),
            }
        )
    return records


def load_source_records(data_dir: Path | None = None, processed_dir: Path | None = None) -> list[dict[str, Any]]:
    data_dir = data_dir or DATA_DIR
    processed_dir = processed_dir or PROCESSED_DIR

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    faq_path = data_dir / "funds_transfer_app_features_faq.json"
    if faq_path.exists():
        records.extend(_flatten_faq_categories(faq_path))

    for path in sorted(processed_dir.glob("*.json")):
        records.extend(_flatten_processed_records(path))

    unique_records = []
    for record in records:
        fingerprint = (
            record["source_file"],
            record["topic"],
            record["question"],
            record["answer"],
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_records.append(record)
    return unique_records
