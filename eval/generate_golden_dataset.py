from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from backend.config import get_settings
from eval.data_sources import load_source_records


def _make_typo_variant(text: str) -> str:
    words = text.split()
    if not words:
        return text
    longest_index = max(range(len(words)), key=lambda index: len(words[index]))
    token = words[longest_index]
    if len(token) < 5:
        return text
    chars = list(token)
    chars[1], chars[2] = chars[2], chars[1]
    words[longest_index] = "".join(chars)
    return " ".join(words)


def _clean_rate_record(record: dict[str, Any]) -> bool:
    answer = record["answer"].lower()
    question = record["question"].lower()
    if "profit rate |" in answer or "tenor |" in answer:
        return False
    if record["sheet"].lower() == "rate sheet" and not any(char.isdigit() for char in answer):
        return False
    if answer.count("|") > 5:
        return False
    return bool(question and answer)


def _variant_rows(record: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    question = record["question"].rstrip("?").strip()
    yield question + "?", "canonical", "easy"
    yield f"Can you tell me {question.lower()}?", "paraphrase", "medium"
    yield f"I'm using NUST Bank and need help with this: {question.lower()}?", "multi_turn", "medium"
    yield _make_typo_variant(question + "?"), "typo", "hard"


def generate_golden_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        if record["source_type"] != "faq" and not _clean_rate_record(record):
            continue

        confidence = 0.95 if record["source_type"] == "faq" else 0.75
        if record["sheet"].lower() == "rate sheet":
            confidence = min(confidence, 0.65)

        query_type = "rate_lookup" if record["sheet"].lower() == "rate sheet" else "faq"
        evidence_text = f"Question: {record['question']}\nAnswer: {record['answer']}"

        for variant_index, (synthetic_query, variant_name, difficulty) in enumerate(_variant_rows(record)):
            dataset.append(
                {
                    "id": f"{record['expected_doc_id']}::{variant_name}",
                    "source_type": record["source_type"],
                    "source_file": record["source_file"],
                    "topic": record["topic"],
                    "question": record["question"],
                    "synthetic_query": synthetic_query,
                    "expected_answer": record["answer"],
                    "evidence_text": evidence_text,
                    "expected_doc_id": record["expected_doc_id"],
                    "query_type": query_type,
                    "difficulty": difficulty,
                    "confidence": confidence,
                    "review_status": "pending_review",
                }
            )

    return dataset


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic golden-candidate dataset.")
    parser.add_argument(
        "--output",
        default=str(get_settings().eval_artifacts_dir / "golden_candidates.jsonl"),
        help="Where to write the generated JSONL dataset.",
    )
    args = parser.parse_args()

    records = load_source_records()
    dataset = generate_golden_candidates(records)
    output_path = Path(args.output)
    write_jsonl(dataset, output_path)
    print(f"Wrote {len(dataset)} golden candidates to {output_path}")


if __name__ == "__main__":
    main()
