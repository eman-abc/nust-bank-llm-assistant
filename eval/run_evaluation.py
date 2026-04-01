from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import get_settings
from backend.orchestrator import bank_bot
from eval.generate_golden_dataset import write_jsonl
from eval.metrics import citation_rank, exact_match, ndcg, numeric_consistency, reciprocal_rank, semantic_similarity, token_f1


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    rows = []

    for sample in samples:
        result = bank_bot.invoke({"user_query": sample["synthetic_query"]})
        citations = list(result.get("citations") or [])
        rank = citation_rank(sample["expected_doc_id"], citations)
        prediction = result.get("final_response", "")
        row = {
            "id": sample["id"],
            "query": sample["synthetic_query"],
            "expected_doc_id": sample["expected_doc_id"],
            "predicted_response": prediction,
            "citations": citations,
            "grounding_passed": bool(result.get("grounding_passed")),
            "retrieval_hit": 1.0 if rank else 0.0,
            "reciprocal_rank": reciprocal_rank(rank),
            "ndcg": ndcg(rank),
            "exact_match": exact_match(prediction, sample["expected_answer"]),
            "token_f1": token_f1(prediction, sample["expected_answer"]),
            "semantic_similarity": semantic_similarity(prediction, sample["expected_answer"]),
            "numeric_consistency": numeric_consistency(prediction, sample["expected_answer"])
            if sample["query_type"] == "rate_lookup"
            else 1.0,
        }
        rows.append(row)

    total = max(len(rows), 1)
    summary = {
        "samples": len(rows),
        "recall_at_k": sum(row["retrieval_hit"] for row in rows) / total,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / total,
        "ndcg": sum(row["ndcg"] for row in rows) / total,
        "citation_hit_rate": sum(row["retrieval_hit"] for row in rows) / total,
        "exact_match": sum(row["exact_match"] for row in rows) / total,
        "token_f1": sum(row["token_f1"] for row in rows) / total,
        "semantic_similarity": sum(row["semantic_similarity"] for row in rows) / total,
        "numeric_consistency": sum(row["numeric_consistency"] for row in rows) / total,
        "groundedness_pass_rate": sum(1.0 for row in rows if row["grounding_passed"]) / total,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation for the hybrid RAG graph.")
    parser.add_argument(
        "--dataset",
        default=str(get_settings().eval_artifacts_dir / "golden_candidates.jsonl"),
        help="Path to the JSONL dataset to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=str(get_settings().eval_artifacts_dir / "evaluation_report.json"),
        help="Where to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--predictions",
        default=str(get_settings().eval_artifacts_dir / "evaluation_predictions.jsonl"),
        help="Where to write per-sample prediction details.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = load_jsonl(dataset_path)
    prediction_rows, summary = evaluate_samples(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(prediction_rows, Path(args.predictions))
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
