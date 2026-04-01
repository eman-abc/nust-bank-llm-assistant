from eval.generate_golden_dataset import generate_golden_candidates


def test_generate_golden_candidates_is_deterministic():
    records = [
        {
            "source_type": "faq",
            "source_file": "funds_transfer_app_features_faq.json",
            "sheet": "Mobile App",
            "topic": "App Features",
            "question": "What should I do if I forget my login password?",
            "answer": "Tap on forgot password on the login screen.",
            "source_row_index": 0,
            "expected_doc_id": "funds-transfer-app-features-faq-json::0::0",
        },
        {
            "source_type": "processed",
            "source_file": "rate_sheet_knowledge.json",
            "sheet": "Rate Sheet",
            "topic": "Profit Rates",
            "question": "What is the profit rate for PLS Savings?",
            "answer": "For PLS Savings, the rate is 0.17.",
            "source_row_index": 1,
            "expected_doc_id": "rate-sheet-knowledge-json::1::1",
        },
    ]

    first = generate_golden_candidates(records)
    second = generate_golden_candidates(records)

    assert first == second
    assert first[0]["review_status"] == "pending_review"
    assert any(row["query_type"] == "rate_lookup" for row in first)
