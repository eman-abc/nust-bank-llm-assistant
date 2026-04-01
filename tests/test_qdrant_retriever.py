from backend.nodes import hybrid_retriever
from backend.services.sparse_encoder import SparseEmbedding


def test_run_hybrid_retriever_queries_qdrant_live(monkeypatch):
    seen = {}

    def fake_embed_text(text):
        seen["dense_text"] = text
        return [0.5, 0.25, 0.75]

    def fake_encode_sparse_text(text):
        seen["sparse_text"] = text
        return SparseEmbedding(indices=[1, 2], values=[1.0, 0.8])

    def fake_hybrid_search(*, dense_vector, sparse_vector, limit, metadata_filters):
        seen["dense_vector"] = dense_vector
        seen["sparse_vector"] = sparse_vector
        seen["limit"] = limit
        seen["metadata_filters"] = metadata_filters
        return [
            {
                "id": "doc-1",
                "score": 0.42,
                "payload": {"chunk_text": "Roshan Digital Account requirements"},
            }
        ]

    monkeypatch.setattr(hybrid_retriever, "embed_text", fake_embed_text)
    monkeypatch.setattr(hybrid_retriever, "encode_sparse_text", fake_encode_sparse_text)
    monkeypatch.setattr(hybrid_retriever, "hybrid_search", fake_hybrid_search)

    result = hybrid_retriever.run_hybrid_retriever(
        {
            "normalized_query": "How do I open a Roshan Digital Account?",
            "metadata_filters": {"sheet": "Mobile App"},
        }
    )

    assert seen["dense_text"] == "How do I open a Roshan Digital Account?"
    assert seen["sparse_text"] == "How do I open a Roshan Digital Account?"
    assert seen["dense_vector"] == [0.5, 0.25, 0.75]
    assert seen["metadata_filters"] == {"sheet": "Mobile App"}
    assert result["retrieval_candidates"][0]["payload"]["chunk_text"] == "Roshan Digital Account requirements"
