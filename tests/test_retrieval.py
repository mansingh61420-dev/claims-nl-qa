from pathlib import Path

from claims_nl_qa.data import build_healthcare_documents, load_claims_frame
from claims_nl_qa.retrieval import chunk_healthcare_documents, retrieve_relevant_chunks

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = _REPO_ROOT / "docs" / "synthetic_claims.csv"


def test_chunking_produces_traceable_rows():
    """Each chunk keeps doc breadcrumbs and stable chunk metadata."""
    claims_df = load_claims_frame(_DEFAULT_CSV)
    docs_df = build_healthcare_documents(claims_df).head(3)
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=80, overlap_sentences=1)

    assert not chunks_df.empty
    assert {"chunk_id", "document_id", "chunk_index", "total_chunks", "chunk_text"}.issubset(
        chunks_df.columns
    )
    assert {"source_type", "effective_date", "region", "payer", "product_line", "doc_version"}.issubset(
        chunks_df.columns
    )


def test_chunking_assigns_total_chunks_per_document():
    """All chunks from one document should share the same total_chunks count."""
    claims_df = load_claims_frame(_DEFAULT_CSV)
    docs_df = build_healthcare_documents(claims_df).head(1)
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=60, overlap_sentences=1)

    assert not chunks_df.empty
    expected_total = int(chunks_df["total_chunks"].iloc[0])
    assert expected_total == len(chunks_df)
    assert chunks_df["total_chunks"].nunique() == 1


def test_retrieve_relevant_chunks_respects_metadata_filters():
    """Retrieval should honor exact-match metadata filters before scoring."""
    claims_df = load_claims_frame(_DEFAULT_CSV)
    docs_df = build_healthcare_documents(claims_df)
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=120)

    payer_value = str(docs_df["payer"].iloc[0])
    out = retrieve_relevant_chunks(
        chunks_df,
        "denied claim diagnosis at facility",
        top_k=5,
        metadata_filters={"payer": payer_value},
    )

    assert not out.empty
    assert out["payer"].nunique() == 1
    assert str(out["payer"].iloc[0]) == payer_value


def test_retrieve_relevant_chunks_returns_stable_top_k():
    """Top-k results should be bounded and sorted by score descending."""
    claims_df = load_claims_frame(_DEFAULT_CSV)
    docs_df = build_healthcare_documents(claims_df).head(10)
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=100)

    out = retrieve_relevant_chunks(chunks_df, "diagnosis denied claim status", top_k=3)

    assert len(out) <= 3
    assert "retrieval_score" in out.columns
    assert out["retrieval_score"].tolist() == sorted(out["retrieval_score"].tolist(), reverse=True)
