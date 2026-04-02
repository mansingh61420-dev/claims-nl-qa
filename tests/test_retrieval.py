from pathlib import Path

from claims_nl_qa.data import build_healthcare_documents, load_claims_frame
from claims_nl_qa.retrieval import chunk_healthcare_documents

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
