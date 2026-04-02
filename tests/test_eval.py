from pathlib import Path

from claims_nl_qa.data import build_healthcare_documents, load_claims_frame
from claims_nl_qa.eval import summarize_retrieval
from claims_nl_qa.retrieval import chunk_healthcare_documents, retrieve_relevant_chunks

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "docs" / "synthetic_claims.csv"


def test_eval_retrieval_hits_denial_keywords():
    """Healthcare-style question should retrieve chunks mentioning denial-related terms."""
    claims_df = load_claims_frame(_CSV)
    docs_df = build_healthcare_documents(claims_df)
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=200)
    summary = summarize_retrieval(
        chunks_df,
        "How many claims were denied and what was the denial reason?",
        top_k=5,
        must_contain_any=("Denied", "denial"),
    )
    assert summary["top_k"] >= 1
    assert summary["max_score"] >= 1
    assert summary["passed_keywords"]


def test_eval_retrieval_respects_payer_filter():
    """Metadata filter narrows retrieval to chunks whose doc metadata matches payer."""
    claims_df = load_claims_frame(_CSV)
    docs_df = build_healthcare_documents(claims_df)
    payer = str(docs_df.loc[docs_df["payer"] == "Medicare", "payer"].iloc[0])
    chunks_df = chunk_healthcare_documents(docs_df, max_chars=200)
    ranked = retrieve_relevant_chunks(
        chunks_df,
        "claim diagnosis and facility",
        top_k=5,
        metadata_filters={"payer": payer},
    )
    assert not ranked.empty
    assert (ranked["payer"] == payer).all()
