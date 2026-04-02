from __future__ import annotations

from typing import Any

import pandas as pd

from claims_nl_qa.retrieval import retrieve_relevant_chunks


def summarize_retrieval(
    chunks_df: pd.DataFrame,
    question: str,
    *,
    top_k: int = 5,
    metadata_filters: dict[str, Any] | None = None,
    must_contain_any: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run top-k retrieval and report overlap with optional keyword expectations."""
    ranked = retrieve_relevant_chunks(
        chunks_df,
        question,
        top_k=top_k,
        metadata_filters=metadata_filters,
    )
    if ranked.empty:
        return {
            "top_k": 0,
            "max_score": 0,
            "keyword_hits": 0,
            "passed_keywords": False,
        }

    blob = " ".join(ranked["chunk_text"].astype(str)).lower()
    hits = sum(1 for token in must_contain_any if token.lower() in blob)
    max_score = int(ranked["retrieval_score"].iloc[0])
    return {
        "top_k": len(ranked),
        "max_score": max_score,
        "keyword_hits": hits,
        "passed_keywords": bool(not must_contain_any or hits > 0),
    }
