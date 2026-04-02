from __future__ import annotations

import re
from typing import Any

import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _split_sentences(text: str) -> list[str]:
    """Split content into sentence-like pieces for policy-style chunk assembly."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_healthcare_documents(
    docs_df: pd.DataFrame,
    *,
    max_chars: int = 220,
    overlap_sentences: int = 1,
) -> pd.DataFrame:
    """Create coherent chunks while preserving metadata breadcrumbs."""
    rows: list[dict[str, Any]] = []

    for doc in docs_df.to_dict(orient="records"):
        content = str(doc.get("content", "")).strip()
        if not content:
            continue

        sentences = _split_sentences(content)
        if not sentences:
            continue

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + 1 + sentence_len > max_chars:
                chunks.append(" ".join(current).strip())
                overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current = [*overlap, sentence]
                current_len = len(" ".join(current))
            else:
                current.append(sentence)
                current_len = len(" ".join(current))

        if current:
            chunks.append(" ".join(current).strip())

        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{doc['document_id']}::chunk_{idx}",
                    "document_id": doc["document_id"],
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_text": chunk_text,
                    # Breadcrumbs to trace answer provenance.
                    "source_type": doc["source_type"],
                    "effective_date": doc["effective_date"],
                    "region": doc["region"],
                    "payer": doc["payer"],
                    "product_line": doc["product_line"],
                    "doc_version": doc["doc_version"],
                }
            )

    return pd.DataFrame(rows)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _score_chunk(question: str, chunk_text: str) -> int:
    """Deterministic lexical overlap score for stable ranking."""
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(chunk_text)
    return len(q_tokens & c_tokens)


def retrieve_relevant_chunks(
    chunks_df: pd.DataFrame,
    question: str,
    *,
    top_k: int = 5,
    metadata_filters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply metadata filters and stable top-k ranking for retrieval."""
    filtered = chunks_df.copy()
    metadata_filters = metadata_filters or {}

    for key, value in metadata_filters.items():
        if key not in filtered.columns or value is None:
            continue
        filtered = filtered[filtered[key] == value]

    if filtered.empty:
        return filtered

    out = filtered.copy()
    out["retrieval_score"] = out["chunk_text"].map(lambda text: _score_chunk(question, str(text)))
    out = out.sort_values(
        by=["retrieval_score", "document_id", "chunk_index"],
        ascending=[False, True, True],
    )
    return out.head(max(top_k, 0)).reset_index(drop=True)
