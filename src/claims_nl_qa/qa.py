from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import duckdb
from openai import OpenAI

from claims_nl_qa.config import Settings
from claims_nl_qa.data import CLAIMS_TABLE, schema_description

logger = logging.getLogger(__name__)

# Don't fetch more than this; keeps the answer prompt small too.
_DEFAULT_MAX_RESULT_ROWS = 500

# If any of these show up, we bail—user only gets SELECTs.
_BANNED_STATEMENT = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|REPLACE|TRUNCATE|DROP|ALTER|CREATE|ATTACH|DETACH|"
    r"COPY|PRAGMA|CALL|EXPORT|IMPORT|CHECKPOINT|VACUUM)\b",
    re.IGNORECASE | re.DOTALL,
)


class QAError(Exception):
    """Show this message to the user; it's meant to be readable, not a stack trace."""


@dataclass(frozen=True)
class QAResult:
    """Answer text, the SQL we ran, row count, and whether we hit the row limit."""

    answer: str
    sql: str
    row_count: int
    truncated: bool


def _client(settings: Settings) -> OpenAI:
    """Small wrapper so we're not sprinkling OpenAI() everywhere."""
    return OpenAI(api_key=settings.openai_api_key)


def _normalize_sql(sql: str) -> str:
    """Strip whitespace; one trailing `;` is fine, two statements in one string is not."""
    s = sql.strip()
    core = s.rstrip(";").strip()
    if ";" in core:
        raise QAError("Only one SQL statement is allowed (no multiple statements).")
    return core


def validate_sql(sql: str) -> str:
    """Must be one SELECT-ish statement that mentions `claims`. Good enough for this dataset."""
    s = _normalize_sql(sql)
    if not s:
        raise QAError("The model returned empty SQL.")

    if _BANNED_STATEMENT.search(s):
        raise QAError("Only read-only SELECT queries are allowed.")

    head = s.lstrip()
    upper = head.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise QAError("Query must start with SELECT or WITH.")

    # Cheap check: query has to touch our table (stops random SELECT 1 nonsense).
    if re.search(r"\bclaims\b", s, re.IGNORECASE) is None:
        raise QAError(f"Query must read from the `{CLAIMS_TABLE}` table.")

    return s


def _preview_rows(columns: list[str], rows: list[tuple[Any, ...]], max_lines: int = 40) -> str:
    """Flatten to a string the second LLM can read without us sending a huge blob."""
    lines: list[str] = [", ".join(columns)]
    for row in rows[:max_lines]:
        lines.append(", ".join("" if v is None else str(v) for v in row))
    if len(rows) > max_lines:
        lines.append(f"... ({len(rows) - max_lines} more rows not shown in preview)")
    return "\n".join(lines)


def execute_readonly_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    max_rows: int = _DEFAULT_MAX_RESULT_ROWS,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Run inside a subquery + LIMIT so even silly SQL can't return unbounded rows."""
    validated = validate_sql(sql)
    wrapped = f"SELECT * FROM ({validated}) AS _subq LIMIT {max_rows + 1}"
    try:
        rel = con.execute(wrapped)
        description = rel.description or []
        columns = [d[0] for d in description]
        rows = rel.fetchall()
    except Exception as exc:  # DuckDB errors vary; user gets a short line
        logger.exception("SQL execution failed")
        raise QAError(f"Could not run the generated query: {exc}") from exc

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    return columns, rows, truncated


def _generate_sql(question: str, schema_text: str, settings: Settings) -> str:
    """Call #1: JSON with a `sql` field. Temperature low so it doesn't get creative."""
    system = (
        "You're writing DuckDB SQL for someone reviewing synthetic claims data. "
        f"There is only one table: `{CLAIMS_TABLE}`. Schema:\n\n"
        f"{schema_text}\n\n"
        f"Write one SELECT (or WITH ... SELECT) that answers their question. "
        f"Use those column names as-is. COUNT/SUM/AVG is fine when it fits.\n"
        f"Output nothing but JSON: {{\"sql\": \"...your single statement...\"}}. "
        f"No markdown, no explanation."
    )
    user = question.strip()

    client = _client(settings)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QAError("The model returned invalid JSON for SQL.") from exc
    sql = data.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise QAError("The model did not return a usable `sql` string.")
    return sql


def _answer_from_results(
    question: str,
    sql: str,
    preview: str,
    settings: Settings,
) -> str:
    """Call #2: plain English, but only from the preview string—no inventing numbers."""
    system = (
        "Someone ran SQL on a claims table and got the block below. "
        "Answer their question using only that block. "
        "If it's empty or all zeros where that matters, say so—don't guess. "
        "Keep it short."
    )
    user = (
        f"What they asked:\n{question.strip()}\n\n"
        f"SQL (for context):\n{sql}\n\n"
        f"What came back:\n{preview or '(no rows)'}\n"
    )

    client = _client(settings)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise QAError("The model returned an empty answer.")
    return text


def ask_question(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    question: str,
    *,
    max_result_rows: int = _DEFAULT_MAX_RESULT_ROWS,
) -> QAResult:
    """Wire everything together. Needs a real API key or we raise before calling OpenAI."""
    if not settings.openai_api_key.strip():
        raise QAError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    q = question.strip()
    if not q:
        raise QAError("Question is empty.")

    schema_text = schema_description(con)
    sql = _generate_sql(q, schema_text, settings)
    sql = validate_sql(sql)

    columns, rows, truncated = execute_readonly_sql(con, sql, max_rows=max_result_rows)
    preview = _preview_rows(columns, rows)
    answer = _answer_from_results(q, sql, preview, settings)

    return QAResult(
        answer=answer,
        sql=sql,
        row_count=len(rows),
        truncated=truncated,
    )
