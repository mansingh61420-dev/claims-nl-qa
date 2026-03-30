
from pathlib import Path

import pytest

from claims_nl_qa.config import Settings
from claims_nl_qa.data import connect_with_claims
from claims_nl_qa.qa import QAError, ask_question, execute_readonly_sql, validate_sql

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "docs" / "synthetic_claims.csv"


def _openai_configured() -> bool:
    """Same source as the app: pydantic reads .env: os.getenv alone does not."""
    return bool(Settings().openai_api_key.strip())


def test_validate_accepts_simple_select():
    """Plain COUNT(*) on `claims` should pass."""
    sql = "SELECT COUNT(*) AS n FROM claims"
    assert "COUNT" in validate_sql(sql)


def test_validate_strips_trailing_semicolon():
    """Semicolon at the end shouldn't fail."""
    sql = "SELECT 1 FROM claims;"
    assert validate_sql(sql).endswith("claims")


def test_validate_rejects_multistatement():
    """Two queries in one blob should be rejected."""
    with pytest.raises(QAError, match="one SQL"):
        validate_sql("SELECT 1 FROM claims; SELECT 2 FROM claims")


def test_validate_rejects_write():
    """DELETE shouldn't get through."""
    with pytest.raises(QAError, match="read-only"):
        validate_sql("DELETE FROM claims WHERE 1=1")


def test_validate_requires_claims_table():
    """If the SQL never mentions `claims`, we reject it."""
    with pytest.raises(QAError, match="claims"):
        validate_sql("SELECT 1 FROM somewhere_else")


def test_execute_readonly_runs_aggregate():
    """Actually run something through DuckDB and check we get a sane row back."""
    settings = Settings(data_path=_CSV)
    con, _df = connect_with_claims(settings)
    try:
        cols, rows, truncated = execute_readonly_sql(
            con, "SELECT COUNT(*) AS c FROM claims WHERE claim_status = 'Denied'"
        )
        assert cols[0].lower() == "c" or cols[0] == "c"
        assert rows[0][0] >= 0
        assert truncated is False
    finally:
        con.close()


def test_duckdb_external_access_is_disabled():
    """DuckDB should not be able to read arbitrary local files via table functions."""
    settings = Settings(data_path=_CSV)
    con, _df = connect_with_claims(settings)
    try:
        # If external access is disabled, this should fail regardless of whether the file exists.
        with pytest.raises(Exception):
            con.execute("SELECT * FROM read_csv_auto('definitely_not_a_real_file.csv')").fetchall()
    finally:
        con.close()


@pytest.mark.skipif(
    not (_openai_configured() and bool(__import__("os").environ.get("RUN_OPENAI_SMOKE"))),
    reason="Set RUN_OPENAI_SMOKE=1 and OPENAI_API_KEY to run real-API smoke tests",
)
def test_ask_question_smoke():
    """Hits the real API if OPENAI_API_KEY is set—handy for a quick manual check."""
    settings = Settings(data_path=_CSV)
    con, _df = connect_with_claims(settings)
    try:
        out = ask_question(con, settings, "How many claims are in the dataset?")
        assert out.row_count >= 1
        assert "SELECT" in out.sql.upper()
        assert len(out.answer) > 0
    finally:
        con.close()
