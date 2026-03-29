from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from claims_nl_qa.config import Settings
from claims_nl_qa.data import connect_with_claims
from claims_nl_qa.qa import execute_readonly_sql

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "docs" / "synthetic_claims.csv"
_GOLDEN = Path(__file__).resolve().parent / "golden.yaml"


def _load_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(_GOLDEN.read_text(encoding="utf-8"))
    return list(data.get("cases") or [])


def _scalar_matches(actual: Any, expect: dict[str, Any]) -> None:
    typ = expect.get("type")
    want = expect["value"]
    if typ == "scalar":
        assert actual == want, f"got {actual!r}, want {want!r}"
    elif typ == "scalar_float":
        tol = float(expect.get("abs_tol", 1e-6))
        assert float(actual) == pytest.approx(float(want), abs=tol), f"got {actual!r}, want ~{want!r} (±{tol})"
    else:
        raise AssertionError(f"unknown expect.type: {typ!r}")


@pytest.fixture(scope="module")
def golden_con():
    """One DuckDB connection for all golden cases (same CSV as production)."""
    settings = Settings(data_path=_CSV)
    con, _df = connect_with_claims(settings)
    try:
        yield con
    finally:
        con.close()


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_golden_sql_matches_expected(golden_con, case: dict[str, Any]) -> None:
    """Each row in golden.yaml must run clean and match the expected scalar."""
    sql = case["sql"]
    expect = case["expect"]
    cols, rows, truncated = execute_readonly_sql(golden_con, sql)
    assert not truncated
    assert len(rows) == 1, f"{case['id']}: expected one row, got {len(rows)}"
    assert len(cols) >= 1
    _scalar_matches(rows[0][0], expect)
