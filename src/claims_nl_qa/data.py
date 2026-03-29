from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from claims_nl_qa.config import Settings

logger = logging.getLogger(__name__)

CLAIMS_TABLE = "claims"

_DATE_COLUMNS = ("service_date", "submission_date")


def load_claims_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Claims CSV not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=list(_DATE_COLUMNS),
        dayfirst=False,
    )
    return df


def register_claims(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    con.register(CLAIMS_TABLE, df)


def schema_description(con: duckdb.DuckDBPyConnection, table: str = CLAIMS_TABLE) -> str:
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    lines = [f"Table `{table}` columns:"]
    for name, type_name, *_rest in rows:
        lines.append(f"  - {name}: {type_name}")
    return "\n".join(lines)


def connect_with_claims(settings: Settings) -> tuple[duckdb.DuckDBPyConnection, pd.DataFrame]:
    df = load_claims_frame(settings.data_path)
    con = duckdb.connect(database=":memory:")
    register_claims(con, df)
    logger.info("Loaded %s rows into DuckDB table %r", len(df), CLAIMS_TABLE)
    return con, df
