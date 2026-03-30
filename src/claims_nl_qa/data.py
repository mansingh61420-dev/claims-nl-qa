from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from claims_nl_qa.config import Settings

logger = logging.getLogger(__name__)

CLAIMS_TABLE = "claims"

# Parse these as datetimes so SQL date filters behave; dob stays a string (messy formats).
_DATE_COLUMNS = ("service_date", "submission_date")


def load_claims_frame(path: Path) -> pd.DataFrame:
    """Read the CSV from disk; raises if the path is wrong."""
    if not path.is_file():
        raise FileNotFoundError(f"Claims CSV not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=list(_DATE_COLUMNS),
        dayfirst=False,
    )
    return df


def register_claims(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Expose the dataframe as a DuckDB table name the LLM will see in prompts."""
    con.register(CLAIMS_TABLE, df)


def schema_description(con: duckdb.DuckDBPyConnection, table: str = CLAIMS_TABLE) -> str:
    """Human-readable column list for stuffing into the SQL-generation prompt."""
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    lines = [f"Table `{table}` columns:"]
    for name, type_name, *_rest in rows:
        lines.append(f"  - {name}: {type_name}")
    return "\n".join(lines)


def connect_with_claims(settings: Settings) -> tuple[duckdb.DuckDBPyConnection, pd.DataFrame]:
    """In-memory DuckDB + registered frame—what we use for one-off questions."""
    df = load_claims_frame(settings.data_path)
    con = duckdb.connect(database=":memory:")
    register_claims(con, df)
    con.execute("SET enable_external_access=false")
    logger.info("Loaded %s rows into DuckDB table %r", len(df), CLAIMS_TABLE)
    return con, df
