from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from claims_nl_qa.config import Settings

logger = logging.getLogger(__name__)

CLAIMS_TABLE = "claims"
HEALTHCARE_DOCS_TABLE = "healthcare_docs"

# Parse these as datetimes so SQL date filters behave; dob stays a string (messy formats).
_DATE_COLUMNS = ("service_date", "submission_date")
_HEALTHCARE_METADATA_COLUMNS = (
    "source_type",
    "effective_date",
    "region",
    "payer",
    "product_line",
    "doc_version",
)


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


def build_healthcare_documents(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Build retrieval-ready healthcare records with required metadata."""
    docs = claims_df.copy()
    docs["source_type"] = "synthetic_claim"
    docs["effective_date"] = docs["submission_date"]
    docs["region"] = "unknown"
    docs["product_line"] = "unknown"
    docs["doc_version"] = "v1"
    docs["document_id"] = docs["claim_id"].astype(str)
    docs["content"] = (
        "Claim "
        + docs["claim_id"].astype(str)
        + " for patient "
        + docs["patient_name"].astype(str)
        + " at "
        + docs["facility"].astype(str)
        + " with diagnosis "
        + docs["diagnosis"].astype(str)
        + " and status "
        + docs["claim_status"].astype(str)
        + "."
    )
    return docs[["document_id", "content", *_HEALTHCARE_METADATA_COLUMNS]].copy()


def validate_healthcare_metadata(docs_df: pd.DataFrame) -> None:
    """Raise if required metadata columns are missing or contain nulls."""
    missing = [name for name in _HEALTHCARE_METADATA_COLUMNS if name not in docs_df.columns]
    if missing:
        raise ValueError(f"Missing healthcare metadata columns: {', '.join(missing)}")

    null_columns = [name for name in _HEALTHCARE_METADATA_COLUMNS if docs_df[name].isna().any()]
    if null_columns:
        raise ValueError(f"Null values found in metadata columns: {', '.join(null_columns)}")


def register_healthcare_documents(
    con: duckdb.DuckDBPyConnection, docs_df: pd.DataFrame
) -> None:
    """Validate and expose healthcare docs to DuckDB."""
    validate_healthcare_metadata(docs_df)
    con.register(HEALTHCARE_DOCS_TABLE, docs_df)


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
    docs_df = build_healthcare_documents(df)
    con = duckdb.connect(database=":memory:")
    register_claims(con, df)
    register_healthcare_documents(con, docs_df)
    con.execute("SET enable_external_access=false")
    logger.info(
        "Loaded %s rows into %r and %s rows into %r",
        len(df),
        CLAIMS_TABLE,
        len(docs_df),
        HEALTHCARE_DOCS_TABLE,
    )
    return con, df
