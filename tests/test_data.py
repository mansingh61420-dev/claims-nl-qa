from pathlib import Path

from claims_nl_qa.config import Settings
from claims_nl_qa.data import CLAIMS_TABLE, connect_with_claims, load_claims_frame, schema_description

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = _REPO_ROOT / "docs" / "synthetic_claims.csv"

_EXPECTED_COLUMNS = {
    "claim_id",
    "mrn",
    "patient_name",
    "dob",
    "ssn_masked",
    "provider",
    "facility",
    "payer",
    "service_date",
    "submission_date",
    "icd10_code",
    "diagnosis",
    "billed_amount",
    "allowed_amount",
    "paid_amount",
    "claim_status",
    "denial_reason",
    "length_of_stay_days",
    "clinical_note_snippet",
}


def test_load_claims_row_and_column_count():
    df = load_claims_frame(_DEFAULT_CSV)
    assert len(df) == 60
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_duckdb_register_and_schema():
    settings = Settings(data_path=_DEFAULT_CSV)
    con, _ = connect_with_claims(settings)
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {CLAIMS_TABLE}").fetchone()[0]
        assert count == 60
        desc = schema_description(con)
        assert "claim_id" in desc
        assert "service_date" in desc
    finally:
        con.close()


def test_settings_default_data_path_exists():
    s = Settings()
    assert s.data_path.name == "synthetic_claims.csv"
    assert s.data_path.is_file()
