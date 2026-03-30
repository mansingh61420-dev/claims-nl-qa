
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from claims_nl_qa.config import Settings
from claims_nl_qa.main import create_app
from claims_nl_qa.qa import QAError

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "docs" / "synthetic_claims.csv"


def test_root_lists_endpoints():
    """Hitting / in a browser should not be a 404—tiny map to docs, health, and /ask."""
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body.get("service") == "claims-nl-qa"
    assert "/docs" in body.get("docs", "")


def test_health_ok_and_row_count():
    """/health should come back fast and report how many rows we loaded."""
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["claims_rows"] == 60


def test_ask_qa_error_returns_400():
    """QAError from the pipeline should surface as 400 with the message in `detail`.

    Note: we patch `ask_question` instead of faking Settings—pydantic-settings will still pull
    OPENAI_API_KEY from `.env` even if you pass `openai_api_key=\"\"`, so empty-key tests are misleading.
    """
    app = create_app()
    with patch(
        "claims_nl_qa.main.ask_question",
        side_effect=QAError("OPENAI_API_KEY is not set. Add it to your environment or .env file."),
    ):
        with TestClient(app) as client:
            r = client.post("/ask", json={"question": "How many rows?"})

    assert r.status_code == 400
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_ask_rejects_too_long_question():
    """Input validation should reject unreasonably long questions at the API boundary."""
    app = create_app()
    payload = {"question": "a" * 1001}
    with TestClient(app) as client:
        r = client.post("/ask", json=payload)
    assert r.status_code == 422


def _openai_configured() -> bool:
    return bool(Settings(data_path=_CSV).openai_api_key.strip())


@pytest.mark.skipif(
    not (_openai_configured() and bool(__import__("os").environ.get("RUN_OPENAI_SMOKE"))),
    reason="Set RUN_OPENAI_SMOKE=1 and OPENAI_API_KEY to run real-API smoke tests",
)
def test_ask_happy_path_smoke():
    """Optional: full /ask round-trip against the real API."""
    app = create_app()
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "How many claims are there?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and "sql" in data
    assert data["row_count"] >= 1
    assert "SELECT" in data["sql"].upper()
