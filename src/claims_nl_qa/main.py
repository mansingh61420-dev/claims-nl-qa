
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from claims_nl_qa.config import Settings, get_settings
from claims_nl_qa.data import connect_with_claims
from claims_nl_qa.qa import QAError, ask_question

logger = logging.getLogger(__name__)


def get_settings_for_request() -> Settings:
    """Thin wrapper so tests can override settings without fighting lru_cache on get_settings."""
    return get_settings()


class AskRequest(BaseModel):
    """Body for POST /ask."""

    question: str = Field(..., min_length=1, description="Natural-language question about the claims data.")


class AskResponse(BaseModel):
    """Grounded answer plus the SQL we ran (handy for debugging and review)."""

    answer: str
    sql: str
    row_count: int
    truncated: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DuckDB once; tear down when the process exits."""
    settings = get_settings()
    con, df = connect_with_claims(settings)
    app.state.db_con = con
    app.state.db_lock = threading.Lock()
    app.state.row_count = len(df)
    logger.info("API ready: %s claims rows loaded", len(df))
    try:
        yield
    finally:
        con.close()
        logger.info("DuckDB connection closed")


def create_app() -> FastAPI:
    """Factory so tests can build a fresh app without import side effects."""
    app = FastAPI(
        title="Claims NL Q&A",
        description="Ask questions about the synthetic claims table in plain English.",
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        """So a browser hit on / isn't a bare 404—points people at the real endpoints."""
        return {
            "service": "claims-nl-qa",
            "docs": "/docs",
            "health": "GET /health",
            "ask": 'POST /ask with JSON body: {"question": "<your question>"}',
        }

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        """Liveness: no LLM call; confirms the process is up and data was loaded."""
        rows = getattr(request.app.state, "row_count", None)
        return {"status": "ok", "claims_rows": rows if rows is not None else 0}

    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings_for_request)],
    ) -> AskResponse:
        """Run the NL→SQL→answer pipeline (needs OPENAI_API_KEY)."""
        try:
            with request.app.state.db_lock:
                out = ask_question(request.app.state.db_con, settings, body.question)
            return AskResponse(
                answer=out.answer,
                sql=out.sql,
                row_count=out.row_count,
                truncated=out.truncated,
            )
        except QAError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
