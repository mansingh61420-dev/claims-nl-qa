# claims-nl-qa

Natural-language Q&A over synthetic healthcare claims. Setup and architecture documentation will be expanded as the project is built.

**Requires Python 3.10+.** Install the package in editable mode with `pip install -e ".[dev]"` from the repository root (use a 3.10+ interpreter).

### Run the API

From the repo root (with `.env` containing `OPENAI_API_KEY`):

```bash
uvicorn claims_nl_qa.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /health` — quick check that the app is up and claims are loaded.
- `POST /ask` — JSON body `{"question": "your question here"}`; response includes `answer`, `sql`, `row_count`, `truncated`.

OpenAPI docs: `http://127.0.0.1:8000/docs`

If `http://127.0.0.1:8000/` returns `{"detail":"Not Found"}`, you’re on an old server process or the wrong app. Stop uvicorn, then start it again from the repo root with `uvicorn claims_nl_qa.main:app --reload` so `GET /` is registered.

### Evaluation (golden set)

Deterministic checks live in [`tests/golden.yaml`](tests/golden.yaml): each case is SQL + an expected aggregate from the same CSV the API uses. No API key required.

```bash
pytest tests/test_golden.py -v
```

That only validates **SQL → numbers** against the file. The LLM can still write different-but-correct SQL or paraphrase answers; this is the honest baseline for “are we grounded in the data?”
