## claims-nl-qa

This project is a lightweight service that allows an internal analyst to ask questions in plain English about a **synthetic healthcare claims dataset** and receive clear, evidence-backed answers.
Behind the scenes, the core flow is still **NL → SQL → DuckDB → explanation of results**, which works well for structured claims data. In addition, the system creates **healthcare-focused “document” representations** for each claim and runs a simple **retrieval and chunking process** to support **source-style citations**.
It also includes a few practical safeguards—if the supporting data is weak, or if a question starts to drift into clinical advice, the system flags that and responds more cautiously instead of overcommitting.

It’s meant more as a **grounded Q&A demo** than a production system.

---

## Quick setup

You’ll need **Python 3.10+**.

From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

If your default Python is outdated (common on Windows), just do:

```bash
py -3.12 -m venv .venv
```

Then copy `.env.example` → `.env` and set your `OPENAI_API_KEY`.
(yeah… don’t commit that file)

---

## Config (what actually matters)

| Variable         | Required         | Notes                                   |
| ---------------- | ---------------- | --------------------------------------- |
| `OPENAI_API_KEY` | Yes (for `/ask`) | Used for SQL + answer generation        |
| `OPENAI_MODEL`   | No               | Defaults to `gpt-4o-mini`               |
| `DATA_PATH`      | No               | Defaults to `docs/synthetic_claims.csv` |
| `RUN_OPENAI_SMOKE` | No             | Set to `1` to enable real-API smoke tests |

One small gotcha: run everything from the repo root, otherwise `.env` might not get picked up.

---

## How it actually works (high level)

Very roughly:


**Ground truth for numbers** still comes from SQL results. Retrieval is there for **traceability and reviewer UX**, not as a substitute for aggregates on tabular claims.
1. Load the CSV into memory and register it as a `claims` table in DuckDB
2. Create a parallel `healthcare_docs` table—basically one synthetic “document” per claim, with metadata to support retrieval
3. Break those documents into chunks using sentence-aware logic so they feel closer to policy or section-style snippets
4. When a question comes in, pull back a handful of the most relevant chunks (mainly for debugging and **citations** in the API response)
5. Have the LLM generate SQL, but only against the `claims` table
6. Do a quick sanity check, then run that SQL in DuckDB
7. Call the LLM again to explain the **actual query results**—making sure the explanation reflects the returned rows, not just the schema
8. Add **safety guardrails**—for example, escalate if the wording looks high-risk, or return an “insufficient evidence” message if the grounding isn’t strong

---

## Slightly more detail

* The dataset is loaded with pandas and registered into an **in-memory DuckDB** table called `claims`

* The LLM is prompted with schema + question and is forced to return JSON like:

  ```json
  { "sql": "SELECT ..." }
  ```

* Before execution, we do some basic checks:

  * only `SELECT` / `WITH`
  * no multi-statements
  * must reference `claims`
  * block dangerous SQL keywords (including DuckDB admin/metadata statements)

* Results are capped (row limit) so nothing explodes
* SQL execution uses a timeout and best-effort interrupt to avoid hanging requests
* OpenAI client calls use explicit timeouts
* SQL execution errors returned to clients are sanitized (full details stay in server logs)

* Then a second LLM call takes a **preview of actual query output** and turns it into a readable answer
  → this helps avoid hallucinating based only on schema

* Citations are built from the **top retrieved chunks** (`chunk_id`, `document_id`, `payer`, `effective_date`) so reviewers can see what narrative context was surfaced alongside the SQL path

---

## API layer

FastAPI app with a few endpoints:

* `GET /` → basic info
* `GET /health` → quick check + row count
* `POST /ask` → main endpoint

Example request:

```json
{
  "question": "What is the average claim amount by provider?"
}
```

Response includes:

* `answer` — a plain-English explanation, finalized after applying safety checks
* `sql` — the validated `SELECT` or `WITH` query that runs against the `claims` table
* `row_count` — the number of rows returned, after applying the internal cap
* `truncated` — indicates whether that row limit was reached
* `citations` — a short list of source snippets pulled from the top-ranked retrieval chunks (used for context and traceability, not as the source of numeric truth)

Docs:

```
http://127.0.0.1:8000/docs
```

If you hit `/` and get `"Not Found"`, it’s almost always because uvicorn wasn’t started from the repo root.

---

## Design: SQL, RAG-style retrieval, and hybrid tradeoffs

**SQL** is really the backbone here. The data is relational, and most analyst questions boil down to filters and aggregates anyway, so SQL gives you consistent, reproducible numbers and makes testing much more straightforward.

Layering in **retrieval and chunking** over the derived claim “documents” adds a more healthcare-friendly context. You get useful metadata—like payer or effective date—along with ranked snippets and **citations** in the API response. That said, it’s not meant to replace SQL when it comes to computing totals or averages in this setup.

If the scope included external materials, like policy PDFs, then a **pure vector-based RAG** approach would make more sense. But since we’re only working with `synthetic_claims.csv`, using lightweight lexical retrieval over generated snippets is a practical, intentional middle ground.


---

## Why DuckDB?

Honestly:

* super lightweight
* great for analytics queries
* no setup needed
* works perfectly in-memory for this scale
* external access is disabled after claims loading, so arbitrary file/network table functions are blocked

---

## Why two LLM calls?

We tried doing it in one step initially… didn’t work great.

Splitting it helps:

1. First call → focus on getting SQL right
2. Second call → focus on explaining actual results

Keeps things cleaner and more reliable.

---

## Tests

Run everything:

```bash
pytest -q
```

Skip anything that hits OpenAI:

```bash
pytest -q -k "not smoke"
```

Run only real-API smoke tests (opt-in):

```bash
# PowerShell
$env:RUN_OPENAI_SMOKE="1"
pytest -q -k "smoke"
```

Notes:

* Smoke tests are intentionally opt-in because they need a working OpenAI API connection.
* If your network/region is not supported by the API, smoke tests will run but fail with a 403 from OpenAI.

---

## What the tests cover

* data loading + schema sanity (including `healthcare_docs` registration)
* SQL validation logic
* API behavior
* chunking + retrieval ranking and filters
* small **eval harness** (`eval.summarize_retrieval`) for retrieval-focused checks
* “golden” queries (predefined SQL → expected outputs)

The golden and retrieval tests are useful because they don’t depend on the LLM.

---

## Evaluation approach


1. **Golden SQL suite** — In `tests/golden.yaml`, we run a set of canonical `SELECT` queries against the same CSV the application uses, and assert on key values like counts, sums, and averages. This serves as the **source of truth** for aggregate correctness.

2. **Retrieval evaluation** — In `tests/test_eval.py`, we validate retrieval behavior using `claims_nl_qa.eval.summarize_retrieval`, along with metadata-based filtering. The goal is to keep improvements to chunking and ranking under control, without needing to call OpenAI during tests.

3. **Smoke tests (optional)** — When `RUN_OPENAI_SMOKE=1` is set and a valid API key is available, a small set of tests runs end-to-end against the real API to confirm everything is wired up correctly.

It’s naturally harder to evaluate LLM-generated explanations in an automated way. To reduce that risk, the system grounds responses in the **actual query results**, and always returns the **`sql` along with deterministic checks**, so the numbers can be verified independently.


---

## Known limitations (important)

This is still a demo, so a few obvious gaps:

* LLM can still generate incorrect SQL (guardrails reduce risk but cannot guarantee perfect SQL)
* No full production security model (no auth, no rate limiting, etc.)
* In-memory DB + single connection → not scalable
* The golden tests don’t try to evaluate how the final `answer` is phrased—they focus strictly on whether the **underlying SQL results** match the expected fixtures when run directly
* The citations come from retrieved snippets that are **derived from the claims data itself**, not from external clinical policies or independent sources
* The retrieval layer is intentionally simple and lexical-based; we chose to skip embeddings and rerankers in favor of keeping the system lightweight, with no additional infrastructure and fully offline testability
* Dataset is synthetic → real-world messiness isn’t represented

So yeah… not production-ready, especially not for anything involving real PHI.

---

## Project structure (quick glance)

```
src/claims_nl_qa/
  config.py
  data.py
  eval.py
  qa.py
  retrieval.py
  main.py

docs/
  synthetic_claims.csv

tests/
  golden.yaml
  test_*.py
```