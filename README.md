## claims-nl-qa

This project is a small service that lets you ask questions in plain English about a **synthetic healthcare claims dataset**. Under the hood, it translates your question into SQL, runs it, and then gives you back both the answer *and* the query it used.

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

One small gotcha: run everything from the repo root, otherwise `.env` might not get picked up.

---

## How it actually works (high level)

Very roughly:

1. Load the CSV into memory
2. Ask the LLM to generate SQL
3. Sanity-check that SQL
4. Run it in DuckDB
5. Ask the LLM again to explain the result

That’s it.

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
  * block obvious dangerous keywords

* Results are capped (row limit) so nothing explodes

* Then a second LLM call takes a **preview of actual query output** and turns it into a readable answer
  → this helps avoid hallucinating based only on schema

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

* answer
* sql
* row_count
* truncated flag

Docs:

```
http://127.0.0.1:8000/docs
```

If you hit `/` and get `"Not Found"`, it’s almost always because uvicorn wasn’t started from the repo root.

---

## Why SQL instead of RAG?

Short answer: this is structured data.

Longer answer:

* Most questions here are aggregations or filters
* Vector search over row text would be awkward and hard to verify
* SQL is deterministic → easier to test

So yeah… SQL just made more sense here.

---

## Why DuckDB?

Honestly:

* super lightweight
* great for analytics queries
* no setup needed
* works perfectly in-memory for this scale

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

---

## What the tests cover

* data loading + schema sanity
* SQL validation logic
* API behavior
* “golden” queries (predefined SQL → expected outputs)

The golden tests are useful because they don’t depend on the LLM at all.

---

## Evaluation approach

There’s a `tests/golden.yaml` file with known queries and expected results.

So instead of asking:

> “did the model explain it well?”

we check:

> “did the system produce the correct numbers?”

Different problem, but much more stable.

---

## Known limitations (important)

This is still a demo, so a few obvious gaps:

* LLM can still generate incorrect SQL (we only catch obvious bad cases)
* No real security model (no auth, no rate limiting, etc.)
* In-memory DB + single connection → not scalable
* Golden tests don’t evaluate answer quality, only correctness of results
* Dataset is synthetic → real-world messiness isn’t represented

So yeah… not production-ready, especially not for anything involving real PHI.

---

## Project structure (quick glance)

```
src/claims_nl_qa/
  config.py
  data.py
  qa.py
  main.py

docs/
  synthetic_claims.csv

tests/
  golden.yaml
  test_*.py
```