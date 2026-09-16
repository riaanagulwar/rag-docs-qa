# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Postgres + pgvector (runs schema.sql on first boot only)
docker compose up -d

# Install deps (add -r requirements-dev.txt too if running tests)
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt -r requirements-dev.txt

# Ingest: chunk + embed every .md/.txt in DOCS_FOLDER, wiping the table first
python3 -m app.ingest

# Run the API (http://localhost:8000, docs at /docs)
uvicorn app.main:app --reload

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "..."}'

# Run the eval harness (retrieval/recall/faithfulness/citation/anti-hallucination
# metrics against eval/questions.json; --retrieval compare shows vector vs hybrid)
python -m eval.run_eval
python -m eval.run_eval --retrieval compare --limit 5

# Unit tests (pure logic only -- no live Postgres/Gemini required; see below)
pytest
```

Requires `GEMINI_API_KEY` in `.env` (copy from `.env.example`).

If the embedding vector dimension changes, `EMBEDDING_DIMENSIONS` in
`app/config.py` and `vector(768)` in `schema.sql` must be updated together, and
Postgres must be re-initialized (`docker compose down -v && docker compose up -d`)
since `schema.sql` only runs on first container boot -- this also applies to the
`chunk_tsv` full-text column hybrid search needs.

## Architecture

Retrieval-Augmented Generation over local docs. Ingest, the API, and the eval
harness all share the same `doc_chunks` table (Postgres + pgvector):

**Ingest** (`app/ingest.py`): load docs → `chunker.chunk_text` → one batched
`embeddings.embed_texts` call per file → `db.insert_chunks`. `TRUNCATE`s the
whole table on every run; there is no incremental re-ingest.

**Query** (`app/main.py` `/ask`): `embed_text(question)` → `db.hybrid_search`
(vector + keyword, fused via Reciprocal Rank Fusion, top-k) →
`embeddings.generate_answer` (Gemini answers from retrieved chunks only, citing
them by number) → answer + source chunks (each flagged `cited: bool`).
Generation is skipped entirely (cost + latency saving, and the real
anti-hallucination gate) when nothing retrieved clears
`config.SIMILARITY_THRESHOLD` -- see `db.has_relevant_chunk`. `/health` checks
Postgres connectivity, not just process liveness. Gemini/DB failures return
502/503 with a clean message rather than leaking a raw traceback.

**Eval** (`eval/run_eval.py`, questions in `eval/questions.json`): scores each
question on retrieval hit-rate, `recall@k`, keyword-based
correctness/faithfulness, citation validity, and anti-hallucination (does an
out-of-scope question correctly refuse?). `--retrieval {vector,hybrid,compare}`
picks which retrieval path to score; `compare` runs both per question at zero
extra Gemini cost (keyword search is Postgres-only) so you can see whether
hybrid search actually helps. Results are appended (not overwritten) to
`eval/results.md` as a dated log, so past runs stay comparable.

**Tests** (`tests/`, run with `pytest`): unit tests for the pure logic only --
chunker splitting/overlap, RRF fusion, citation parsing, retry-on-transient-error,
eval scoring helpers, `main.py`'s request handlers called directly with
monkeypatched `db`/Gemini calls. Nothing here touches a real Postgres or Gemini
API (`google.genai.Client` is patched out in `tests/conftest.py`) -- functions
that need a live DB (`similarity_search`, `keyword_search`, `insert_chunks`,
...) aren't unit-tested; `python -m eval.run_eval` is what exercises those for
real.

Key cross-file details:

- **pgvector has no Python type.** `db._to_pgvector_literal` serializes float
  lists to `"[...]"` strings; all SQL casts them with `::vector`.
- **Embedding task_type matters.** Chunks use `embeddings.TASK_DOCUMENT`,
  questions use `embeddings.TASK_QUERY` — asymmetric embeddings, don't unify them.
- **Hybrid search = RRF, not score blending.** `db.hybrid_search` fuses
  `similarity_search` (cosine) and `keyword_search` (Postgres full-text,
  `ts_rank`) by rank position (`_reciprocal_rank_fusion`), since cosine
  similarity and `ts_rank` aren't on comparable scales.
- **Citations are numbered, not filenames.** `embeddings._build_context_block`
  numbers chunks `[1]`, `[2]`... in the prompt; `parse_citations` maps a
  model's `[N]` back to a chunk index, silently dropping out-of-range numbers
  (a citation to nothing is itself a hallucination, not a bug to crash on).
- **Chunking recognizes headers beyond markdown `#`.** `chunker.py` also
  detects plain-text headings via a heuristic (short, isolated between blank
  lines, title-cased) so headerless `.txt`-style docs still get topic-aware
  chunking, not just markdown. Falls back to recursive character splitting
  (paragraph → line → sentence → word) for oversized sections, with
  `CHUNK_OVERLAP_CHARS` of shared trailing text between consecutive sub-chunks.
- Gemini calls (`embeddings._with_retry`) retry transient 503/429 errors with
  backoff; other errors (bad request, auth, safety block) fail immediately.
- Config models are hardcoded in `app/config.py`; everything else (chunk size,
  overlap, top_k, similarity threshold) comes from env vars with defaults.
- `ivfflat` index (`lists = 100`) and the `chunk_tsv` GIN index are both
  created in `schema.sql` before any data exists; `insert_chunks` runs
  `ANALYZE` after bulk loads.
