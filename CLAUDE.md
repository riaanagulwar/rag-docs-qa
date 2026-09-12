# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Postgres + pgvector (runs schema.sql on first boot only)
docker compose up -d

# Install deps
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt

# Ingest: chunk + embed every .md/.txt in DOCS_FOLDER, wiping the table first
python3 -m app.ingest

# Run the API (http://localhost:8000, docs at /docs)
uvicorn app.main:app --reload

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "..."}'
```

Requires `GEMINI_API_KEY` in `.env` (copy from `.env.example`). There is no
test suite, linter, or build step.

If the embedding vector dimension changes, `EMBEDDING_DIMENSIONS` in
`app/config.py` and `vector(768)` in `schema.sql` must be updated together, and
Postgres must be re-initialized (drop the `pgvector_data` volume) since
`schema.sql` only runs on first container boot.

## Architecture

Retrieval-Augmented Generation over local docs. Two flows share the same
`doc_chunks` table (Postgres + pgvector):

**Ingest** (`app/ingest.py`): load docs → `chunker.chunk_text` → `embeddings.embed_text`
per chunk → `db.insert_chunks`. `TRUNCATE`s the whole table on every run; there
is no incremental re-ingest.

**Query** (`app/main.py` `/ask`): `embed_text(question)` → `db.similarity_search`
(cosine `<=>`, top-k) → `embeddings.generate_answer` (Gemini answers from
retrieved chunks only) → answer + source chunks.

Key cross-file details:

- **pgvector has no Python type.** `db._to_pgvector_literal` serializes float
  lists to `"[...]"` strings; all SQL casts them with `::vector`.
- **Embedding task_type matters.** Chunks use `embeddings.TASK_DOCUMENT`,
  questions use `embeddings.TASK_QUERY` — asymmetric embeddings, don't unify them.
- **Grounding is prompt-only.** `similarity_search` has no distance threshold, so
  it almost always returns `top_k` rows regardless of relevance; the "I couldn't
  find this" behavior comes from the `generate_answer` prompt, not from empty
  retrieval.
- **Chunking is header-aware** (`chunker.py`): split on markdown headers first,
  then paragraph/sentence boundaries for oversized sections, re-attaching the
  header as a `## ` prefix to each sub-chunk. `CHUNK_OVERLAP_CHARS` of trailing
  text is carried between consecutive sub-chunks of the same section (not
  between separate header sections).
- Config models are hardcoded in `app/config.py` (`gemini-embedding-001`,
  `gemini-flash-latest`); everything else comes from env vars with defaults.
- `ivfflat` index (`lists = 100`) is created in `schema.sql` before any data
  exists; `insert_chunks` runs `ANALYZE` after bulk loads.
