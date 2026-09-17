# RAG-based Internal Docs Q&A

A small Retrieval-Augmented Generation service: ask questions over your own
markdown/text docs and get citation-grounded answers, powered by
Postgres+pgvector for semantic search and Gemini for embeddings/generation.

## Why this architecture

- **pgvector instead of a dedicated vector DB** — if you already run
  Postgres, there's no new infra to operate, back up, or monitor. For a
  corpus of a few thousand chunks, `ivfflat` cosine similarity is pretty
  fast; a dedicated vector DB only starts to matter at a scale this
  project isn't targeting.
- **Grounded generation, not free-text generation** — the prompt explicitly
  instructs the model to say "I couldn't find this in the provided
  documents" when retrieval comes back empty or irrelevant, instead of
  letting it hallucinate an answer. This is arguably the most important
  part of a RAG system and the part most tutorials skip.


## Architecture

```
Docs (markdown/text files)
   |
   v
chunker.py  --  header-aware chunks (markdown "#" or plain-text headings),
                 recursive splitting + overlap for oversized sections
   |
   v
embeddings.py  --  Gemini gemini-embedding-001, truncated to 768-dim
   |
   v
Postgres + pgvector  --  doc_chunks table, ivfflat cosine index

Query flow:
User question
   -> embed_text(question, task_type="RETRIEVAL_QUERY")
   -> db.hybrid_search()  pgvector cosine search + in-process BM25 keyword
                           search (rank_bm25), fused via Reciprocal Rank Fusion
   -> generate_answer()  Gemini answers using only retrieved context, citing
                          sources by number ([1], [2]) -- skipped entirely if
                          nothing retrieved clears SIMILARITY_THRESHOLD
   -> answer + source chunks (flagged `cited: true/false`) returned to caller
```

## Setup

### 1. Start Postgres with pgvector

```bash
docker compose up -d
```

This starts Postgres on `localhost:5432` and runs `schema.sql`
automatically on first boot (creates the `vector` extension and the
`doc_chunks` table).

### 2. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` (free key at
https://aistudio.google.com/apikey).

### 4. Add documents

Drop `.md` or `.txt` files into the `docs/` folder. A small sample corpus
(`docs/test-doc*.md`) is already there so the app works out of the box;
swap in your own docs whenever you're ready (and update
`eval/questions.json` to match if you want the eval harness to keep working).

### 5. Ingest documents

```bash
python3 -m app.ingest
```

This chunks each file, embeds every chunk via Gemini, and stores the
embeddings in Postgres.

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

### 7. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What year did the Space Race begin with the launch of Sputnik?"}'
```

Response:

```json
{
  "answer": "The Space Race began in 1957, when the Soviet Union launched Sputnik [1].",
  "sources": [
    {
      "source_file": "test-doc2.md",
      "chunk_index": 1,
      "similarity": 0.81,
      "chunk_text": "The Space Race was a Cold War-era competition...",
      "cited": true
    }
  ]
}
```



## Testing and evaluation

```bash
pip3 install -r requirements-dev.txt   # adds pytest
pytest                                  # unit tests, no live DB/Gemini needed

python -m eval.run_eval                 # retrieval/recall/faithfulness/
python -m eval.run_eval --retrieval compare  # citation/anti-hallucination
```

`pytest` covers pure logic (chunking, rank fusion, citation parsing, retry
behavior, request handlers with mocked dependencies) -- see `CLAUDE.md` for
what it does and doesn't cover. `eval/run_eval.py` is what actually exercises
retrieval and generation against real Postgres + Gemini; results are appended
to `eval/results.md` as a running log.

## Next steps (if you want to extend this later)

- Semantic chunking instead of size-based splitting for oversized sections
- Reranking retrieved chunks before generation
- Incremental re-ingest (currently every `app.ingest` run wipes and rebuilds
  the whole corpus)
- Simple web UI instead of curl/Postman
- Streaming responses


