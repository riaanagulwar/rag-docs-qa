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
chunker.py  --  fixed-size chunks with overlap
   |
   v
embeddings.py  --  Gemini text-embedding-004 (768-dim)
   |
   v
Postgres + pgvector  --  doc_chunks table, ivfflat cosine index

Query flow:
User question
   -> embed_text(question, task_type="retrieval_query")
   -> db.similarity_search()  top-k nearest chunks
   -> generate_answer()  Gemini answers using only retrieved context
   -> answer + source chunks returned to caller
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

Drop `.md` or `.txt` files into the `docs/` folder. See `docs/README.md`
for suggestions on what to use.

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
  -d '{"question": "What is the difference between a mutex and a semaphore?"}'
```

Response:

```json
{
  "answer": "A mutex ... [Source: lld_guide.md]",
  "sources": [
    {
      "source_file": "lld_guide.md",
      "chunk_index": 3,
      "similarity": 0.86,
      "chunk_text": "..."
    }
  ]
}
```



## Next steps (if you want to extend this later)

- Semantic chunking instead of fixed-size splitting
- Hybrid search (keyword + vector) for exact-term queries (error codes,
  function names)
- Reranking retrieved chunks before generation
- Simple web UI instead of curl/Postman
- Streaming responses


