-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table storing document chunks and their embeddings
-- Gemini's text-embedding-004 model outputs 768-dim vectors
CREATE TABLE IF NOT EXISTS doc_chunks (
    id            SERIAL PRIMARY KEY,
    source_file   TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     vector(768) NOT NULL,
    -- Full-text search vector for keyword/hybrid search (db.keyword_search,
    -- db.hybrid_search). Generated from chunk_text automatically, so it's
    -- always in sync -- nothing in app code writes to it directly.
    chunk_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Index for fast approximate nearest-neighbor search
-- ivfflat needs ANALYZE after data is loaded to build properly
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for full-text keyword search
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_idx
    ON doc_chunks
    USING GIN (chunk_tsv);

-- NOTE: this file only runs automatically against a fresh Postgres volume
-- (see docker-compose.yml). If doc_chunks already exists without chunk_tsv,
-- run `docker compose down -v && docker compose up -d` to recreate it (then
-- re-ingest), rather than expecting this file to alter it in place.
