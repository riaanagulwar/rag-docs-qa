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
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Index for fast approximate nearest-neighbor search
-- ivfflat needs ANALYZE after data is loaded to build properly
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Keyword search (db.keyword_search, db.hybrid_search) runs via BM25 in the
-- app itself (see app/db.py), not Postgres full-text search -- no schema
-- support needed for it here.
