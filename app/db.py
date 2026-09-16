"""All Postgres / pgvector access.

pgvector's `vector` type has no Python driver support, so embeddings are passed
as bracketed literal strings ("[0.1,0.2,...]") and cast with ::vector in SQL.
"""
from contextlib import contextmanager

import pg8000.dbapi as pg8000

from app import config


# Format a float sequence as a pgvector literal: [0.1,0.2,0.3]
def _to_pgvector_literal(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


# Yield a connection and always close it. Commits are the caller's job.
@contextmanager
def _connect():
    conn = pg8000.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    try:
        yield conn
    finally:
        conn.close()


# Bulk-insert chunk rows: (source_file, chunk_index, chunk_text, embedding_list).
# ANALYZE afterwards so the ivfflat index planner has fresh stats after a load.
def insert_chunks(rows):
    serialized = [
        (source_file, chunk_index, chunk_text, _to_pgvector_literal(embedding))
        for source_file, chunk_index, chunk_text, embedding in rows
    ]

    with _connect() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO doc_chunks (source_file, chunk_index, chunk_text, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            serialized,
        )
        cur.execute("ANALYZE doc_chunks;")
        conn.commit()
        cur.close()


# Return the top_k chunks nearest to query_embedding by cosine distance (<=>).
# Distance is converted to a 0..1 similarity score for display. This always
# returns up to top_k rows if the table is non-empty, however weak the match --
# use has_relevant_chunk to decide whether the best one is worth acting on.
def similarity_search(query_embedding, top_k):
    literal = _to_pgvector_literal(query_embedding)

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_file, chunk_index, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (literal, literal, top_k),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "source_file": r[0],
            "chunk_index": r[1],
            "chunk_text": r[2],
            "similarity": float(r[3]),
        }
        for r in rows
    ]


# Whether the best-matching chunk from similarity_search is worth generating
# an answer from, per config.SIMILARITY_THRESHOLD. Callers should skip the
# (costly) generation call and return the refusal directly when this is
# False, rather than always asking Gemini to answer from a weak match.
def has_relevant_chunk(chunks):
    return bool(chunks) and chunks[0]["similarity"] >= config.SIMILARITY_THRESHOLD


# Full-text (keyword) search over chunk_tsv, ranked by ts_rank. Same return
# shape as similarity_search, except "similarity" here holds the ts_rank
# score, not cosine similarity -- the two aren't on the same scale, so don't
# compare them directly (that's what hybrid_search's rank fusion is for).
# Catches exact terms (names, numbers, acronyms) that embedding similarity
# can miss.
def keyword_search(query_text, top_k):
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_file, chunk_index, chunk_text,
                   ts_rank(chunk_tsv, plainto_tsquery('english', %s)) AS rank
            FROM doc_chunks
            WHERE chunk_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
            """,
            (query_text, query_text, top_k),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "source_file": r[0],
            "chunk_index": r[1],
            "chunk_text": r[2],
            "similarity": float(r[3]),
        }
        for r in rows
    ]


# Merge multiple ranked result lists into one via Reciprocal Rank Fusion:
# each item scores sum(1 / (k + rank)) across every list it appears in
# (rank is 1-based). This avoids normalizing cosine similarity against
# ts_rank, which are on unrelated scales -- only rank position matters.
def _reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    items = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = (item["source_file"], item["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, item)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [{**items[key], "rrf_score": scores[key]} for key in ranked_keys]


# Hybrid search: fuse vector similarity and keyword search results via RRF.
# Each side is fetched wider (fetch_k) than what's ultimately returned, so
# fusion has enough candidates from both to rank fairly.
def hybrid_search(query_text, query_embedding, top_k, fetch_k=20):
    vector_results = similarity_search(query_embedding, fetch_k)
    keyword_results = keyword_search(query_text, fetch_k)
    return _reciprocal_rank_fusion([vector_results, keyword_results])[:top_k]


# Wipe every chunk and reset the id sequence. Called at the start of each
# full re-ingest.
def clear_all_chunks():
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE doc_chunks RESTART IDENTITY;")
        conn.commit()
        cur.close()


# Row count of doc_chunks. Lets callers (e.g. the eval harness) fail fast with
# a clear message instead of a confusing empty-results run.
def count_chunks():
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doc_chunks;")
        count = cur.fetchone()[0]
        cur.close()
    return count


# All chunks belonging to one source file, in order. Used by the eval harness
# to work out which chunks are actually relevant to a question (for recall@k)
# without having to hardcode chunk indices in the question set.
def get_chunks_by_source(source_file):
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_index, chunk_text FROM doc_chunks "
            "WHERE source_file = %s ORDER BY chunk_index;",
            (source_file,),
        )
        rows = cur.fetchall()
        cur.close()
    return [{"chunk_index": r[0], "chunk_text": r[1]} for r in rows]
