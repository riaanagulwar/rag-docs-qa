"""All Postgres / pgvector access.

pgvector's `vector` type has no Python driver support, so embeddings are passed
as bracketed literal strings ("[0.1,0.2,...]") and cast with ::vector in SQL.
"""
import logging
import re
from contextlib import contextmanager

import pg8000.dbapi as pg8000
from rank_bm25 import BM25Okapi

from app import config

log = logging.getLogger("app.db")


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


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# Rank `chunks` (each a dict with a "chunk_text" key) against query_text with
# BM25 -- the same ranking algorithm behind Elasticsearch/Lucene keyword
# search. A chunk matching only some of the query's words still gets a
# (lower) score instead of being excluded outright, so a real question like
# "Which Roman legion did Julius Caesar lead across the Rubicon?" still
# surfaces the chunk that says "Caesar", "Rubicon", and "legion" even though
# it never says "Roman". Pure function (no DB) so it's unit-testable without
# Postgres.
#
# Excluded only by actual token overlap, not by score sign: BM25's score can
# come out zero or negative for a chunk that DOES share a query term, purely
# from IDF math on very common words in a small corpus (a term in most/all
# chunks gets a negative IDF weight) -- filtering on "score > 0" would wrongly
# drop a real partial match. Zero overlap is the only case that's genuinely
# "no match".
def _bm25_rank(chunks, query_text, top_k):
    if not chunks:
        return []
    query_tokens = set(_tokenize(query_text))
    chunk_tokens = [_tokenize(c["chunk_text"]) for c in chunks]
    scores = BM25Okapi(chunk_tokens).get_scores(list(query_tokens))

    candidates = [
        (c, score) for c, score, tokens in zip(chunks, scores, chunk_tokens)
        if query_tokens & set(tokens)
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [{**c, "similarity": float(score)} for c, score in candidates[:top_k]]


# Keyword search via BM25, computed in memory over the whole corpus (cheap at
# this scale; rebuilt fresh each call so it's always current, no cache to
# invalidate on ingest). Same return shape as similarity_search, except
# "similarity" here holds the BM25 score, not cosine similarity -- the two
# aren't on the same scale, so don't compare them directly (that's what
# hybrid_search's rank fusion is for). Catches exact terms (names, numbers,
# acronyms) that embedding similarity can miss.
def keyword_search(query_text, top_k):
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT source_file, chunk_index, chunk_text FROM doc_chunks;")
        rows = cur.fetchall()
        cur.close()

    chunks = [{"source_file": r[0], "chunk_index": r[1], "chunk_text": r[2]} for r in rows]
    return _bm25_rank(chunks, query_text, top_k)


# Merge multiple ranked result lists into one via Reciprocal Rank Fusion:
# each item scores sum(1 / (k + rank)) across every list it appears in
# (rank is 1-based). This avoids normalizing cosine similarity against
# BM25 score, which are on unrelated scales -- only rank position matters.
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


# One-line summary of a result list for logging: how many hits, and the
# identity + score of each (cosine similarity or BM25 score, whichever this list is).
def _format_results(results):
    return ", ".join(f"{r['source_file']}#{r['chunk_index']} ({r['similarity']:.3f})" for r in results) or "none"


# Hybrid search: fuse vector similarity and keyword search results via RRF.
# Each side is fetched wider (fetch_k) than what's ultimately returned, so
# fusion has enough candidates from both to rank fairly. Logs each side's raw
# results before fusion -- the fused/RRF output alone hides which method
# actually found what, which matters when you're trying to tell whether
# keyword search is pulling its weight for a given query.
def hybrid_search(query_text, query_embedding, top_k, fetch_k=20):
    vector_results = similarity_search(query_embedding, fetch_k)
    keyword_results = keyword_search(query_text, fetch_k)
    log.info("hybrid_search: vector: %d hit(s): %s", len(vector_results), _format_results(vector_results))
    log.info("hybrid_search: keyword: %d hit(s): %s", len(keyword_results), _format_results(keyword_results))

    fused = _reciprocal_rank_fusion([vector_results, keyword_results])[:top_k]
    log.info(
        "hybrid_search: fused top-%d: %s",
        top_k,
        ", ".join(f"{r['source_file']}#{r['chunk_index']} (rrf={r['rrf_score']:.4f})" for r in fused) or "none",
    )
    return fused


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
