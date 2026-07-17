import pg8000.dbapi as pg8000

from app import config


def _to_pgvector_literal(embedding):
    """
    pgvector's `vector` type expects bracketed literal syntax, e.g. "[0.1,0.2,0.3]".
    We build this string ourselves and cast it with ::vector in SQL, since no
    Python driver knows about pgvector's custom type out of the box.
    """
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def get_connection():
    return pg8000.connect(
        host=config.DB_HOST,
        port=int(config.DB_PORT),
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def insert_chunks(rows):
    """
    rows: list of tuples (source_file, chunk_index, chunk_text, embedding_list)
    """
    serialized_rows = [
        (source_file, chunk_index, chunk_text, _to_pgvector_literal(embedding))
        for source_file, chunk_index, chunk_text, embedding in rows
    ]

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO doc_chunks (source_file, chunk_index, chunk_text, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            serialized_rows,
        )
        conn.commit()

        # Rebuild ANALYZE so ivfflat index stats are fresh after a bulk load
        cur.execute("ANALYZE doc_chunks;")
        conn.commit()
        cur.close()
    finally:
        conn.close()


def similarity_search(query_embedding, top_k):
    """
    Returns top_k most similar chunks using cosine distance (<=> operator).
    Lower distance = more similar. We convert to a similarity score for display.
    """
    embedding_literal = _to_pgvector_literal(query_embedding)

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_file, chunk_index, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (embedding_literal, embedding_literal, top_k),
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
    finally:
        conn.close()


def clear_all_chunks():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE doc_chunks RESTART IDENTITY;")
        conn.commit()
        cur.close()
    finally:
        conn.close()
