"""(Re)ingest every document in the docs folder into Postgres.

    python -m app.ingest

Each run wipes the doc_chunks table and rebuilds it from scratch -- there is no
incremental update. A failure partway through (e.g. a Gemini error) leaves the
corpus partially populated.
"""
import os
import sys
import time

from app import config, db
from app.chunker import chunk_text
from app.embeddings import TASK_DOCUMENT, embed_texts


# Read every .md / .txt file in `folder` as (filename, text).
def load_docs(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".md", ".txt")):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                docs.append((fname, f.read()))
    return docs


def main():
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    docs = load_docs(config.DOCS_FOLDER)
    if not docs:
        print(f"No .md or .txt files found in {config.DOCS_FOLDER}")
        sys.exit(1)

    print(f"Found {len(docs)} document(s). Clearing existing chunks...")
    db.clear_all_chunks()

    total_chunks = 0
    for fname, text in docs:
        chunks = chunk_text(text)
        print(f"  {fname}: {len(chunks)} chunks")

        # One embedding call per file instead of one per chunk.
        embeddings = embed_texts(chunks, task_type=TASK_DOCUMENT)
        rows = [(fname, i, chunk, emb) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]

        db.insert_chunks(rows)
        total_chunks += len(chunks)
        time.sleep(0.05)  # stay under free-tier rate limits between files

    print(f"Done. Ingested {total_chunks} chunks from {len(docs)} document(s).")


if __name__ == "__main__":
    main()
