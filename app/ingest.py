"""
Run this script to (re)ingest all documents in the docs folder.

Usage:
    python -m app.ingest
"""
import os
import sys
import time

from app import config, db
from app.chunker import chunk_text
from app.embeddings import embed_text


def load_docs(folder: str):
    docs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".md", ".txt")):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
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

        rows = []
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk, task_type="retrieval_document")
            rows.append((fname, i, chunk, embedding))
            time.sleep(0.05)  # gentle on free-tier rate limits

        db.insert_chunks(rows)
        total_chunks += len(chunks)

    print(f"Done. Ingested {total_chunks} chunks from {len(docs)} document(s).")


if __name__ == "__main__":
    main()
