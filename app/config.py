import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_docs")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_pass")

DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

TOP_K = int(os.getenv("TOP_K", "4"))

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-flash-latest"
EMBEDDING_DIMENSIONS = 768  # must match vector(768) in schema.sql
