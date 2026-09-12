"""Central configuration, loaded once from the environment (.env file).

Every value has a safe local-dev default so the app can boot without a .env,
except GEMINI_API_KEY which is required for any real work.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Gemini -----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.8-flash"

# gemini-embedding-001 returns 3072 dims by default; we truncate (Matryoshka)
# to this many. MUST stay in sync with vector(...) in schema.sql -- changing it
# also requires re-initializing Postgres (drop the pgvector_data volume).
EMBEDDING_DIMENSIONS = 768

# --- Postgres --------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_docs")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_pass")

# --- Ingest / retrieval --------------------------------------------------------
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# Number of nearest chunks to retrieve per question.
TOP_K = int(os.getenv("TOP_K", "4"))

# Below this cosine similarity, the best retrieved chunk is treated as
# irrelevant: we skip the generation call and return the refusal directly
# instead of asking Gemini to answer from a weak match. Tune against your
# corpus/embedding model -- 0.5 is a conservative starting point.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
