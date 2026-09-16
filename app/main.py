"""FastAPI app: POST /ask embeds the question, retrieves nearby chunks via
hybrid (vector + keyword) search, and has Gemini answer from them. GET
/health checks Postgres connectivity for liveness/readiness probes.

Every /ask request logs its full flow at INFO level -- question, retrieved
chunks (with score), the relevance-gate decision, the answer, and which
chunks it cited -- so you can see exactly what happened for a given request
instead of just the JSON response. Configured here (not left to uvicorn's
own logging) so it shows up whether you run via `uvicorn app.main:app` or
import the app some other way.
"""
import logging

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors
from pydantic import BaseModel

from app import config, db
from app.embeddings import TASK_QUERY, embed_text, generate_answer, parse_citations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="RAG Docs Q&A", version="0.1.0")
log = logging.getLogger("app.main")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None  # override config.TOP_K per request


class ChunkResult(BaseModel):
    source_file: str
    chunk_index: int
    similarity: float
    chunk_text: str
    cited: bool  # whether the answer actually cited this chunk (see parse_citations)


class AskResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]


# Length of the chunk preview echoed back in `sources` (full text is not returned).
_PREVIEW_CHARS = 300
_REFUSAL = "I couldn't find this in the provided documents."


def _format_chunk(c):
    # hybrid_search results carry rrf_score -- that's what actually decided
    # their order, so log it instead of the leftover cosine similarity from
    # whichever source list the chunk was first seen in.
    if "rrf_score" in c:
        return f"{c['source_file']}#{c['chunk_index']} (rrf={c['rrf_score']:.4f})"
    return f"{c['source_file']}#{c['chunk_index']} (sim={c['similarity']:.3f})"


# Checks Postgres connectivity (not just that the process is alive) so this
# is useful as a readiness probe, not just a liveness one.
@app.get("/health")
def health():
    try:
        chunk_count = db.count_chunks()
    except Exception as e:
        log.warning("health check: database unreachable: %s", e)
        raise HTTPException(status_code=503, detail="database unreachable") from e
    return {"status": "ok", "chunks": chunk_count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = req.top_k or config.TOP_K
    log.info("[ask] question: %r (top_k=%d)", question[:200], top_k)

    try:
        query_embedding = embed_text(question, task_type=TASK_QUERY)
        chunks = db.hybrid_search(question, query_embedding, top_k)
    except genai_errors.APIError as e:
        log.error("[ask] embedding call failed: %s", e)
        raise HTTPException(status_code=502, detail="embedding service unavailable, try again") from e
    except Exception as e:
        log.error("[ask] retrieval failed: %s", e)
        raise HTTPException(status_code=503, detail="database unavailable, try again") from e

    log.info(
        "[ask] retrieved %d chunk(s): %s",
        len(chunks),
        ", ".join(_format_chunk(c) for c in chunks) or "none",
    )

    # Skip generation entirely -- and the API call it costs -- when nothing
    # retrieved is a close enough match to be worth answering from.
    if not db.has_relevant_chunk(chunks):
        log.info(
            "[ask] best chunk below SIMILARITY_THRESHOLD=%s, refusing without calling generate_answer",
            config.SIMILARITY_THRESHOLD,
        )
        return AskResponse(answer=_REFUSAL, sources=[])

    try:
        answer = generate_answer(question, chunks)
    except genai_errors.APIError as e:
        log.error("[ask] generation call failed: %s", e)
        raise HTTPException(status_code=502, detail="generation service unavailable, try again") from e

    log.info("[ask] answer: %s", answer if len(answer) <= 200 else answer[:200] + "...")

    cited_indices = parse_citations(answer, chunks)
    log.info(
        "[ask] cited: %s",
        ", ".join(_format_chunk(chunks[i]) for i in sorted(cited_indices)) or "none",
    )

    return AskResponse(
        answer=answer,
        sources=[
            ChunkResult(
                source_file=c["source_file"],
                chunk_index=c["chunk_index"],
                similarity=round(c["similarity"], 4),
                chunk_text=_preview(c["chunk_text"]),
                cited=(i in cited_indices),
            )
            for i, c in enumerate(chunks)
        ],
    )


def _preview(text):
    return text[:_PREVIEW_CHARS] + ("..." if len(text) > _PREVIEW_CHARS else "")
