"""FastAPI app: POST /ask embeds the question, retrieves nearby chunks, and has
Gemini answer from them. GET /health for liveness checks."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config, db
from app.embeddings import TASK_QUERY, embed_text, generate_answer

app = FastAPI(title="RAG Docs Q&A", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None  # override config.TOP_K per request


class ChunkResult(BaseModel):
    source_file: str
    chunk_index: int
    similarity: float
    chunk_text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]


# Length of the chunk preview echoed back in `sources` (full text is not returned).
_PREVIEW_CHARS = 300


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = req.top_k or config.TOP_K

    query_embedding = embed_text(question, task_type=TASK_QUERY)
    chunks = db.similarity_search(query_embedding, top_k)

    # Skip generation entirely -- and the API call it costs -- when nothing
    # retrieved is a close enough match to be worth answering from.
    if not db.has_relevant_chunk(chunks):
        return AskResponse(
            answer="I couldn't find this in the provided documents.",
            sources=[],
        )

    answer = generate_answer(question, chunks)

    return AskResponse(
        answer=answer,
        sources=[
            ChunkResult(
                source_file=c["source_file"],
                chunk_index=c["chunk_index"],
                similarity=round(c["similarity"], 4),
                chunk_text=_preview(c["chunk_text"]),
            )
            for c in chunks
        ],
    )


def _preview(text):
    return text[:_PREVIEW_CHARS] + ("..." if len(text) > _PREVIEW_CHARS else "")
