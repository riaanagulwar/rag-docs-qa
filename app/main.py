from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config, db
from app.embeddings import embed_text, generate_answer

app = FastAPI(title="RAG Docs Q&A", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class ChunkResult(BaseModel):
    source_file: str
    chunk_index: int
    similarity: float
    chunk_text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = req.top_k or config.TOP_K

    query_embedding = embed_text(req.question, task_type="retrieval_query")
    chunks = db.similarity_search(query_embedding, top_k)

    if not chunks:
        return AskResponse(
            answer="I couldn't find this in the provided documents.",
            sources=[],
        )

    answer = generate_answer(req.question, chunks)

    return AskResponse(
        answer=answer,
        sources=[
            ChunkResult(
                source_file=c["source_file"],
                chunk_index=c["chunk_index"],
                similarity=round(c["similarity"], 4),
                chunk_text=c["chunk_text"][:300] + ("..." if len(c["chunk_text"]) > 300 else ""),
            )
            for c in chunks
        ],
    )
