from google import genai
from google.genai import types

from app import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT"):
    """
    task_type differs for docs vs queries per Gemini's embedding API:
    - "RETRIEVAL_DOCUMENT" when embedding chunks to store
    - "RETRIEVAL_QUERY" when embedding a user's question

    gemini-embedding-001 outputs 3072 dimensions by default, but supports
    truncating via Matryoshka Representation Learning -- we request 768
    to match our pgvector schema (doc_chunks.embedding vector(768)) and
    keep storage/search cheap.
    """
    result = _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )
    return result.embeddings[0].values


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Builds a grounded prompt from retrieved chunks and calls Gemini.
    Explicitly instructs the model to say when it doesn't know,
    rather than hallucinating -- this is the key RAG safety behavior.
    """
    context_block = "\n\n".join(
        f"[Source: {c['source_file']} chunk {c['chunk_index']}]\n{c['chunk_text']}"
        for c in context_chunks
    )

    prompt = f"""You are a documentation assistant. Answer the question using ONLY
the context below. If the context does not contain enough information to answer,
say "I couldn't find this in the provided documents." Do not make up information.

Context:
{context_block}

Question: {question}

Answer (cite source filenames where relevant):"""

    response = _client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    )
    return response.text
