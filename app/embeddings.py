"""Gemini wrappers: text -> embedding vector, and retrieved chunks -> answer."""
import time

from google import genai
from google.genai import errors, types

from app import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_MAX_RETRIES = 2
_BASE_DELAY_SECONDS = 2  # doubles each retry: 2s, 4s


# Retry `fn` on transient Gemini errors: 5xx (e.g. 503 model overloaded) and
# 429 (rate limit). Anything else -- bad request, auth, safety block -- is not
# transient and is raised immediately.
def _with_retry(fn):
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except errors.ServerError:
            if attempt == _MAX_RETRIES:
                raise
        except errors.ClientError as e:
            if e.code != 429 or attempt == _MAX_RETRIES:
                raise
        time.sleep(_BASE_DELAY_SECONDS * (2 ** attempt))

# Gemini uses asymmetric embeddings: a chunk being stored and a question being
# searched are embedded with different task types. Keep these two distinct.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

_GROUNDED_PROMPT = """You are a documentation assistant. Answer the question using ONLY
the context below. If the context does not contain enough information to answer,
say "I couldn't find this in the provided documents." Do not make up information.

Context:
{context}

Question: {question}

Answer (cite source filenames where relevant):"""


# Embed a batch of strings in a single API call, each truncated (Matryoshka) to
# config.EMBEDDING_DIMENSIONS. Prefer this over calling embed_text in a loop --
# one call for N texts instead of N calls. Pass TASK_DOCUMENT for chunks being
# stored, TASK_QUERY for user questions.
def embed_texts(texts, task_type=TASK_DOCUMENT):
    if not texts:
        return []
    result = _with_retry(lambda: _client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    ))
    return [e.values for e in result.embeddings]


# Embed a single string. Convenience wrapper for call sites that only ever
# have one piece of text (e.g. one user question) -- for more than one, call
# embed_texts directly instead of looping this.
def embed_text(text, task_type=TASK_DOCUMENT):
    return embed_texts([text], task_type)[0]


# Answer `question` strictly from `context_chunks`. The prompt is the only thing
# stopping the model from hallucinating when retrieval returns a weak match --
# callers should skip this entirely below config.SIMILARITY_THRESHOLD (see
# db.has_relevant_chunk) rather than relying on the prompt alone.
def generate_answer(question, context_chunks):
    context_block = "\n\n".join(
        f"[Source: {c['source_file']} chunk {c['chunk_index']}]\n{c['chunk_text']}"
        for c in context_chunks
    )
    prompt = _GROUNDED_PROMPT.format(context=context_block, question=question)

    response = _with_retry(lambda: _client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    ))
    # response.text is None when the model returns no text (e.g. a safety block).
    return response.text or "I couldn't find this in the provided documents."
