"""Gemini wrappers: text -> embedding vector, and retrieved chunks -> answer."""
import re
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

Context sources are numbered [1], [2], etc. When you use information from a
source, cite it inline with its number in square brackets, e.g. "The Republic
began in 509 BC [1]." If a claim draws on more than one source, cite each one,
e.g. "[1][2]". Only cite numbers that appear in the context below.

Context:
{context}

Question: {question}

Answer (with inline [N] citations):"""

_CITATION_NUM_RE = re.compile(r"\[(\d+)\]")


# Number each chunk 1-based, in the order given -- this is the mapping
# parse_citations relies on to turn a model's "[N]" back into a chunk.
def _build_context_block(context_chunks):
    return "\n\n".join(
        f"[{i}] Source: {c['source_file']} (chunk {c['chunk_index']})\n{c['chunk_text']}"
        for i, c in enumerate(context_chunks, start=1)
    )


# Parse "[N]"-style citations out of `answer`, mapping them back to
# context_chunks (the same list/order passed to generate_answer). Returns the
# set of 0-based indices into context_chunks that were actually cited.
# Numbers outside 1..len(context_chunks) are dropped, not raised -- citing a
# number that doesn't exist is itself a hallucination, not a bug to crash on.
def parse_citations(answer, context_chunks):
    n = len(context_chunks)
    cited = set()
    for m in _CITATION_NUM_RE.findall(answer):
        idx = int(m) - 1
        if 0 <= idx < n:
            cited.add(idx)
    return cited


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
    prompt = _GROUNDED_PROMPT.format(context=_build_context_block(context_chunks), question=question)

    response = _with_retry(lambda: _client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    ))
    # response.text is None when the model returns no text (e.g. a safety block).
    return response.text or "I couldn't find this in the provided documents."
