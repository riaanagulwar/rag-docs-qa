"""Tests for app.embeddings: context numbering, citation parsing, and the
retry-on-transient-error wrapper. No real Gemini calls -- google.genai.Client
is patched out in conftest.py, and _client.models.* is stubbed per test."""
from unittest import mock

import pytest
from google.genai import errors

from app import embeddings as emb


# --- _build_context_block / parse_citations ---------------------------------

def test_build_context_block_numbers_chunks_from_one():
    chunks = [
        {"source_file": "a.md", "chunk_index": 0, "chunk_text": "Alpha."},
        {"source_file": "b.md", "chunk_index": 2, "chunk_text": "Beta."},
    ]
    block = emb._build_context_block(chunks)
    assert "[1] Source: a.md (chunk 0)\nAlpha." in block
    assert "[2] Source: b.md (chunk 2)\nBeta." in block


def test_parse_citations_single_and_multi():
    chunks = [{}, {}, {}]  # only length matters to parse_citations
    assert emb.parse_citations("Fact [1].", chunks) == {0}
    assert emb.parse_citations("Combined [1][2].", chunks) == {0, 1}


def test_parse_citations_drops_out_of_range_and_zero():
    chunks = [{}, {}]
    assert emb.parse_citations("Bogus [7].", chunks) == set()
    assert emb.parse_citations("Bad [0].", chunks) == set()  # 1-based, 0 is invalid


def test_parse_citations_empty_when_nothing_cited():
    assert emb.parse_citations("No brackets here.", [{}]) == set()


# --- _with_retry --------------------------------------------------------------

def test_retry_recovers_after_transient_server_errors(monkeypatch):
    monkeypatch.setattr(emb, "_MAX_RETRIES", 2)
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})
        return "ok"

    assert emb._with_retry(flaky) == "ok"
    assert attempts["n"] == 3


def test_retry_gives_up_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(emb, "_MAX_RETRIES", 2)
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)

    def always_fails():
        raise errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})

    with pytest.raises(errors.ServerError):
        emb._with_retry(always_fails)


def test_retry_does_not_retry_non_transient_client_errors(monkeypatch):
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def bad_request():
        attempts["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(errors.ClientError):
        emb._with_retry(bad_request)
    assert attempts["n"] == 1


def test_retry_does_retry_rate_limit_429(monkeypatch):
    monkeypatch.setattr(emb, "_MAX_RETRIES", 2)
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def rate_limited_then_ok():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise errors.ClientError(429, {"error": {"message": "rate limited", "status": "RESOURCE_EXHAUSTED"}})
        return "ok"

    assert emb._with_retry(rate_limited_then_ok) == "ok"
    assert attempts["n"] == 2


# --- generate_answer / embed_texts (thin wrappers around the client) -------

def test_generate_answer_falls_back_to_refusal_on_empty_response():
    emb._client.models.generate_content = mock.Mock(return_value=mock.Mock(text=None))
    answer = emb.generate_answer("What?", [{"source_file": "a.md", "chunk_index": 0, "chunk_text": "..."}])
    assert answer == "I couldn't find this in the provided documents."


def test_embed_texts_returns_empty_list_for_empty_input():
    assert emb.embed_texts([]) == []
