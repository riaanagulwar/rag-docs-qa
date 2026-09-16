"""Tests for app.main's request handlers, called directly (not through an
HTTP client -- avoids needing httpx as an extra test dependency). db and
Gemini calls are monkeypatched; nothing here touches a real Postgres or
Gemini."""
from unittest import mock

import pytest
from fastapi import HTTPException
from google.genai import errors as genai_errors

from app import main


def _stub_chunks():
    return [{
        "source_file": "test-doc1.md", "chunk_index": 1,
        "chunk_text": "The Republic began in 509 BC.", "similarity": 0.9, "rrf_score": 0.03,
    }]


def test_ask_uses_hybrid_search_not_vector_only(monkeypatch):
    calls = {"hybrid": 0, "vector": 0}
    monkeypatch.setattr(main.db, "hybrid_search", lambda q, e, k: (calls.__setitem__("hybrid", calls["hybrid"] + 1), _stub_chunks())[1])
    monkeypatch.setattr(main.db, "similarity_search", lambda e, k: (calls.__setitem__("vector", calls["vector"] + 1), _stub_chunks())[1])
    monkeypatch.setattr(main.db, "has_relevant_chunk", lambda c: True)
    monkeypatch.setattr(main, "embed_text", lambda q, task_type: [0.1])
    monkeypatch.setattr(main, "generate_answer", lambda q, c: "The Republic began in 509 BC [1].")

    main.ask(main.AskRequest(question="When did the Republic begin?"))

    assert calls == {"hybrid": 1, "vector": 0}


def test_ask_marks_cited_sources(monkeypatch):
    monkeypatch.setattr(main.db, "hybrid_search", lambda q, e, k: _stub_chunks())
    monkeypatch.setattr(main.db, "has_relevant_chunk", lambda c: True)
    monkeypatch.setattr(main, "embed_text", lambda q, task_type: [0.1])
    monkeypatch.setattr(main, "generate_answer", lambda q, c: "The Republic began in 509 BC [1].")

    resp = main.ask(main.AskRequest(question="When?"))

    assert resp.sources[0].cited is True
    assert resp.answer == "The Republic began in 509 BC [1]."


def test_ask_refuses_without_calling_generate_when_nothing_relevant(monkeypatch):
    generate_called = {"n": 0}
    monkeypatch.setattr(main.db, "hybrid_search", lambda q, e, k: [])
    monkeypatch.setattr(main.db, "has_relevant_chunk", lambda c: False)
    monkeypatch.setattr(main, "embed_text", lambda q, task_type: [0.1])
    monkeypatch.setattr(main, "generate_answer", lambda q, c: generate_called.__setitem__("n", generate_called["n"] + 1))

    resp = main.ask(main.AskRequest(question="Unrelated question?"))

    assert resp.answer == main._REFUSAL
    assert resp.sources == []
    assert generate_called["n"] == 0


def test_ask_rejects_empty_question():
    with pytest.raises(HTTPException) as exc_info:
        main.ask(main.AskRequest(question="   "))
    assert exc_info.value.status_code == 400


def test_ask_returns_503_when_database_is_unreachable(monkeypatch):
    monkeypatch.setattr(main.db, "hybrid_search", mock.Mock(side_effect=RuntimeError("connection refused")))
    monkeypatch.setattr(main, "embed_text", lambda q, task_type: [0.1])

    with pytest.raises(HTTPException) as exc_info:
        main.ask(main.AskRequest(question="q?"))
    assert exc_info.value.status_code == 503


def test_ask_returns_502_when_embedding_call_fails(monkeypatch):
    monkeypatch.setattr(
        main, "embed_text",
        mock.Mock(side_effect=genai_errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})),
    )
    with pytest.raises(HTTPException) as exc_info:
        main.ask(main.AskRequest(question="q?"))
    assert exc_info.value.status_code == 502


def test_health_reports_ok_with_chunk_count(monkeypatch):
    monkeypatch.setattr(main.db, "count_chunks", lambda: 18)
    assert main.health() == {"status": "ok", "chunks": 18}


def test_health_returns_503_when_database_is_unreachable(monkeypatch):
    monkeypatch.setattr(main.db, "count_chunks", mock.Mock(side_effect=RuntimeError("down")))
    with pytest.raises(HTTPException) as exc_info:
        main.health()
    assert exc_info.value.status_code == 503
