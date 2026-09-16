"""Tests for the pure-logic parts of app.db: pgvector literal formatting,
Reciprocal Rank Fusion, the relevance threshold, and hybrid_search's wiring.
No real Postgres connection is made -- _connect()/cursor-based functions
(similarity_search, keyword_search, insert_chunks, ...) need a live database
and aren't covered here; that's what `python -m eval.run_eval` verifies."""
from app import db


def test_to_pgvector_literal_formats_as_bracketed_csv():
    assert db._to_pgvector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_to_pgvector_literal_coerces_to_float():
    assert db._to_pgvector_literal([1, 2, 3]) == "[1.0,2.0,3.0]"


def test_has_relevant_chunk_respects_threshold(monkeypatch):
    monkeypatch.setattr(db.config, "SIMILARITY_THRESHOLD", 0.5)
    assert db.has_relevant_chunk([{"similarity": 0.6}]) is True
    assert db.has_relevant_chunk([{"similarity": 0.4}]) is False
    assert db.has_relevant_chunk([]) is False


def test_rrf_rewards_a_chunk_present_in_both_lists():
    # rank 3 in vector, rank 1 in keyword -- should still beat something
    # that's rank 1 in vector alone.
    vector_results = [
        {"source_file": "a.md", "chunk_index": 0, "similarity": 0.9},
        {"source_file": "a.md", "chunk_index": 1, "similarity": 0.8},
        {"source_file": "b.md", "chunk_index": 0, "similarity": 0.7},
    ]
    keyword_results = [
        {"source_file": "b.md", "chunk_index": 0, "similarity": 0.5},
    ]
    fused = db._reciprocal_rank_fusion([vector_results, keyword_results])
    assert (fused[0]["source_file"], fused[0]["chunk_index"]) == ("b.md", 0)


def test_rrf_ranking_is_order_independent_of_input_list_order():
    a = [{"source_file": "x.md", "chunk_index": 0}]
    b = [{"source_file": "y.md", "chunk_index": 0}]
    fused_ab = db._reciprocal_rank_fusion([a, b])
    fused_ba = db._reciprocal_rank_fusion([b, a])
    # same items, same scores either way (a always rank-1 in its own list)
    assert {r["source_file"] for r in fused_ab} == {r["source_file"] for r in fused_ba}


def test_hybrid_search_fuses_vector_and_keyword_results(monkeypatch):
    vector_results = [
        {"source_file": "a.md", "chunk_index": 0, "chunk_text": "x", "similarity": 0.9},
        {"source_file": "b.md", "chunk_index": 0, "chunk_text": "z", "similarity": 0.4},
    ]
    keyword_results = [
        {"source_file": "b.md", "chunk_index": 0, "chunk_text": "z", "similarity": 3.1},
    ]
    monkeypatch.setattr(db, "similarity_search", lambda emb, k: vector_results)
    monkeypatch.setattr(db, "keyword_search", lambda q, k: keyword_results)

    result = db.hybrid_search("some query", [0.1] * 768, top_k=2)

    # b.md#0 is the strongest keyword hit and present in both lists -> wins
    assert (result[0]["source_file"], result[0]["chunk_index"]) == ("b.md", 0)
    assert len(result) == 2
