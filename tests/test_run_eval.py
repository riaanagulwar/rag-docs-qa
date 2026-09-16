"""Tests for the pure-logic scoring helpers in eval.run_eval: ground-truth
chunk derivation, retrieval scoring, citation checking, and the report
formatting helpers. db.get_chunks_by_source is monkeypatched -- no Postgres
needed; these never call Gemini either."""
from eval import run_eval


def _patch_chunks_by_source(monkeypatch, chunks_by_file):
    monkeypatch.setattr(
        run_eval.db, "get_chunks_by_source",
        lambda source_file: chunks_by_file.get(source_file, []),
    )


# --- _relevant_chunk_ids / _score_retrieval ---------------------------------

def test_relevant_chunk_ids_matches_chunks_containing_all_keywords(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {
        "a.md": [
            {"chunk_index": 0, "chunk_text": "Title only."},
            {"chunk_index": 1, "chunk_text": "Founded in 509 BC by consuls."},
            {"chunk_index": 2, "chunk_text": "Something about consuls, no date."},
        ],
    })
    q = {"expected_source_file": "a.md", "expected_keywords": ["509 BC", "consuls"]}
    assert run_eval._relevant_chunk_ids(q) == {1}


def test_relevant_chunk_ids_none_for_out_of_scope():
    q = {"expected_source_file": None, "expected_keywords": ["irrelevant"]}
    assert run_eval._relevant_chunk_ids(q) is None


def test_relevant_chunk_ids_none_when_no_chunk_matches(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {"a.md": [{"chunk_index": 0, "chunk_text": "unrelated"}]})
    q = {"expected_source_file": "a.md", "expected_keywords": ["nonexistent phrase"]}
    assert run_eval._relevant_chunk_ids(q) is None


def test_score_retrieval_hit_and_recall(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {
        "a.md": [
            {"chunk_index": 0, "chunk_text": "totally unrelated content"},
            {"chunk_index": 1, "chunk_text": "has the keyword"},
        ],
    })
    q = {"expected_source_file": "a.md", "expected_keywords": ["keyword"]}
    chunks = [{"source_file": "a.md", "chunk_index": 1}]
    score = run_eval._score_retrieval(q, chunks)
    assert score == {"got_source": "a.md", "retrieval_hit": True, "recall_at_k": 1.0}


def test_score_retrieval_miss(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {
        "a.md": [{"chunk_index": 1, "chunk_text": "has the keyword"}],
    })
    q = {"expected_source_file": "a.md", "expected_keywords": ["keyword"]}
    chunks = [{"source_file": "b.md", "chunk_index": 0}]
    score = run_eval._score_retrieval(q, chunks)
    assert score["retrieval_hit"] is False
    assert score["recall_at_k"] == 0.0


# --- _check_citation ---------------------------------------------------------

def test_citation_valid_when_it_cites_the_relevant_chunk(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {
        "a.md": [
            {"chunk_index": 0, "chunk_text": "Title only."},
            {"chunk_index": 1, "chunk_text": "509 BC founding fact."},
        ],
    })
    q = {"category": "in_scope", "expected_source_file": "a.md", "expected_keywords": ["509 BC"]}
    chunks = [
        {"source_file": "a.md", "chunk_index": 0, "chunk_text": "Title only."},
        {"source_file": "a.md", "chunk_index": 1, "chunk_text": "509 BC founding fact."},
    ]
    assert run_eval._check_citation(q, "It happened in 509 BC [2].", chunks) is True


def test_citation_invalid_when_it_cites_a_real_but_irrelevant_chunk(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {
        "a.md": [
            {"chunk_index": 0, "chunk_text": "Title only."},
            {"chunk_index": 1, "chunk_text": "509 BC founding fact."},
        ],
    })
    q = {"category": "in_scope", "expected_source_file": "a.md", "expected_keywords": ["509 BC"]}
    chunks = [
        {"source_file": "a.md", "chunk_index": 0, "chunk_text": "Title only."},
        {"source_file": "a.md", "chunk_index": 1, "chunk_text": "509 BC founding fact."},
    ]
    assert run_eval._check_citation(q, "It happened in 509 BC [1].", chunks) is False


def test_citation_none_when_nothing_cited(monkeypatch):
    _patch_chunks_by_source(monkeypatch, {"a.md": [{"chunk_index": 0, "chunk_text": "509 BC"}]})
    q = {"category": "in_scope", "expected_source_file": "a.md", "expected_keywords": ["509 BC"]}
    chunks = [{"source_file": "a.md", "chunk_index": 0, "chunk_text": "509 BC"}]
    assert run_eval._check_citation(q, "It happened in 509 BC.", chunks) is None


def test_citation_out_of_scope_valid_only_if_nothing_cited():
    q = {"category": "out_of_scope", "expected_source_file": None, "expected_keywords": ["couldn't find"]}
    assert run_eval._check_citation(q, run_eval.REFUSAL, []) is True
    assert run_eval._check_citation(q, "See [1] for more.", [{"source_file": "a.md", "chunk_index": 0}]) is False


# --- report formatting helpers ----------------------------------------------

def test_rate_formats_as_fraction():
    rows = [{"k": True}, {"k": False}, {"k": True}]
    assert run_eval._rate(rows, "k") == "2/3"
    assert run_eval._rate([], "k") == "n/a"


def test_avg_recall_ignores_none_values():
    rows = [{"recall_at_k": 1.0}, {"recall_at_k": None}, {"recall_at_k": 0.5}]
    assert run_eval._avg_recall(rows) == "0.75"
    assert run_eval._avg_recall([{"recall_at_k": None}]) == "n/a"


def test_yn_formats_tri_state():
    assert run_eval._yn(True) == "Y"
    assert run_eval._yn(False) == "N"
    assert run_eval._yn(None) == "n/a"


def test_citation_rate_excludes_uncited_from_denominator():
    rows = [
        {"citation_valid": True},
        {"citation_valid": False},
        {"citation_valid": None},
    ]
    assert run_eval._citation_rate(rows) == "1/2 (1 uncited)"
