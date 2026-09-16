"""Automated eval for the RAG pipeline: retrieval, recall@k,
faithfulness/correctness, and anti-hallucination -- run against the questions
in eval/questions.json.

    python -m eval.run_eval [--questions PATH] [--top-k N] [--retrieval MODE]

--retrieval chooses what does retrieval:
    vector   (default) -- db.similarity_search only, unchanged baseline.
    hybrid   -- db.hybrid_search only (vector + keyword, fused via RRF).
    compare  -- runs BOTH for every question and reports them side by side,
                so you can see whether keyword search actually helps. Answer
                generation only runs once per question, from the hybrid
                result -- comparing retrieval costs zero extra Gemini calls
                (keyword search is Postgres-only, and the question embedding
                is already computed once for the whole run).

Requires Postgres running and ingested (docker compose up -d && python -m
app.ingest) with the sample docs in docs/test-doc1.md, docs/test-doc2.md, and
docs/test-doc3.md.
hybrid/compare additionally require the chunk_tsv column from schema.sql,
which only exists on a fresh Postgres volume -- see schema.sql's note. If you
replace docs/ with your own corpus, update eval/questions.json to match.

Each question is graded on:
- retrieval_hit: did the expected source file come back in the top-k chunks?
               (always True for out-of-scope questions, which have none)
- recall_at_k: of the chunks that actually contain the expected keywords
               (the "relevant" chunks, found by scanning every chunk of the
               expected source file), what fraction came back in the top-k?
               None when there's no ground truth to compare against (e.g.
               out-of-scope questions, or keywords that don't literally
               appear in any single chunk).
- keyword_pass: does the generated answer contain every expected keyword?
               For in-scope questions this is the correctness/faithfulness
               signal (a wrong or hallucinated answer won't contain the real
               chunk's keyword). For out-of-scope questions the "keyword" is
               the refusal phrase, so this doubles as the anti-hallucination
               check -- though most out-of-scope questions never reach
               generation at all: below config.SIMILARITY_THRESHOLD the
               refusal is returned directly, skipping the API call.
- citation_valid: does the answer's inline [N] citation (see
               embeddings.parse_citations) actually point at a relevant
               chunk? True/False, or None when the answer cited nothing at
               all (nothing to verify). Out-of-scope: valid only if nothing
               is cited. In-scope: valid only if a cited chunk's index is
               among the ones _relevant_chunk_ids already identified as
               containing the expected keywords -- not just any retrieved
               chunk. Doesn't gate pass/fail -- plenty of correct answers
               don't bother citing inline.
A question passes if retrieval_hit and keyword_pass both hold; recall_at_k
and citation_valid are reported but don't gate pass/fail, since both are
graded/optional signals rather than binary requirements.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config, db
from app.embeddings import TASK_QUERY, embed_texts, generate_answer, parse_citations

REFUSAL = "I couldn't find this in the provided documents."
DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"
DEFAULT_RESULTS_FILE = Path(__file__).parent / "results.md"

# Paced between generation calls (embeddings are batched into one call for
# the whole run, so this only spaces out generate_answer) to stay under
# Gemini's free-tier per-minute rate limit -- a tight loop of back-to-back
# calls hits it well before the daily quota does.
_INTER_QUESTION_DELAY_SECONDS = 4

# Progress goes through logging (stderr) so it stays separate from the
# final report table (printed to stdout) -- redirecting stdout to a file
# captures a clean report while progress still shows on screen.
log = logging.getLogger("eval")


# Ground-truth "relevant" chunks for a question: every chunk of its expected
# source file that contains all of the expected keywords. Derived from the
# question set rather than hand-picked chunk indices, so it stays correct
# across re-chunking (a different chunk_size just changes which/how many
# chunks match).
def _relevant_chunk_ids(q):
    if q["expected_source_file"] is None:
        return None
    chunks = db.get_chunks_by_source(q["expected_source_file"])
    relevant = [
        c["chunk_index"] for c in chunks
        if all(kw.lower() in c["chunk_text"].lower() for kw in q["expected_keywords"])
    ]
    return set(relevant) if relevant else None


# Score one retrieval result set against a question's expectations. Doesn't
# care which retrieval method produced `chunks` -- used for both the vector
# and hybrid sides of a comparison.
def _score_retrieval(q, chunks):
    got_sources = [c["source_file"] for c in chunks]
    retrieval_hit = (
        q["expected_source_file"] is None
        or q["expected_source_file"] in got_sources
    )
    relevant_ids = _relevant_chunk_ids(q)
    recall_at_k = None
    if relevant_ids is not None:
        retrieved_ids = {c["chunk_index"] for c in chunks if c["source_file"] == q["expected_source_file"]}
        recall_at_k = len(retrieved_ids & relevant_ids) / len(relevant_ids)
    return {
        "got_source": got_sources[0] if got_sources else "-",
        "retrieval_hit": retrieval_hit,
        "recall_at_k": recall_at_k,
    }


# Whether the answer's inline [N] citation(s) actually point at a relevant
# chunk, not just any retrieved one. `chunks` is whatever was actually passed
# to generate_answer, so indices from parse_citations line up with it.
# out_of_scope: valid only if nothing is cited.
# in_scope: valid if a cited chunk (restricted to the expected source file)
# is one of the chunks _relevant_chunk_ids already identified as containing
# the expected keywords -- reuses that same "relevant chunk" concept rather
# than just checking the citation isn't fabricated. None (not False) when
# the answer cites nothing -- nothing to verify, and plenty of correct
# answers don't bother citing inline.
def _check_citation(q, answer, chunks):
    cited_indices = parse_citations(answer, chunks)

    if q["category"] == "out_of_scope":
        return not cited_indices
    if not cited_indices:
        return None

    relevant_ids = _relevant_chunk_ids(q)
    if relevant_ids is None:
        # No ground-truth chunk indices to check against -- fall back to
        # "cited something from the expected source file at all".
        cited_sources = {chunks[i]["source_file"] for i in cited_indices}
        return q["expected_source_file"] in cited_sources

    cited_chunk_indices = {
        chunks[i]["chunk_index"] for i in cited_indices
        if chunks[i]["source_file"] == q["expected_source_file"]
    }
    return bool(cited_chunk_indices & relevant_ids)


def _format_chunk(c):
    # Fused (hybrid) results carry rrf_score -- that's what actually decided
    # their order, so show it instead of the leftover cosine similarity from
    # whichever source list the chunk was first seen in (which would be
    # misleading: it doesn't explain the ranking, RRF score does).
    if "rrf_score" in c:
        return f"{c['source_file']}#{c['chunk_index']} (rrf={c['rrf_score']:.4f})"
    return f"{c['source_file']}#{c['chunk_index']} (sim={c['similarity']:.3f})"


def _log_chunks(label, chunks):
    log.info(
        "  %s: retrieved %d chunk(s): %s",
        label,
        len(chunks),
        ", ".join(_format_chunk(c) for c in chunks) or "none",
    )


# Retrieve, generate, and score one question. `query_embedding` is
# precomputed (all questions are embedded together in one batch call, see
# main()). `retrieval_mode` is "vector", "hybrid", or "compare" (both, with
# hybrid used for generation). Logs each step so a live run shows what's
# actually happening, not just a silent pause until the final table.
def run_one(q, query_embedding, top_k, position, total, retrieval_mode="vector"):
    log.info("[%s] (%d/%d) %s", q["id"], position, total, q["question"])

    vector_chunks = db.similarity_search(query_embedding, top_k)
    _log_chunks("vector", vector_chunks)

    hybrid_chunks = None
    if retrieval_mode in ("hybrid", "compare"):
        hybrid_chunks = db.hybrid_search(q["question"], query_embedding, top_k)
        _log_chunks("hybrid", hybrid_chunks)

    # What generation actually uses: hybrid when requested, vector otherwise.
    primary_chunks = hybrid_chunks if hybrid_chunks is not None else vector_chunks

    # Skip the generation call entirely when nothing retrieved is a close
    # enough match -- both a cost saving and the actual anti-hallucination
    # mechanism now, instead of relying on the prompt alone.
    if db.has_relevant_chunk(primary_chunks):
        answer = generate_answer(q["question"], primary_chunks)
    else:
        answer = REFUSAL
        log.info("  skipped generation (no chunk above SIMILARITY_THRESHOLD)")
    preview = answer if len(answer) <= 150 else answer[:150] + "..."
    log.info("  answer: %s", preview)

    cited_indices = parse_citations(answer, primary_chunks)
    log.info(
        "  cited: %s",
        ", ".join(_format_chunk(primary_chunks[i]) for i in sorted(cited_indices)) or "none",
    )

    keyword_pass = all(kw.lower() in answer.lower() for kw in q["expected_keywords"])
    citation_valid = _check_citation(q, answer, primary_chunks)
    primary_score = _score_retrieval(q, primary_chunks)

    log.info(
        "  retrieval_hit=%s recall@k=%s keyword_pass=%s citation=%s",
        "Y" if primary_score["retrieval_hit"] else "N",
        "n/a" if primary_score["recall_at_k"] is None else f"{primary_score['recall_at_k']:.2f}",
        "Y" if keyword_pass else "N",
        "n/a" if citation_valid is None else ("Y" if citation_valid else "N"),
    )

    result = {
        "id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "expected_source": q["expected_source_file"] or "-",
        "got_source": primary_score["got_source"],
        "retrieval_hit": primary_score["retrieval_hit"],
        "recall_at_k": primary_score["recall_at_k"],
        "keyword_pass": keyword_pass,
        "citation_valid": citation_valid,
        "passed": primary_score["retrieval_hit"] and keyword_pass,
    }

    # In compare mode, primary_* above is hybrid (since hybrid_chunks was
    # computed); attach vector's numbers too as the baseline to show
    # alongside it.
    if retrieval_mode == "compare":
        vector_score = _score_retrieval(q, vector_chunks)
        result["vector_got_source"] = vector_score["got_source"]
        result["vector_retrieval_hit"] = vector_score["retrieval_hit"]
        result["vector_recall_at_k"] = vector_score["recall_at_k"]

    return result


def _rate(rows, key):
    return f"{sum(r[key] for r in rows)}/{len(rows)}" if rows else "n/a"


def _avg_recall(rows, key="recall_at_k"):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals) / len(vals):.2f}" if vals else "n/a"


def _yn(value):
    return "n/a" if value is None else ("Y" if value else "N")


# citation_valid is True/False/None -- exclude the Nones (nothing cited, so
# nothing to verify) from the rate rather than counting them as failures.
def _citation_rate(rows):
    checked = [r for r in rows if r["citation_valid"] is not None]
    if not checked:
        return "n/a"
    valid = sum(1 for r in checked if r["citation_valid"])
    uncited = len(rows) - len(checked)
    suffix = f" ({uncited} uncited)" if uncited else ""
    return f"{valid}/{len(checked)}{suffix}"


# Print a per-question table and a summary broken out by metric. Grows a
# second retrieval/recall column pair (vector vs. hybrid) when any result
# carries compare-mode data.
def report(results):
    is_compare = any("vector_retrieval_hit" in r for r in results)

    if is_compare:
        header = (f"{'id':<4} {'cat':<12} {'vec-hit':<8} {'vec-rec':<8} "
                   f"{'hyb-hit':<8} {'hyb-rec':<8} {'keyword':<8} {'cite':<6} {'pass':<5} expected -> got")
    else:
        header = (f"{'id':<4} {'cat':<12} {'retrieval':<10} {'recall@k':<9} "
                   f"{'keyword':<8} {'cite':<6} {'pass':<5} expected -> got")

    print("\n=== Results ===\n")
    print(header)
    print("-" * len(header))
    for r in results:
        recall_str = "n/a" if r["recall_at_k"] is None else f"{r['recall_at_k']:.2f}"
        if is_compare:
            vec_recall_str = "n/a" if r["vector_recall_at_k"] is None else f"{r['vector_recall_at_k']:.2f}"
            print(
                f"{r['id']:<4} {r['category']:<12} "
                f"{'Y' if r['vector_retrieval_hit'] else 'N':<8} {vec_recall_str:<8} "
                f"{'Y' if r['retrieval_hit'] else 'N':<8} {recall_str:<8} "
                f"{'Y' if r['keyword_pass'] else 'N':<8} "
                f"{_yn(r['citation_valid']):<6} "
                f"{'Y' if r['passed'] else 'N':<5} "
                f"{r['expected_source']} -> {r['got_source']}"
            )
        else:
            print(
                f"{r['id']:<4} {r['category']:<12} "
                f"{'Y' if r['retrieval_hit'] else 'N':<10} "
                f"{recall_str:<9} "
                f"{'Y' if r['keyword_pass'] else 'N':<8} "
                f"{_yn(r['citation_valid']):<6} "
                f"{'Y' if r['passed'] else 'N':<5} "
                f"{r['expected_source']} -> {r['got_source']}"
            )

    in_scope = [r for r in results if r["category"] == "in_scope"]
    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]

    print()
    if is_compare:
        print(f"Retrieval hit rate (in-scope)   vector: {_rate(in_scope, 'vector_retrieval_hit'):<8} "
              f"hybrid: {_rate(in_scope, 'retrieval_hit')}")
        print(f"Mean recall@k (in-scope)        vector: {_avg_recall(in_scope, 'vector_recall_at_k'):<8} "
              f"hybrid: {_avg_recall(in_scope, 'recall_at_k')}")
    else:
        print(f"Retrieval hit rate (in-scope):        {_rate(in_scope, 'retrieval_hit')}")
        print(f"Mean recall@k (in-scope):             {_avg_recall(in_scope)}")
    print(f"Correctness/faithfulness (in-scope):  {_rate(in_scope, 'keyword_pass')}")
    print(f"Anti-hallucination (out-of-scope):    {_rate(out_of_scope, 'keyword_pass')}")
    print(f"Citation validity:                    {_citation_rate(results)}")
    print(f"Overall:                              {_rate(results, 'passed')}")


# Render the same info as report() into a Markdown section for one run, so
# results are kept somewhere to compare against later, not just on screen.
def _format_run_markdown(results, top_k, retrieval_mode):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    is_compare = any("vector_retrieval_hit" in r for r in results)
    in_scope = [r for r in results if r["category"] == "in_scope"]
    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]

    lines = [
        f"## Run {timestamp}",
        "",
        f"- retrieval={retrieval_mode}, top_k={top_k}, embedding_model={config.EMBEDDING_MODEL}, "
        f"generation_model={config.GENERATION_MODEL}, "
        f"similarity_threshold={config.SIMILARITY_THRESHOLD}",
    ]
    if is_compare:
        lines.append(f"- Retrieval hit rate (in-scope): vector {_rate(in_scope, 'vector_retrieval_hit')}, "
                      f"hybrid {_rate(in_scope, 'retrieval_hit')}")
        lines.append(f"- Mean recall@k (in-scope): vector {_avg_recall(in_scope, 'vector_recall_at_k')}, "
                      f"hybrid {_avg_recall(in_scope, 'recall_at_k')}")
    else:
        lines.append(f"- Retrieval hit rate (in-scope): {_rate(in_scope, 'retrieval_hit')}")
        lines.append(f"- Mean recall@k (in-scope): {_avg_recall(in_scope)}")
    lines += [
        f"- Correctness/faithfulness (in-scope): {_rate(in_scope, 'keyword_pass')}",
        f"- Anti-hallucination (out-of-scope): {_rate(out_of_scope, 'keyword_pass')}",
        f"- Citation validity: {_citation_rate(results)}",
        f"- Overall: {_rate(results, 'passed')}",
        "",
    ]

    if is_compare:
        lines.append("| id | category | vec-hit | vec-recall | hyb-hit | hyb-recall | keyword | cite | pass | expected -> got |")
        lines.append("|----|----------|---------|------------|---------|------------|---------|------|------|-----------------|")
        for r in results:
            recall_str = "n/a" if r["recall_at_k"] is None else f"{r['recall_at_k']:.2f}"
            vec_recall_str = "n/a" if r["vector_recall_at_k"] is None else f"{r['vector_recall_at_k']:.2f}"
            lines.append(
                f"| {r['id']} | {r['category']} | {'Y' if r['vector_retrieval_hit'] else 'N'} | "
                f"{vec_recall_str} | {'Y' if r['retrieval_hit'] else 'N'} | {recall_str} | "
                f"{'Y' if r['keyword_pass'] else 'N'} | {_yn(r['citation_valid'])} | "
                f"{'Y' if r['passed'] else 'N'} | {r['expected_source']} -> {r['got_source']} |"
            )
    else:
        lines.append("| id | category | retrieval | recall@k | keyword | cite | pass | expected -> got |")
        lines.append("|----|----------|-----------|----------|---------|------|------|-----------------|")
        for r in results:
            recall_str = "n/a" if r["recall_at_k"] is None else f"{r['recall_at_k']:.2f}"
            lines.append(
                f"| {r['id']} | {r['category']} | {'Y' if r['retrieval_hit'] else 'N'} | "
                f"{recall_str} | {'Y' if r['keyword_pass'] else 'N'} | {_yn(r['citation_valid'])} | "
                f"{'Y' if r['passed'] else 'N'} | {r['expected_source']} -> {r['got_source']} |"
            )

    lines.append("")
    lines.append("")
    return "\n".join(lines)


# Append this run's results to `path` as a new dated section, so past runs
# stay on record for comparison (e.g. did recall@k regress after a chunking
# change?) instead of only ever being visible in the console.
def save_results(results, top_k, retrieval_mode, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# Eval results\n\n")
        f.write(_format_run_markdown(results, top_k, retrieval_mode))
    log.info("Appended results to %s", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the first N questions (useful for a quick smoke test)")
    parser.add_argument("--retrieval", choices=["vector", "hybrid", "compare"], default="vector",
                         help="retrieval method to use/compare (default: vector)")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE,
                         help="Markdown file to append this run's results to (default: eval/results.md)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-question progress logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    try:
        total_chunks = db.count_chunks()
    except Exception as e:
        print(f"ERROR: could not reach Postgres at {config.DB_HOST}:{config.DB_PORT} "
              f"-- is `docker compose up -d` running? ({e})")
        sys.exit(1)

    if total_chunks == 0:
        print("ERROR: doc_chunks is empty. Run `python -m app.ingest` first.")
        sys.exit(1)

    questions = json.loads(args.questions.read_text())
    if args.limit is not None:
        questions = questions[:args.limit]
    top_k = args.top_k or config.TOP_K
    log.info("Loaded %d question(s) from %s. top_k=%d retrieval=%s", len(questions), args.questions, top_k, args.retrieval)

    # One embedding call for every question instead of one per question.
    query_embeddings = embed_texts([q["question"] for q in questions], task_type=TASK_QUERY)

    results = []
    for i, (q, query_embedding) in enumerate(zip(questions, query_embeddings), start=1):
        results.append(run_one(q, query_embedding, top_k, i, len(questions), args.retrieval))
        if i < len(questions):  # no need to wait after the last question
            time.sleep(_INTER_QUESTION_DELAY_SECONDS)

    report(results)
    save_results(results, top_k, args.retrieval, args.output)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
