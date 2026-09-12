"""Automated eval for the RAG pipeline: retrieval, recall@k,
faithfulness/correctness, and anti-hallucination -- run against the questions
in eval/questions.json.

    python -m eval.run_eval [--questions PATH] [--top-k N]

Requires Postgres running and ingested (docker compose up -d && python -m
app.ingest) with the sample docs in docs/test-doc1.md and
docs/test-doc2.md. If you replace docs/ with your own corpus, update
eval/questions.json to match -- the checked-in questions are written against
the sample corpus.

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
A question passes if retrieval_hit and keyword_pass both hold; recall_at_k is
reported but doesn't gate pass/fail since it's a graded, not binary, signal.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config, db
from app.embeddings import TASK_QUERY, embed_texts, generate_answer

REFUSAL = "I couldn't find this in the provided documents."
DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"
DEFAULT_RESULTS_FILE = Path(__file__).parent / "results.md"

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


# Retrieve, generate, and score one question against its expectations.
# `query_embedding` is precomputed (all questions are embedded together in one
# batch call, see main()). Logs each step so a live run shows what's actually
# happening, not just a silent pause until the final table.
def run_one(q, query_embedding, top_k, position, total):
    log.info("[%s] (%d/%d) %s", q["id"], position, total, q["question"])

    chunks = db.similarity_search(query_embedding, top_k)
    log.info(
        "  retrieved %d chunk(s): %s",
        len(chunks),
        ", ".join(f"{c['source_file']}#{c['chunk_index']} (sim={c['similarity']:.3f})" for c in chunks) or "none",
    )

    # Skip the generation call entirely when nothing retrieved is a close
    # enough match -- both a cost saving and the actual anti-hallucination
    # mechanism now, instead of relying on the prompt alone.
    if db.has_relevant_chunk(chunks):
        answer = generate_answer(q["question"], chunks)
    else:
        answer = REFUSAL
        log.info("  skipped generation (no chunk above SIMILARITY_THRESHOLD)")
    preview = answer if len(answer) <= 150 else answer[:150] + "..."
    log.info("  answer: %s", preview)

    got_sources = [c["source_file"] for c in chunks]
    retrieval_hit = (
        q["expected_source_file"] is None
        or q["expected_source_file"] in got_sources
    )
    keyword_pass = all(kw.lower() in answer.lower() for kw in q["expected_keywords"])

    relevant_ids = _relevant_chunk_ids(q)
    recall_at_k = None
    if relevant_ids is not None:
        retrieved_ids = {c["chunk_index"] for c in chunks if c["source_file"] == q["expected_source_file"]}
        recall_at_k = len(retrieved_ids & relevant_ids) / len(relevant_ids)

    log.info(
        "  retrieval_hit=%s recall@k=%s keyword_pass=%s",
        "Y" if retrieval_hit else "N",
        "n/a" if recall_at_k is None else f"{recall_at_k:.2f}",
        "Y" if keyword_pass else "N",
    )

    return {
        "id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "expected_source": q["expected_source_file"] or "-",
        "got_source": got_sources[0] if got_sources else "-",
        "retrieval_hit": retrieval_hit,
        "recall_at_k": recall_at_k,
        "keyword_pass": keyword_pass,
        "passed": retrieval_hit and keyword_pass,
    }


# Print a per-question table and a summary broken out by metric.
def report(results):
    header = f"{'id':<4} {'cat':<12} {'retrieval':<10} {'recall@k':<9} {'keyword':<8} {'pass':<5} expected -> got"
    print("\n=== Results ===\n")
    print(header)
    print("-" * len(header))
    for r in results:
        recall_str = "n/a" if r["recall_at_k"] is None else f"{r['recall_at_k']:.2f}"
        print(
            f"{r['id']:<4} {r['category']:<12} "
            f"{'Y' if r['retrieval_hit'] else 'N':<10} "
            f"{recall_str:<9} "
            f"{'Y' if r['keyword_pass'] else 'N':<8} "
            f"{'Y' if r['passed'] else 'N':<5} "
            f"{r['expected_source']} -> {r['got_source']}"
        )

    in_scope = [r for r in results if r["category"] == "in_scope"]
    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]

    def rate(rows, key):
        return f"{sum(r[key] for r in rows)}/{len(rows)}" if rows else "n/a"

    def avg_recall(rows):
        vals = [r["recall_at_k"] for r in rows if r["recall_at_k"] is not None]
        return f"{sum(vals) / len(vals):.2f}" if vals else "n/a"

    print()
    print(f"Retrieval hit rate (in-scope):        {rate(in_scope, 'retrieval_hit')}")
    print(f"Mean recall@k (in-scope):             {avg_recall(in_scope)}")
    print(f"Correctness/faithfulness (in-scope):  {rate(in_scope, 'keyword_pass')}")
    print(f"Anti-hallucination (out-of-scope):    {rate(out_of_scope, 'keyword_pass')}")
    print(f"Overall:                              {rate(results, 'passed')}")


# Render the same info as report() into a Markdown section for one run, so
# results are kept somewhere to compare against later, not just on screen.
def _format_run_markdown(results, top_k):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    in_scope = [r for r in results if r["category"] == "in_scope"]
    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]

    def rate(rows, key):
        return f"{sum(r[key] for r in rows)}/{len(rows)}" if rows else "n/a"

    def avg_recall(rows):
        vals = [r["recall_at_k"] for r in rows if r["recall_at_k"] is not None]
        return f"{sum(vals) / len(vals):.2f}" if vals else "n/a"

    lines = [
        f"## Run {timestamp}",
        "",
        f"- top_k={top_k}, embedding_model={config.EMBEDDING_MODEL}, "
        f"generation_model={config.GENERATION_MODEL}, "
        f"similarity_threshold={config.SIMILARITY_THRESHOLD}",
        f"- Retrieval hit rate (in-scope): {rate(in_scope, 'retrieval_hit')}",
        f"- Mean recall@k (in-scope): {avg_recall(in_scope)}",
        f"- Correctness/faithfulness (in-scope): {rate(in_scope, 'keyword_pass')}",
        f"- Anti-hallucination (out-of-scope): {rate(out_of_scope, 'keyword_pass')}",
        f"- Overall: {rate(results, 'passed')}",
        "",
        "| id | category | retrieval | recall@k | keyword | pass | expected -> got |",
        "|----|----------|-----------|----------|---------|------|-----------------|",
    ]
    for r in results:
        recall_str = "n/a" if r["recall_at_k"] is None else f"{r['recall_at_k']:.2f}"
        lines.append(
            f"| {r['id']} | {r['category']} | {'Y' if r['retrieval_hit'] else 'N'} | "
            f"{recall_str} | {'Y' if r['keyword_pass'] else 'N'} | "
            f"{'Y' if r['passed'] else 'N'} | {r['expected_source']} -> {r['got_source']} |"
        )
    lines.append("")
    lines.append("")
    return "\n".join(lines)


# Append this run's results to `path` as a new dated section, so past runs
# stay on record for comparison (e.g. did recall@k regress after a chunking
# change?) instead of only ever being visible in the console.
def save_results(results, top_k, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# Eval results\n\n")
        f.write(_format_run_markdown(results, top_k))
    log.info("Appended results to %s", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=None)
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
    top_k = args.top_k or config.TOP_K
    log.info("Loaded %d question(s) from %s. top_k=%d", len(questions), args.questions, top_k)

    # One embedding call for every question instead of one per question.
    query_embeddings = embed_texts([q["question"] for q in questions], task_type=TASK_QUERY)

    results = []
    for i, (q, query_embedding) in enumerate(zip(questions, query_embeddings), start=1):
        results.append(run_one(q, query_embedding, top_k, i, len(questions)))
        time.sleep(0.2)  # stay under free-tier rate limits between generation calls

    report(results)
    save_results(results, top_k, args.output)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
