# Eval results

## Run 2026-09-12 15:28:30 UTC

- top_k=4, embedding_model=gemini-embedding-001, generation_model=gemini-3.8-flash, similarity_threshold=0.5
- Retrieval hit rate (in-scope): 6/6
- Mean recall@k (in-scope): 1.00
- Correctness/faithfulness (in-scope): 6/6
- Anti-hallucination (out-of-scope): 2/2
- Overall: 8/8

| id | category | retrieval | recall@k | keyword | pass | expected -> got |
|----|----------|-----------|----------|---------|------|-----------------|
| q1 | in_scope | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |
| q2 | in_scope | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |
| q3 | in_scope | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |
| q4 | in_scope | Y | 1.00 | Y | Y | test-doc2.md -> test-doc2.md |
| q5 | in_scope | Y | 1.00 | Y | Y | test-doc2.md -> test-doc2.md |
| q6 | in_scope | Y | 1.00 | Y | Y | test-doc2.md -> test-doc2.md |
| q7 | out_of_scope | Y | n/a | Y | Y | - -> test-doc1.md |
| q8 | out_of_scope | Y | n/a | Y | Y | - -> test-doc2.md |

## Run 2026-09-16 20:30:26 UTC

- retrieval=compare, top_k=4, embedding_model=gemini-embedding-001, generation_model=gemini-3.8-flash, similarity_threshold=0.5
- Retrieval hit rate (in-scope): vector 3/3, hybrid 3/3
- Mean recall@k (in-scope): vector 1.00, hybrid 1.00
- Correctness/faithfulness (in-scope): 3/3
- Anti-hallucination (out-of-scope): n/a
- Overall: 3/3

| id | category | vec-hit | vec-recall | hyb-hit | hyb-recall | keyword | pass | expected -> got |
|----|----------|---------|------------|---------|------------|---------|------|-----------------|
| q1 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |
| q2 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |
| q3 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | test-doc1.md -> test-doc1.md |

## Run 2026-09-17 00:39:34 UTC

- retrieval=compare, top_k=4, embedding_model=gemini-embedding-001, generation_model=gemini-3.8-flash, similarity_threshold=0.5
- Retrieval hit rate (in-scope): vector 6/6, hybrid 6/6
- Mean recall@k (in-scope): vector 1.00, hybrid 1.00
- Correctness/faithfulness (in-scope): 6/6
- Anti-hallucination (out-of-scope): 2/2
- Citation validity: 8/8
- Overall: 8/8

| id | category | vec-hit | vec-recall | hyb-hit | hyb-recall | keyword | cite | pass | expected -> got |
|----|----------|---------|------------|---------|------------|---------|------|------|-----------------|
| q1 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc1.md -> test-doc1.md |
| q2 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc1.md -> test-doc1.md |
| q3 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc2.md -> test-doc2.md |
| q4 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc2.md -> test-doc2.md |
| q5 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc3.md -> test-doc3.md |
| q6 | in_scope | Y | 1.00 | Y | 1.00 | Y | Y | Y | test-doc3.md -> test-doc3.md |
| q7 | out_of_scope | Y | n/a | Y | n/a | Y | Y | Y | - -> test-doc1.md |
| q8 | out_of_scope | Y | n/a | Y | n/a | Y | Y | Y | - -> test-doc2.md |

