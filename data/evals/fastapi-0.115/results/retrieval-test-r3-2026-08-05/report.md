# FastAPI 0.115 technical-documentation RAG benchmark

Benchmark: `fastapi-0.115` · version `1.0.0` · corpus benchmark has 80 cases; this run has 30 unique cases (test) · generated 2026-08-05T07:55:13.956796+00:00

## Retrieval and engineering comparison

| Strategy | Recall@6 | MRR@6 | NDCG@6 | p50 | p95 | Known LLM cost | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | 1.000 | 0.977 | 0.978 | 2230 ms | 3524 ms | — | 0.0% |
| rewrite_hybrid | 1.000 | 1.000 | 0.988 | 4014 ms | 5288 ms | 0.000436 | 0.0% |
| rewrite_rrf | 0.955 | 0.932 | 0.925 | 4494 ms | 6849 ms | 0.000451 | 0.0% |
| rrf | 1.000 | 0.977 | 0.978 | 2552 ms | 4182 ms | — | 0.0% |
| vector | 0.909 | 0.833 | 0.849 | 2091 ms | 6464 ms | — | 6.7% |

## Interpretation safeguards

- Retrieval metrics are document-level: a wider chunk candidate pool is deduplicated by source before scoring K documents.
- Citation alignment is an automatic source-match proxy, not a semantic entailment claim.
- Human-audit metrics remain blank until the generated audit JSONL is reviewed.
- Known LLM cost covers only usage emitted by the configured provider; it is not an invoice.
