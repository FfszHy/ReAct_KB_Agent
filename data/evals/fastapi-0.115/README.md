# FastAPI 0.115 benchmark

This is the project’s public, reproducible technical-documentation benchmark.
It pins FastAPI’s upstream `0.115.0` source revision and evaluates retrieval at
the document level; database UUIDs are never part of the labels.

- `questions.dev.jsonl` contains 50 development questions. It is the only
  split to use while tuning weights, prompts, chunking, or a reranker.
- `questions.test.jsonl` contains 30 frozen final questions. Do not tune on
  this split.
- Each answerable row identifies graded relevant source documents and the
  atomic facts an answer reviewer should check. Eight negative cases assert that
  the agent refuses unsupported questions rather than inventing facts.
- `corpus.json` is fetched and ingested as stable `eval://fastapi-0.115/<key>`
  URIs. `pkb-agent eval ingest-corpus` writes `corpus.lock.json` with exact
  bytes hashed at ingestion time.

Run the workflow after configuring Supabase, embeddings, and (for query rewrite
or end-to-end agent evaluation) DeepSeek:

```bash
pkb-agent eval validate data/evals/fastapi-0.115
pkb-agent eval ingest-corpus data/evals/fastapi-0.115
pkb-agent eval run data/evals/fastapi-0.115 --split dev --repetitions 3
pkb-agent eval run data/evals/fastapi-0.115 --split test --repetitions 3 --with-agent
```

The `eval run` commands create raw JSONL, a static HTML dashboard, a Markdown
report, and an SVG comparison chart under `artifacts/evals/`. Agent runs also
create an editable human-audit sheet. Retrieval reports include 95% bootstrap intervals over
unique answerable questions plus NDCG@K per-question win/loss/tie against
Vector; repetitions are averaged inside a question before comparison. Complete
the audit sheet and re-score it with `pkb-agent eval report` before claiming
semantic citation precision or factual consistency in a README or résumé.
