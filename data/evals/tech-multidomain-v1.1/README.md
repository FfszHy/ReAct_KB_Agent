# Multi-domain technical-documentation benchmark

`tech-multidomain-v1.1` is the primary benchmark for retrieval and Agent
experiments. It preserves `tech-multidomain-v1@1.0.0` as a historical frozen
artifact, and fixes one source-label defect without silently rewriting it: the
FastAPI query-alias case now cites the pinned page that actually documents
`Query(alias=...)`.

| Domain | Pinned upstream revision | Documents | Frozen test cases |
| --- | --- | ---: | ---: |
| FastAPI | `0.115.0` | 11 | 30 (22 answerable, 8 refusal) |
| Pydantic | `v2.10.6` | 10 | 30 (22 answerable, 8 refusal) |
| Kubernetes | `release-1.31` | 10 | 30 (22 answerable, 8 refusal) |
| SQLAlchemy | `rel_2_0_36` | 10 | 30 (22 answerable, 8 refusal) |
| **Total** | — | **41** | **120 (88 answerable, 32 refusal)** |

The development split has 80 answerable questions (the frozen FastAPI 50 plus
10 per new domain). The test split is the only split for final comparisons.
This version snapshots the FastAPI JSONL files locally and composes them with
the 30 development / 90 test cross-domain cases from v1.0, so the historical
benchmark is not altered or silently relabelled.

`test-query-02` now uses `tutorial/query-params-str-validations`, whose
"Alias parameters" section explicitly describes assigning a public request
name that differs from the Python identifier. The old `tutorial/query-params`
page remains in the corpus, but is no longer treated as evidence for that fact.

## Deliberate challenge coverage

The test questions are tagged and stratified rather than being a random pile
of single-document facts. Across the frozen test set they include 15
multi-document questions, 12 paraphrases, 18 terminology-ambiguity questions,
13 pinned-version traps, 32 questions requiring a grounded refusal, and 8
protected external-fetch requests. The latter exercise the Agent's
`web_fetch` permission path: evaluation denies confirmation, so the expected
behavior is a safe refusal after selecting the protected tool.

Answerable agent cases require `rag_search`; external-page cases require
`web_fetch`. This makes tool selection and permission handling measurable,
instead of inferred from an answer's prose.

## Reproducibility and workflow

`corpus.json` contains only source URLs, pinned revisions, and SHA-256 hashes;
the raw documentation is fetched during ingestion and stored under a dedicated
evaluation user scope in the database. No downloaded source copy needs to be
committed to the repository. The ingestion command verifies every source hash
and writes a `corpus.lock.json` recording the exact bytes actually ingested.

```bash
conda run -n pkb-agent pkb-agent eval validate data/evals/tech-multidomain-v1.1
SUPABASE_TIMEOUT=90 conda run -n pkb-agent pkb-agent eval ingest-corpus data/evals/tech-multidomain-v1.1 \
  --user eval-tech-multidomain-v1.1 --timeout 90 --attempts 4 --concurrency 2

# Tune only on development questions.
SUPABASE_TIMEOUT=90 conda run -n pkb-agent pkb-agent eval run data/evals/tech-multidomain-v1.1 --split dev \
  --user eval-tech-multidomain-v1.1 --repetitions 3

# Primary Agent score: KB catalog + rag_search + rag_read only; no web tools.
SUPABASE_TIMEOUT=90 conda run -n pkb-agent pkb-agent eval run data/evals/tech-multidomain-v1.1 --split test \
  --user eval-tech-multidomain-v1.1 --strategies "" --with-agent \
  --agent-profile kb_only --repetitions 1

# Permission score: web_fetch is available, but non-interactive evaluation denies confirmation.
SUPABASE_TIMEOUT=90 conda run -n pkb-agent pkb-agent eval run data/evals/tech-multidomain-v1.1 --split test \
  --user eval-tech-multidomain-v1.1 --strategies "" --with-agent \
  --agent-profile permission --repetitions 1
```

Use the new user scope exactly as shown. Reusing `eval-tech-multidomain-v1`
would leave the old 40 documents beside the new 41-document corpus and make
retrieval measurements invalid.

Each report shows document-level Recall@K, MRR@K, NDCG@K, engineering metrics,
and a 95% nonparametric bootstrap confidence interval over unique answerable
cases. Repeated runs are averaged within a case before resampling. Each
retrieval strategy is also compared with Vector on every shared question using
NDCG@K, with a win/loss/tie total and an expandable per-question table. These
intervals quantify uncertainty for this fixed benchmark; they do not establish
performance on arbitrary technical corpora. Agent engineering metrics separate
hard execution failures from tool-observation failures, bounded trace-write
degradation, and the normal no-tool finalization path used when retrieval
budget is exhausted.
