# Multi-domain technical-documentation benchmark

`tech-multidomain-v1` is the primary benchmark for retrieval and agent
experiments. It keeps the original FastAPI `0.115.0` annotations intact and
adds three deliberately different technical documentation domains:

| Domain | Pinned upstream revision | Documents | Frozen test cases |
| --- | --- | ---: | ---: |
| FastAPI | `0.115.0` | 10 | 30 (22 answerable, 8 refusal) |
| Pydantic | `v2.10.6` | 10 | 30 (22 answerable, 8 refusal) |
| Kubernetes | `release-1.31` | 10 | 30 (22 answerable, 8 refusal) |
| SQLAlchemy | `rel_2_0_36` | 10 | 30 (22 answerable, 8 refusal) |
| **Total** | — | **40** | **120 (88 answerable, 32 refusal)** |

The development split has 80 answerable questions (the frozen FastAPI 50 plus
10 per new domain). The test split is the only split for final comparisons.
`benchmark.json` composes the existing FastAPI JSONL files with the new files,
so the older benchmark is not altered or silently relabelled.

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
pkb-agent eval validate data/evals/tech-multidomain-v1
SUPABASE_TIMEOUT=90 pkb-agent eval ingest-corpus data/evals/tech-multidomain-v1 \
  --user eval-tech-multidomain-v1 --timeout 90 --attempts 4 --concurrency 2

# Tune only on development questions.
pkb-agent eval run data/evals/tech-multidomain-v1 --split dev \
  --user eval-tech-multidomain-v1 --repetitions 3

# Use the frozen test only for the final comparison and agent check.
pkb-agent eval run data/evals/tech-multidomain-v1 --split test \
  --user eval-tech-multidomain-v1 --with-agent --repetitions 3
```

Each report shows document-level Recall@K, MRR@K, NDCG@K, engineering metrics,
and a 95% nonparametric bootstrap confidence interval over unique answerable
cases. Repeated runs are averaged within a case before resampling. Each
retrieval strategy is also compared with Vector on every shared question using
NDCG@K, with a win/loss/tie total and an expandable per-question table. These
intervals quantify uncertainty for this fixed benchmark; they do not establish
performance on arbitrary technical corpora.
