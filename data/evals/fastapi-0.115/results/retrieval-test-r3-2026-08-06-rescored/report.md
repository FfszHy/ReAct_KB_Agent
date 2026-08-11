# FastAPI 0.115 technical-documentation RAG benchmark

Benchmark: `fastapi-0.115` · version `1.0.0` · corpus benchmark has 80 cases; this run has 30 unique cases (test) · generated 2026-08-06T06:35:52.108410+00:00

## Retrieval and engineering comparison

| Strategy | Recall@6 | MRR@6 | NDCG@6 | p50 | p95 | Known LLM cost | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | 1.000 | 0.977 | 0.978 | 2230 ms | 3524 ms | — | 0.0% |
| rewrite_hybrid | 1.000 | 1.000 | 0.988 | 4014 ms | 5288 ms | 0.000436 | 0.0% |
| rewrite_rrf | 0.955 | 0.932 | 0.925 | 4494 ms | 6849 ms | 0.000451 | 0.0% |
| rrf | 1.000 | 0.977 | 0.978 | 2552 ms | 4182 ms | — | 0.0% |
| vector | 0.909 | 0.833 | 0.849 | 2091 ms | 6464 ms | — | 6.7% |

## Bootstrap uncertainty

95% nonparametric percentile bootstrap intervals over unique answerable cases; repeated executions are averaged within each case.

| Strategy | Recall@6 95% CI | MRR@6 95% CI | NDCG@6 95% CI | Cases |
| --- | ---: | ---: | ---: | ---: |
| hybrid | 1.000 [1.000, 1.000] | 0.977 [0.932, 1.000] | 0.978 [0.938, 1.000] | 22 |
| rewrite_hybrid | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.988 [0.966, 1.000] | 22 |
| rewrite_rrf | 0.955 [0.864, 1.000] | 0.932 [0.818, 1.000] | 0.925 [0.828, 0.997] | 22 |
| rrf | 1.000 [1.000, 1.000] | 0.977 [0.932, 1.000] | 0.978 [0.938, 1.000] | 22 |
| vector | 0.909 [0.773, 1.000] | 0.833 [0.689, 0.955] | 0.849 [0.704, 0.966] | 22 |

## Case-level wins, losses, and ties

Each strategy is compared with `vector` on `ndcg_at_k`. A tie means the per-case score is identical.

| Strategy | Wins | Losses | Ties | Mean delta | 95% CI of delta | Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | 5 | 0 | 17 | 0.129 | 0.129 [0.021, 0.257] | 22 |
| rewrite_hybrid | 5 | 0 | 17 | 0.138 | 0.138 [0.031, 0.273] | 22 |
| rewrite_rrf | 5 | 2 | 15 | 0.076 | 0.076 [-0.079, 0.244] | 22 |
| rrf | 5 | 0 | 17 | 0.129 | 0.129 [0.021, 0.265] | 22 |

<details>
<summary>hybrid vs vector: per-case ndcg_at_k</summary>

| Case | Outcome | Strategy | Baseline |
| --- | --- | ---: | ---: |
| test-body-01 | tie | 1.000 | 1.000 |
| test-body-02 | win | 1.000 | 0.631 |
| test-cors-01 | tie | 1.000 | 1.000 |
| test-cors-02 | tie | 1.000 | 1.000 |
| test-dep-01 | tie | 1.000 | 1.000 |
| test-dep-02 | tie | 1.000 | 1.000 |
| test-deploy-01 | tie | 1.000 | 1.000 |
| test-deploy-02 | tie | 1.000 | 1.000 |
| test-error-01 | tie | 1.000 | 1.000 |
| test-error-02 | tie | 1.000 | 1.000 |
| test-jwt-01 | tie | 1.000 | 1.000 |
| test-jwt-02 | win | 1.000 | 0.631 |
| test-multi-01 | win | 0.587 | 0.494 |
| test-multi-02 | tie | 0.933 | 0.933 |
| test-path-01 | win | 1.000 | 0.000 |
| test-path-02 | tie | 1.000 | 1.000 |
| test-query-01 | win | 1.000 | 0.000 |
| test-query-02 | tie | 1.000 | 1.000 |
| test-response-01 | tie | 1.000 | 1.000 |
| test-response-02 | tie | 1.000 | 1.000 |
| test-security-01 | tie | 1.000 | 1.000 |
| test-security-02 | tie | 1.000 | 1.000 |

</details>

<details>
<summary>rewrite_hybrid vs vector: per-case ndcg_at_k</summary>

| Case | Outcome | Strategy | Baseline |
| --- | --- | ---: | ---: |
| test-body-01 | tie | 1.000 | 1.000 |
| test-body-02 | win | 1.000 | 0.631 |
| test-cors-01 | tie | 1.000 | 1.000 |
| test-cors-02 | tie | 1.000 | 1.000 |
| test-dep-01 | tie | 1.000 | 1.000 |
| test-dep-02 | tie | 1.000 | 1.000 |
| test-deploy-01 | tie | 1.000 | 1.000 |
| test-deploy-02 | tie | 1.000 | 1.000 |
| test-error-01 | tie | 1.000 | 1.000 |
| test-error-02 | tie | 1.000 | 1.000 |
| test-jwt-01 | tie | 1.000 | 1.000 |
| test-jwt-02 | win | 1.000 | 0.631 |
| test-multi-01 | win | 0.797 | 0.494 |
| test-multi-02 | tie | 0.933 | 0.933 |
| test-path-01 | win | 1.000 | 0.000 |
| test-path-02 | tie | 1.000 | 1.000 |
| test-query-01 | win | 1.000 | 0.000 |
| test-query-02 | tie | 1.000 | 1.000 |
| test-response-01 | tie | 1.000 | 1.000 |
| test-response-02 | tie | 1.000 | 1.000 |
| test-security-01 | tie | 1.000 | 1.000 |
| test-security-02 | tie | 1.000 | 1.000 |

</details>

<details>
<summary>rewrite_rrf vs vector: per-case ndcg_at_k</summary>

| Case | Outcome | Strategy | Baseline |
| --- | --- | ---: | ---: |
| test-body-01 | loss | 0.631 | 1.000 |
| test-body-02 | win | 1.000 | 0.631 |
| test-cors-01 | tie | 1.000 | 1.000 |
| test-cors-02 | tie | 1.000 | 1.000 |
| test-dep-01 | tie | 1.000 | 1.000 |
| test-dep-02 | tie | 1.000 | 1.000 |
| test-deploy-01 | loss | 0.000 | 1.000 |
| test-deploy-02 | tie | 1.000 | 1.000 |
| test-error-01 | tie | 1.000 | 1.000 |
| test-error-02 | tie | 1.000 | 1.000 |
| test-jwt-01 | tie | 1.000 | 1.000 |
| test-jwt-02 | win | 1.000 | 0.631 |
| test-multi-01 | win | 0.797 | 0.494 |
| test-multi-02 | tie | 0.933 | 0.933 |
| test-path-01 | win | 1.000 | 0.000 |
| test-path-02 | tie | 1.000 | 1.000 |
| test-query-01 | win | 1.000 | 0.000 |
| test-query-02 | tie | 1.000 | 1.000 |
| test-response-01 | tie | 1.000 | 1.000 |
| test-response-02 | tie | 1.000 | 1.000 |
| test-security-01 | tie | 1.000 | 1.000 |
| test-security-02 | tie | 1.000 | 1.000 |

</details>

<details>
<summary>rrf vs vector: per-case ndcg_at_k</summary>

| Case | Outcome | Strategy | Baseline |
| --- | --- | ---: | ---: |
| test-body-01 | tie | 1.000 | 1.000 |
| test-body-02 | win | 1.000 | 0.631 |
| test-cors-01 | tie | 1.000 | 1.000 |
| test-cors-02 | tie | 1.000 | 1.000 |
| test-dep-01 | tie | 1.000 | 1.000 |
| test-dep-02 | tie | 1.000 | 1.000 |
| test-deploy-01 | tie | 1.000 | 1.000 |
| test-deploy-02 | tie | 1.000 | 1.000 |
| test-error-01 | tie | 1.000 | 1.000 |
| test-error-02 | tie | 1.000 | 1.000 |
| test-jwt-01 | tie | 1.000 | 1.000 |
| test-jwt-02 | win | 1.000 | 0.631 |
| test-multi-01 | win | 0.587 | 0.494 |
| test-multi-02 | tie | 0.933 | 0.933 |
| test-path-01 | win | 1.000 | 0.000 |
| test-path-02 | tie | 1.000 | 1.000 |
| test-query-01 | win | 1.000 | 0.000 |
| test-query-02 | tie | 1.000 | 1.000 |
| test-response-01 | tie | 1.000 | 1.000 |
| test-response-02 | tie | 1.000 | 1.000 |
| test-security-01 | tie | 1.000 | 1.000 |
| test-security-02 | tie | 1.000 | 1.000 |

</details>

## Interpretation safeguards

- Retrieval metrics are document-level: a wider chunk candidate pool is deduplicated by source before scoring K documents.
- Citation alignment is an automatic source-match proxy, not a semantic entailment claim.
- Human-audit metrics remain blank until the generated audit JSONL is reviewed.
- Bootstrap intervals quantify uncertainty in this fixed benchmark, not generalization to arbitrary corpora.
- Known LLM cost covers only usage emitted by the configured provider; it is not an invoice.
