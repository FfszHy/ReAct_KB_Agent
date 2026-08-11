# Approved-web acceptance benchmark

This is a four-case operational acceptance suite, separate from the
KB-only RAG leaderboard. Each case names one version-pinned raw GitHub source
and requires `web_fetch`; the caller must provide the exact approved host on
the command line.

The suite deliberately does not use `rag_search`, `rag_read`, or `web_search`.
It measures the bounded workflow “a user approved this exact documentation
host, then the Agent fetched and cited it.” The raw GitHub URLs point at pinned
release tags already represented in the multidomain corpus manifest. External
content and transport can still vary, so treat it as an operational acceptance
check and complete the generated human-audit template before making semantic
quality claims.

It reuses the existing `eval-tech-multidomain-v1` corpus scope; no extra corpus
ingestion is necessary.

```bash
SUPABASE_TIMEOUT=90 pkb-agent eval run data/evals/tech-web-approved-v1 \
  --split test \
  --user eval-tech-multidomain-v1 \
  --strategies "" \
  --with-agent \
  --agent-profile web_approved \
  --web-allow-host raw.githubusercontent.com \
  --repetitions 1
```
