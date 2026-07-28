# PKB-Agent

A Supabase-backed **ReAct** personal knowledge base agent. The agent runtime
reasons in a Thought → Action → Observation loop and interacts with the world
**only** through a permissioned, traced tool layer. There is no direct database
or network access from the runtime itself.

## Tech stack

| Layer | Choice |
|------|--------|
| Language | Python 3.12 |
| LLM | DeepSeek Chat Completions (native `tools` calling), model `deepseek-v4-flash` |
| Storage | Supabase Postgres |
| Vector | `pgvector` (similarity via `<=>`, `<#>`, `<->`) |
| Lexical | Postgres Full Text Search (`tsvector`, `ts_rank`) |
| DB client | `supabase-py` (`rpc()` for search functions) |
| Embeddings | External OpenAI-compatible `/embeddings` endpoint (configurable) |
| Web search | Tavily / Serper / Bing (selectable via `WEB_SEARCH_PROVIDER`) |
| CLI | Typer (no web UI in v1) |

## Architecture

```
CLI User → Typer CLI → DeepSeek ReAct Runtime → Tool Registry + Permission Manager
                                                              ↓
   Tools: rag_search, rag_read, web_search, web_fetch,
          memory_search, memory_write, calculator, now
                                                              ↓
                         Repositories → Supabase Postgres
   (documents, document_chunks, chunk_embeddings, agent_runs,
    agent_steps, tool_calls, task_memory, tool_permissions)
```

**Core invariant:** the Agent Runtime never touches Supabase, the network, or
memory directly. It can only call tools. Each tool call passes through:
parameter validation → permission check → trace recording → execution → result
truncation → error wrapping.

### Prompt lifecycle

Prompt bodies live in `config/prompts/*.md`; their runtime lifecycle is declared
in `config/prompts/manifest.yaml` rather than inferred from filenames.

- `system_react` is included once per run.
- `memory_policy` is included only when `memory_write` is registered; secrets
  are additionally rejected by the code-side memory policy.
- `query_rewrite` runs only immediately before `rag_search` or `web_search`.
  It returns focused query variants, falls back to the original query on failure,
  and records the prompt hash plus original/effective tool arguments in traces.

Set `prompts.query_rewrite.enabled: false` in `config/default.yaml` to avoid
the extra planning model call. Prompt and rewrite provenance requires migration
`008_prompt_trace.sql` in addition to the earlier Supabase migrations.

### Dynamic tool permissions

`config/permissions.yaml` is the baseline policy. A row in the global
`tool_permissions` table replaces the complete rule (`permission` and
`constraints`) for that exact tool; no row means the YAML rule remains active.
By default the runtime refreshes this table before every tool call, so a policy
edit applies during a long-lived run. Configure `permissions.refresh_seconds`
to trade freshness for fewer reads, or set `permissions.db_overrides_enabled`
to `false` for YAML-only operation. If the table cannot be read, the runtime
clears dynamic rules and safely falls back to YAML.

## Project layout

See the spec tree under `src/pkb_agent/`. Highlights:

- `agent/` — ReAct runtime, loop state, errors
- `llm/` — DeepSeek client + tool-call schemas
- `tools/` — base/registry/permissions/result + `builtin/` implementations
- `rag/` — chunking, embeddings, ingestion, retriever, ranking, citations
- `memory/` — policy + manager
- `trace/` — recorder + secret redaction
- `storage/` — supabase client + repositories
- `web/` — search provider, fetcher, sanitizer
- `security/` — URL safety, schemas, secrets

## Setup

### 1. Environment

```bash
conda create -n pkb-agent python=3.12 -y
conda activate pkb-agent
pip install -e ".[dev]"
```

### 2. Configuration

```bash
cp .env.example .env
# fill in DEEPSEEK_API_KEY, SUPABASE_* , EMBEDDING_*, WEB_SEARCH_API_KEY
```

**Important:** `EMBEDDING_DIMENSIONS` MUST equal your embedding model's real
output dimension. The `vector(n)` column in Supabase is created with this value;
vectors from different embedding models must not be mixed.

### 3. Database

Apply the migrations under `supabase/migrations/` to your Supabase project (via
the Supabase dashboard, `supabase db push`, or any SQL client). The migrations
enable `vector`, create core tables, search functions, trace tables, memory
tables, permission tables, and RLS policies.

### 4. Ingest documents

```bash
pkb-agent ingest ./my-notes.md
pkb-agent ingest ./papers/ --recursive
```

### 5. Ask

```bash
pkb-agent ask "Summarize the auth design decisions in my notes"
```

## Embedding model

The embedding client speaks the OpenAI-compatible `/embeddings` protocol, so it
works with DashScope (default: `text-embedding-v4` / 1536 dims), OpenAI,
SiliconFlow, or any local server. Set `EMBEDDING_API_BASE_URL`,
`EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS` accordingly.

## License

MIT
