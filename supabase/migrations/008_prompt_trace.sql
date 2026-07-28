-- Prompt provenance and query-rewrite provenance for reproducible agent traces.

alter table public.agent_runs
    add column if not exists prompt_context jsonb not null default '{}'::jsonb;

alter table public.agent_steps
    add column if not exists original_tool_args jsonb,
    add column if not exists prompt_context jsonb not null default '{}'::jsonb;

alter table public.tool_calls
    add column if not exists original_arguments jsonb,
    add column if not exists prompt_context jsonb not null default '{}'::jsonb;
