-- 004_trace_tables.sql
-- Agent execution trace: runs, steps, and individual tool calls.

-- ---------------------------------------------------------------------------
-- agent_runs: one row per agent invocation.
-- ---------------------------------------------------------------------------
create table if not exists public.agent_runs (
    id            uuid primary key,
    user_id       text        not null default 'default',
    question      text        not null,
    status        text        not null default 'running',  -- running|finished|error|max_steps|aborted
    final_answer  text,
    error         text,
    step_count    integer     not null default 0,
    usage         jsonb       not null default '{}'::jsonb,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

create index if not exists agent_runs_user_id_idx    on public.agent_runs (user_id);
create index if not exists agent_runs_created_at_idx on public.agent_runs (created_at desc);

-- ---------------------------------------------------------------------------
-- agent_steps: one row per ReAct step (Thought -> Action -> Observation).
-- ---------------------------------------------------------------------------
create table if not exists public.agent_steps (
    id          uuid primary key default gen_random_uuid(),
    run_id      uuid        not null references public.agent_runs(id) on delete cascade,
    step_index  integer     not null,
    thought     text,
    tool_name   text,
    tool_args   jsonb       not null default '{}'::jsonb,
    observation text,
    status      text        not null default 'pending',  -- pending|ok|error|skipped
    error       text,
    started_at  timestamptz not null default now(),
    ended_at    timestamptz
);

create index if not exists agent_steps_run_id_idx on public.agent_steps (run_id, step_index);

-- ---------------------------------------------------------------------------
-- tool_calls: granular per-tool-call records (args, result, timing).
-- ---------------------------------------------------------------------------
create table if not exists public.tool_calls (
    id           uuid primary key default gen_random_uuid(),
    run_id       uuid        not null references public.agent_runs(id) on delete cascade,
    step_index   integer     not null,
    tool_name    text        not null,
    arguments    jsonb       not null default '{}'::jsonb,
    result       jsonb,
    ok           boolean     not null default true,
    truncated    boolean     not null default false,
    duration_ms  integer,
    error        text,
    created_at   timestamptz not null default now()
);

create index if not exists tool_calls_run_id_idx on public.tool_calls (run_id, step_index);
create index if not exists tool_calls_tool_idx   on public.tool_calls (tool_name);
