-- Canonical structured answer and verification audit for every completed run.
-- Citation metadata remains materialized here even when the source page later
-- changes or expires.

alter table public.agent_runs
    add column if not exists answer_payload jsonb,
    add column if not exists verification jsonb not null default '{}'::jsonb;
