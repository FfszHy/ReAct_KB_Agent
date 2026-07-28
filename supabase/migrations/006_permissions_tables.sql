-- 006_permissions_tables.sql
-- Runtime tool-permission overrides. YAML provides the baseline fallback;
-- a row in this table replaces the YAML rule for that exact tool.

create table if not exists public.tool_permissions (
    id          uuid primary key default gen_random_uuid(),
    tool_name   text        not null unique,
    permission  text        not null default 'allow',  -- allow | ask | deny
    constraints jsonb       not null default '{}'::jsonb,
    updated_at  timestamptz not null default now()
);

create index if not exists tool_permissions_name_idx on public.tool_permissions (tool_name);
