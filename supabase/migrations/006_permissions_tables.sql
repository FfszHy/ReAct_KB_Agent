-- 006_permissions_tables.sql
-- Optional runtime tool-permission overrides (the YAML policy is the primary
-- source of truth; this table allows dynamic overrides stored in DB).

create table if not exists public.tool_permissions (
    id          uuid primary key default gen_random_uuid(),
    tool_name   text        not null unique,
    permission  text        not null default 'allow',  -- allow | ask | deny
    constraints jsonb       not null default '{}'::jsonb,
    updated_at  timestamptz not null default now()
);

create index if not exists tool_permissions_name_idx on public.tool_permissions (tool_name);
