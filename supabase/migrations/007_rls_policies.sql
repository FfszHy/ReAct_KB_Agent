-- 007_rls_policies.sql
-- Row Level Security policies.
--
-- The agent typically uses the service-role key, which bypasses RLS. These
-- policies guard access when the anon key (or a JWT-bearing client) is used:
-- a user may only read/write rows where row.user_id matches the JWT sub, or
-- where user_id equals the default 'default' user (single-user personal mode).

-- Helper: resolve the effective user id from the current JWT, falling back to
-- 'default' when there is no authenticated principal.
create or replace function public.app_user_id()
returns text
language sql
stable
as $$
    coalesce(nullif(current_setting('request.jwt.claims', true)::jsonb->>'sub', ''), 'default');
$$;

-- ---- documents ----
alter table public.documents enable row level security;
drop policy if exists documents_select_own on public.documents;
create policy documents_select_own on public.documents
    for select using (user_id = public.app_user_id() or user_id = 'default');
drop policy if exists documents_modify_own on public.documents;
create policy documents_modify_own on public.documents
    for all using (user_id = public.app_user_id() or user_id = 'default')
    with check (user_id = public.app_user_id() or user_id = 'default');

-- ---- document_chunks ----
alter table public.document_chunks enable row level security;
drop policy if exists chunks_own on public.document_chunks;
create policy chunks_own on public.document_chunks
    for all using (user_id = public.app_user_id() or user_id = 'default')
    with check (user_id = public.app_user_id() or user_id = 'default');

-- ---- chunk_embeddings ----
alter table public.chunk_embeddings enable row level security;
drop policy if exists embeddings_via_chunks on public.chunk_embeddings;
create policy embeddings_via_chunks on public.chunk_embeddings
    for all using (
        exists (
            select 1 from public.document_chunks dc
            where dc.id = chunk_embeddings.chunk_id
              and (dc.user_id = public.app_user_id() or dc.user_id = 'default')
        )
    );

-- ---- agent_runs ----
alter table public.agent_runs enable row level security;
drop policy if exists runs_own on public.agent_runs;
create policy runs_own on public.agent_runs
    for all using (user_id = public.app_user_id() or user_id = 'default')
    with check (user_id = public.app_user_id() or user_id = 'default');

-- ---- agent_steps ----
alter table public.agent_steps enable row level security;
drop policy if exists steps_via_runs on public.agent_steps;
create policy steps_via_runs on public.agent_steps
    for all using (
        exists (
            select 1 from public.agent_runs r
            where r.id = agent_steps.run_id
              and (r.user_id = public.app_user_id() or r.user_id = 'default')
        )
    );

-- ---- tool_calls ----
alter table public.tool_calls enable row level security;
drop policy if exists toolcalls_via_runs on public.tool_calls;
create policy toolcalls_via_runs on public.tool_calls
    for all using (
        exists (
            select 1 from public.agent_runs r
            where r.id = tool_calls.run_id
              and (r.user_id = public.app_user_id() or r.user_id = 'default')
        )
    );

-- ---- task_memory ----
alter table public.task_memory enable row level security;
drop policy if exists memory_own on public.task_memory;
create policy memory_own on public.task_memory
    for all using (user_id = public.app_user_id() or user_id = 'default')
    with check (user_id = public.app_user_id() or user_id = 'default');

-- ---- tool_permissions ----
-- Permissions are global config: readable by all, writable only by service role
-- (service role bypasses RLS, so no write policy needed for regular clients).
alter table public.tool_permissions enable row level security;
drop policy if exists permissions_read on public.tool_permissions;
create policy permissions_read on public.tool_permissions for select using (true);
