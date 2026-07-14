-- 005_memory_tables.sql
-- Persistent agent memory: short-term (per-run) and long-term (cross-run) notes.
-- Each note may carry an embedding for semantic recall.

create table if not exists public.task_memory (
    id          uuid primary key default gen_random_uuid(),
    user_id     text        not null default 'default',
    run_id      uuid,  -- nullable: long-term memory is not tied to a run
    scope       text        not null default 'long',   -- short | long
    kind        text        not null default 'fact',   -- preference|fact|decision|reference|...
    content     text        not null,
    embedding   vector(1536),
    meta        jsonb       not null default '{}'::jsonb,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists task_memory_user_id_idx  on public.task_memory (user_id);
create index if not exists task_memory_scope_idx    on public.task_memory (user_id, scope);
create index if not exists task_memory_embedding_idx
    on public.task_memory using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ---------------------------------------------------------------------------
-- Semantic search over memory notes.
-- ---------------------------------------------------------------------------
create or replace function public.memory_vector_search(
    p_embedding   vector(1536),
    p_match_count integer default 5,
    p_user_id     text    default null,
    p_scope       text    default null
) returns table (
    id         uuid,
    user_id    text,
    run_id     uuid,
    scope      text,
    kind       text,
    content    text,
    meta       jsonb,
    score      float,
    created_at timestamptz
)
language sql stable
as $$
    select
        tm.id,
        tm.user_id,
        tm.run_id,
        tm.scope,
        tm.kind,
        tm.content,
        tm.meta,
        (1 - (tm.embedding <=> p_embedding))::float as score,
        tm.created_at
    from public.task_memory tm
    where tm.embedding is not null
      and (p_user_id is null or tm.user_id = p_user_id)
      and (p_scope is null or tm.scope = p_scope)
    order by tm.embedding <=> p_embedding
    limit p_match_count;
$$;
