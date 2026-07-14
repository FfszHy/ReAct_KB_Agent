-- 003_search_functions.sql
-- Postgres functions wrapping vector + full-text retrieval, invoked via the
-- Supabase client's rpc(). Keeping the queries in DB functions is the
-- Supabase-recommended pattern for vector similarity search.
--
-- Hybrid fusion (RRF) is performed client-side in Python (rag/ranking.py),
-- so this file exposes the two constituent retrievers separately.

-- Normalize an embedding list literal into vector(1536).
-- (Change 1536 here too if you change the embedding dimension.)

-- ---------------------------------------------------------------------------
-- Vector (semantic) search over chunk_embeddings.
-- Returns chunks ordered by cosine distance (most similar first).
-- ---------------------------------------------------------------------------
create or replace function public.rag_vector_search(
    p_embedding  vector(1536),
    p_match_count integer default 6,
    p_user_id     text    default null
) returns table (
    chunk_id      uuid,
    document_id   uuid,
    user_id       text,
    chunk_index   integer,
    content       text,
    chunk_meta    jsonb,
    doc_title     text,
    source_uri    text,
    vector_score  float
)
language sql stable
as $$
    select
        dc.id            as chunk_id,
        dc.document_id   as document_id,
        dc.user_id       as user_id,
        dc.chunk_index   as chunk_index,
        dc.content       as content,
        dc.meta          as chunk_meta,
        d.title          as doc_title,
        d.source_uri     as source_uri,
        (1 - (ce.embedding <=> p_embedding))::float as vector_score
    from public.chunk_embeddings ce
    join public.document_chunks dc on dc.id = ce.chunk_id
    join public.documents d         on d.id = dc.document_id
    where (p_user_id is null or dc.user_id = p_user_id)
    order by ce.embedding <=> p_embedding
    limit p_match_count;
$$;

-- ---------------------------------------------------------------------------
-- Full-text search over document_chunks.
-- Uses websearch_to_tsquery for a forgiving, Google-like query syntax.
-- ---------------------------------------------------------------------------
create or replace function public.rag_fts_search(
    p_query       text,
    p_match_count integer default 6,
    p_user_id     text    default null
) returns table (
    chunk_id      uuid,
    document_id   uuid,
    user_id       text,
    chunk_index   integer,
    content       text,
    chunk_meta    jsonb,
    doc_title     text,
    source_uri    text,
    fts_score     float
)
language sql stable
as $$
    select
        dc.id            as chunk_id,
        dc.document_id   as document_id,
        dc.user_id       as user_id,
        dc.chunk_index   as chunk_index,
        dc.content       as content,
        dc.meta          as chunk_meta,
        d.title          as doc_title,
        d.source_uri     as source_uri,
        ts_rank(dc.fts, websearch_to_tsquery('simple', p_query))::float as fts_score
    from public.document_chunks dc
    join public.documents d on d.id = dc.document_id
    where dc.fts @@ websearch_to_tsquery('simple', p_query)
      and (p_user_id is null or dc.user_id = p_user_id)
    order by fts_score desc
    limit p_match_count;
$$;

-- ---------------------------------------------------------------------------
-- Optional DB-side hybrid fusion (weighted score). Kept for convenience; the
-- agent runtime uses client-side RRF by default (see rag/ranking.py).
-- ---------------------------------------------------------------------------
create or replace function public.rag_hybrid_search(
    p_query         text,
    p_embedding     vector(1536),
    p_match_count   integer default 6,
    p_user_id       text    default null,
    p_vector_weight float   default 0.6,
    p_fts_weight    float   default 0.4
) returns table (
    chunk_id       uuid,
    document_id    uuid,
    chunk_index    integer,
    content        text,
    doc_title      text,
    source_uri     text,
    vector_score   float,
    fts_score      float,
    combined_score float
)
language sql stable
as $$
    with vec as (
        select chunk_id, vector_score
        from public.rag_vector_search(p_embedding, p_match_count * 4, p_user_id)
    ),
    fts as (
        select chunk_id, fts_score
        from public.rag_fts_search(p_query, p_match_count * 4, p_user_id)
    )
    select
        dc.id            as chunk_id,
        dc.document_id   as document_id,
        dc.chunk_index   as chunk_index,
        dc.content       as content,
        d.title          as doc_title,
        d.source_uri     as source_uri,
        coalesce(vec.vector_score, 0) as vector_score,
        coalesce(fts.fts_score, 0)    as fts_score,
        (coalesce(vec.vector_score, 0) * p_vector_weight
         + coalesce(fts.fts_score, 0) * p_fts_weight)::float as combined_score
    from public.document_chunks dc
    join public.documents d on d.id = dc.document_id
    left join vec on vec.chunk_id = dc.id
    left join fts on fts.chunk_id = dc.id
    where vec.chunk_id is not null or fts.chunk_id is not null
    order by combined_score desc
    limit p_match_count;
$$;
