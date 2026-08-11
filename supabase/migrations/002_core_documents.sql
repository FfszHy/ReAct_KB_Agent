-- 002_core_documents.sql
-- Core knowledge-base tables: documents, document_chunks, chunk_embeddings.
--
-- NOTE on dimensions: the vector literal below uses 1536 to match the default
-- EMBEDDING_DIMENSIONS (DashScope qwen3.7-text-embedding). You MUST change every
-- `vector(1536)` occurrence (here and in 003/005) to match your embedding
-- model's real output dimension. Vectors from different embedding models must
-- never be compared.

-- ---------------------------------------------------------------------------
-- documents: an ingested source (file, url, or raw text).
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
    id            uuid primary key default gen_random_uuid(),
    user_id       text        not null default 'default',
    title         text        not null,
    source_uri    text,
    source_type   text        not null default 'text',  -- file | url | text
    content_hash  text,                                  -- sha256 of normalized content
    char_count    integer     not null default 0,
    chunk_count   integer     not null default 0,
    meta          jsonb       not null default '{}'::jsonb,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

create index if not exists documents_user_id_idx     on public.documents (user_id);
create index if not exists documents_content_hash_idx on public.documents (user_id, content_hash);
create index if not exists documents_created_at_idx   on public.documents (created_at desc);

-- ---------------------------------------------------------------------------
-- document_chunks: chunked text ready for retrieval.
--   fts is a generated tsvector kept in sync with content for full-text search.
-- ---------------------------------------------------------------------------
create table if not exists public.document_chunks (
    id            uuid primary key default gen_random_uuid(),
    document_id   uuid        not null references public.documents(id) on delete cascade,
    user_id       text        not null default 'default',
    chunk_index   integer     not null,
    content       text        not null,
    token_count   integer     not null default 0,
    meta          jsonb       not null default '{}'::jsonb,
    fts           tsvector generated always as (
        to_tsvector('simple', coalesce(content, ''))
    ) stored,
    created_at    timestamptz not null default now()
);

create index if not exists chunks_document_id_idx on public.document_chunks (document_id);
create index if not exists chunks_user_id_idx     on public.document_chunks (user_id);
create index if not exists chunks_fts_idx          on public.document_chunks using gin (fts);

-- ---------------------------------------------------------------------------
-- chunk_embeddings: one vector per chunk.
-- ---------------------------------------------------------------------------
create table if not exists public.chunk_embeddings (
    chunk_id     uuid        primary key references public.document_chunks(id) on delete cascade,
    embedding    vector(1536) not null,
    model        text        not null,
    dimensions   integer     not null default 1536,
    created_at   timestamptz not null default now()
);

-- IVFFlat index for approximate nearest neighbour (cosine distance <=>).
-- For small datasets a exact scan is fine; this index helps at scale.
create index if not exists chunk_embeddings_embedding_idx
    on public.chunk_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 100);
