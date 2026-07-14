-- 001_extensions.sql
-- Enable required Postgres extensions.
-- pgvector provides the vector(N) type and distance operators (<=>, <#>, <->).

create extension if not exists "vector";
create extension if not exists "pg_trgm";   -- trigram similarity (optional helper)
create extension if not exists "unaccent";   -- accent-insensitive FTS
