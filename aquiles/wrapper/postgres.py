import asyncpg
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex
from aquiles.wrapper.basewrapper import BaseWrapper
from fastapi import HTTPException
import re

Pool = asyncpg.Pool
IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _validate_ident(name: str):
    if not IDENT_RE.match(name):
        raise HTTPException(status_code=400, detail=f"Invalid identifier: {name}")
    return f'"{name}"'

def _table_name_for_index(indexname: str) -> str:
    # table per collection approach
    return f"chunks__{indexname}"

def _serialize_vector(vec) -> str:
    # pgvector accepts literal of form '[0.1,0.2,...]'::vector
    return "[" + ",".join(map(str, vec)) + "]"

class PostgreSQLRAG(BaseWrapper):
    def __init__(self, client: Pool):
        self.client = client

    async def create_index(self, q: CreateIndex):
        if not IDENT_RE.match(q.indexname):
            raise HTTPException(400, detail="Invalid indexname")

        table_unquoted = _table_name_for_index(q.indexname)
        t = _validate_ident(table_unquoted)
        idx = _validate_ident(q.indexname + "_embedding_hnsw")

        create_sql = f"""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS public.{t} (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            resource_id uuid,
            name_chunk text,
            chunk_id uuid,
            chunk_size integer,
            raw_text text,
            raw_text_tsv tsvector,
            embedding vector({int(q.embeddings_dim)}) NOT NULL,
            embedding_model text,   -- single model string, like qdrant payload
            metadata jsonb,
            created_at timestamptz DEFAULT now()
        );
        CREATE FUNCTION IF NOT EXISTS chunks_tsv_trigger() RETURNS trigger AS $$
        begin
          new.raw_text_tsv := to_tsvector('spanish', coalesce(new.raw_text,''));
          return new;
        end
        $$ LANGUAGE plpgsql;
        -- create trigger only if not exists (psql doesn't have CREATE TRIGGER IF NOT EXISTS,
        -- so we try/catch by checking pg_trigger)
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname = 'chunks_tsv_update'
              AND tgrelid = (quote_ident('public') || '.' || quote_ident($1))::regclass
          ) THEN
            EXECUTE format('CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE ON public.%s FOR EACH ROW EXECUTE PROCEDURE chunks_tsv_trigger();', $1);
          END IF;
        END
        $$ LANGUAGE plpgsql;
        """

        m = getattr(q, "m", 16)
        ef_construct = getattr(q, "ef_construction", 200)
        concurrently = getattr(q, "concurrently", False)

        create_idx_sql = (
            f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}IF NOT EXISTS public.{idx} "
            f"ON public.{t} USING hnsw (embedding vector_cosine_ops) WITH (m = {int(m)}, ef_construction = {int(ef_construct)});"
        )

        async with self.client.acquire() as conn:
            try:
                create_sql_sub = create_sql.replace("$1", table_unquoted)

                await conn.execute(create_sql_sub)
                regclass = await conn.fetchval("SELECT to_regclass($1);", f"public.{q.indexname}_embedding_hnsw")
                if regclass and not q.delete_the_index_if_it_exists:
                    raise HTTPException(400, detail=f"Index public.{q.indexname}_embedding_hnsw exists")
                if regclass and q.delete_the_index_if_it_exists:
                    drop_sql = f"DROP INDEX {'CONCURRENTLY ' if concurrently else ''}IF EXISTS public.{idx};"
                    await conn.execute(drop_sql)

                try:
                    await conn.execute(create_idx_sql)
                except Exception as e:
                    if concurrently and "cannot run CREATE INDEX CONCURRENTLY inside a transaction block" in str(e):
                        raise HTTPException(500, detail=("CREATE INDEX CONCURRENTLY cannot run inside a transaction block. "
                                                        "Run with concurrently=False or execute the CONCURRENTLY statement on a dedicated connection."))
                    raise
                ef_runtime = getattr(q, "ef_runtime", 100)
                await conn.execute(f"SET hnsw.ef_search = {int(ef_runtime)};")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, detail=str(e))

    async def send(self, q: SendRAG):
        return await super().send(q)

    async def query(self, q: QueryRAG, emb_vector):
        return await super().query(q, emb_vector)

    async def drop_index(self, q: DropIndex):
        if not IDENT_RE.match(q.index_name):
            raise HTTPException(400, detail="Invalid index_name")

        table_unquoted = _table_name_for_index(q.index_name)
        t = _validate_ident(table_unquoted)
        idx = _validate_ident(f"{q.index_name}_embedding_hnsw")

        async with self.client.acquire() as conn:
            try:
                if q.delete_docs:
                    await conn.execute(f"DROP TABLE IF EXISTS public.{t} CASCADE;")
                    return {"status": "dropped_table", "drop-index": q.index_name}
                else:
                    await conn.execute(f"DROP INDEX IF EXISTS public.{idx};")
                    return {"status": "dropped_index", "drop-index": q.index_name}
            except Exception as e:
                raise HTTPException(500, detail=str(e))
        
    async def get_ind(self):
        return await super().get_ind()

    async def ready(self):
        async with self.client.acquire() as conn:
            try:
                await conn.fetchval("SELECT 1;")
                return True
            except Exception:
                return False