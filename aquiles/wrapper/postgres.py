import asyncpg
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex
from aquiles.wrapper.basewrapper import BaseWrapper
from fastapi import HTTPException
import re

Pool = asyncpg.Pool
IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

class PostgreSQLRAG(BaseWrapper):
    def __init__(self, client: Pool):
        self.client = client

    async def create_index(self, q: CreateIndex):

        # I would assume this works, it can change after validation if something fails

        s = await self._safe_ident('public')
        t = await self._safe_ident('chunks')
        idx = await self._safe_ident(q.indexname)

        create_sql = f"""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {s}.{t} (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            resource_id uuid NOT NULL,
            name_chunk text,
            chunk_id integer,
            chunk_size integer,
            raw_text text,
            raw_text_tsv tsvector,
            embedding vector({q.embeddings_dim}) NOT NULL,
            embedding_model text[],
            metadata jsonb,
            created_at timestamptz DEFAULT now()
        );
        CREATE FUNCTION IF NOT EXISTS chunks_tsv_trigger() RETURNS trigger AS $$
        begin new.raw_text_tsv := to_tsvector('spanish', coalesce(new.raw_text,''));
        return new;
        end
        $$ LANGUAGE plpgsql;
        CREATE TRIGGER IF NOT EXISTS chunks_tsv_update
            BEFORE INSERT OR UPDATE ON {s}.{t}
            FOR EACH ROW EXECUTE PROCEDURE chunks_tsv_trigger();
        """

        async with self.client.acquire() as conn:
            try:
                await conn.execute(create_sql)

                exists = await conn.fetchval("SELECT to_regclass($1);", f"public.{q.indexname}")
                if exists and not q.delete_the_index_if_it_exists:
                    raise HTTPException(400, detail=f"Index public.{q.indexname} exists")
                if exists and q.delete_the_index_if_it_exists:
                    drop_sql = f"DROP INDEX {'CONCURRENTLY ' if q.concurrently else ''}IF EXISTS {s}.{idx};"
                    await conn.execute(drop_sql)

                create_idx_sql = (
                    f"CREATE INDEX {'CONCURRENTLY ' if q.concurrently else ''}IF NOT EXISTS {s}.{idx} "
                    f"ON {s}.{t} USING hnsw (embedding vector_cosine_ops) WITH (m = 16 , ef_construction = 200);"
                )

                await conn.execute(create_idx_sql)

                await conn.execute(f"SET hnsw.ef_search = 100;")

            except Exception as e:
                if "cannot run CREATE INDEX CONCURRENTLY inside a transaction block" in str(e):
                    raise HTTPException(500,
                                        detail="CREATE INDEX CONCURRENTLY cannot run inside a transaction. Run without concurrently or on a dedicated connection.")
                raise HTTPException(500, detail=str(e))

    async def _safe_ident(self, ident: str) -> str:
        if not IDENT_RE.match(ident):
            raise HTTPException(status_code=400, detail=f"Invalid identifier: {ident}")
        return f'"{ident}"'

    async def send(self, q: SendRAG):
        return await super().send(q)

    async def query(self, q: QueryRAG, emb_vector):
        return await super().query(q, emb_vector)

    async def drop_index(self, q: DropIndex):
        return await super().drop_index(q)
        
    async def get_ind(self):
        return await super().get_ind()

    async def ready(self):
        return await super().ready()