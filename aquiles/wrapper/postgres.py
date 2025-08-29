import asyncpg
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex
from aquiles.wrapper.basewrapper import BaseWrapper
from fastapi import HTTPException
import re
import json
from uuid import uuid4

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
        if not IDENT_RE.match(q.index):
            raise HTTPException(400, detail="Invalid index")

        table_unquoted = _table_name_for_index(q.index)
        t = _validate_ident(table_unquoted)

        chosen_id = uuid4()
        embedding_model_val = None
        try:
            val = getattr(q, "embedding_model", None)
            embedding_model_val = None if val is None else str(val).strip()
        except Exception:
            embedding_model_val = None
        embedding_model_val = embedding_model_val or "__unknown__"

        vector = getattr(q, "embeddings", None)
        if vector is None:
            raise HTTPException(400, detail="No vector provided in q.embeddings")

        vec_literal = _serialize_vector(vector)  # like "[0.1,0.2,...]"

        insert_sql = f"""
        INSERT INTO public.{t} (id, resource_id, name_chunk, chunk_id, chunk_size, raw_text, embedding, embedding_model, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9)
        RETURNING id;
        """

        async with self.client.acquire() as conn:
            try:
                metadata_val = getattr(q, "metadata", None)
                if isinstance(metadata_val, dict):
                    metadata_val = json.dumps(metadata_val)
                row = await conn.fetchrow(
                    insert_sql,
                    str(chosen_id),
                    getattr(q, "resource_id", None),
                    getattr(q, "name_chunk", None),
                    str(getattr(q, "chunk_id", chosen_id)),
                    getattr(q, "chunk_size", None),
                    getattr(q, "raw_text", None),
                    vec_literal,  # passed as text, casted to vector by $7::vector
                    embedding_model_val,
                    metadata_val
                )
                return str(row['id'])
            except Exception as e:
                raise HTTPException(500, detail=f"Error inserting point: {e}")

    async def query(self, q: QueryRAG, emb_vector):
        if not IDENT_RE.match(q.index):
            raise HTTPException(400, detail="Invalid index")
        table_unquoted = _table_name_for_index(q.index)
        t = _validate_ident(table_unquoted)

        model_val = getattr(q, "embedding_model", None)
        where_clause = ""
        params = []

        if model_val:
            model_val = str(model_val).strip()
            if model_val:
                where_clause = "WHERE embedding_model = $2"
                params.append(model_val)

        ef_runtime = getattr(q, "ef_runtime", None)
        top_k = int(getattr(q, "top_k", 5))

        vec_literal = _serialize_vector(emb_vector)


        query_sql = f"""
        SELECT name_chunk, chunk_id, chunk_size, raw_text, metadata,
               (embedding <=> $1::vector) AS distance
        FROM public.{t}
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $3;
        """

        async with self.client.acquire() as conn:
            try:
                if ef_runtime is not None:
                    await conn.execute(f"SET hnsw.ef_search = {int(ef_runtime)};")

                if where_clause:
                    rows = await conn.fetch(query_sql, vec_literal, params[0], top_k)
                else:
                    rows = await conn.fetch(query_sql, vec_literal, top_k)

                results = []
                for r in rows:
                    dist = r['distance']
                    similarity = None
                    try:
                        similarity = 1.0 - float(dist) if dist is not None else None
                    except Exception:
                        similarity = None

                    results.append({
                        "name_chunk": r['name_chunk'],
                        "chunk_id": str(r['chunk_id']) if r['chunk_id'] is not None else None,
                        "chunk_size": int(r['chunk_size']) if r['chunk_size'] is not None else None,
                        "raw_text": r['raw_text'],
                        "score": similarity,
                        "embedding_model": r['embedding_model'],
                    })


                if getattr(q, "cosine_distance_threshold", None) is not None:
                    try:
                        dist_thr = float(q.cosine_distance_threshold)
                        filtered = []
                        for r in results:
                            s = r.get("score")
                            if s is None:
                                continue
                            distance_like = 1.0 - s 
                            if distance_like <= dist_thr:
                                filtered.append(r)
                        results = filtered
                    except Exception:
                        pass

                results = results[: top_k]
                return results

            except Exception as e:
                raise HTTPException(500, detail=f"Search error: {e}")

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