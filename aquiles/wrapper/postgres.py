import asyncpg
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex, allow_metadata
from aquiles.wrapper.basewrapper import BaseWrapper
from typing import Any
from fastapi import HTTPException
import re
import json
from uuid import uuid4
import logging
from datetime import datetime
from datetime import timezone

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

def _parse_to_datetime(val) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        dt = val
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(int(val), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        try:
            s = val
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            try:
                return datetime.fromtimestamp(int(val), tz=timezone.utc)
            except Exception:
                return None
    return None

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
            embedding_model text,
            metadata jsonb, -- I think I can save the metadata here like in Qdrant and Redis
            created_at timestamptz DEFAULT now()
        );

        -- USE CREATE OR REPLACE FUNCTION (NO IF NOT EXISTS)
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        begin
          new.raw_text_tsv := to_tsvector('spanish', coalesce(new.raw_text,''));
          return new;
        end
        $$ LANGUAGE plpgsql;

        -- crear trigger solo si no existe
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname = 'chunks_tsv_update'
              AND tgrelid = (quote_ident('public') || '.' || quote_ident($1))::regclass
          ) THEN
            EXECUTE format(
              'CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE ON public.%s FOR EACH ROW EXECUTE PROCEDURE chunks_tsv_trigger();',
              $1
            );
          END IF;
        END
        $$ LANGUAGE plpgsql;
        """

        m = getattr(q, "m", 16)
        ef_construct = getattr(q, "ef_construction", 200)
        concurrently = getattr(q, "concurrently", False)

        create_idx_sql = (
            f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}{idx} "
            f"ON public.{t} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {int(m)}, ef_construction = {int(ef_construct)});"
        )

        async with self.client.acquire() as conn:
            try:
                create_sql_sub = create_sql.replace("$1", f"'{table_unquoted}'")
                logging.info("create_sql_sub:\n%s", create_sql_sub)
                logging.info("create_idx_sql:\n%s", create_idx_sql)

                await conn.execute(create_sql_sub)

                # chequeo del índice existente (ya lo tenías)
                regclass = await conn.fetchval(
                    "SELECT to_regclass($1);",
                    f"public.{q.indexname}_embedding_hnsw"
                )
                if regclass and not q.delete_the_index_if_it_exists:
                    raise HTTPException(400, detail=f"Index public.{q.indexname}_embedding_hnsw exists")
                if regclass and q.delete_the_index_if_it_exists:
                    drop_sql = f"DROP INDEX {'CONCURRENTLY ' if concurrently else ''}IF EXISTS {idx};"
                    await conn.execute(drop_sql)

                try:
                    if concurrently:
                        async with self.client.acquire() as idx_conn:
                            await idx_conn.execute(create_idx_sql)
                    else:
                        await conn.execute(create_idx_sql)
                except Exception as e:
                    if concurrently and "cannot run CREATE INDEX CONCURRENTLY inside a transaction block" in str(e):
                        raise HTTPException(500, detail=(f"Error:{e},""CREATE INDEX CONCURRENTLY cannot run inside a transaction block. "
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

        metadata_val = getattr(q, "metadata", None)
        created_at_val = None
        if isinstance(metadata_val, dict):
            md = dict(metadata_val)
            if "created_at" in md:
                dt = _parse_to_datetime(md.get("created_at"))
                if dt:
                    created_at_val = dt
                md.pop("created_at", None)
            try:
                metadata_json = json.dumps(md)
            except Exception:
                metadata_json = None
        else:
            metadata_json = None


        insert_sql = f"""
        INSERT INTO public.{t}
            (id, resource_id, name_chunk, chunk_id, chunk_size, raw_text, embedding, embedding_model, metadata{', created_at' if True else ''})
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9{', $10' if True else None})
        RETURNING id;
        """

        async with self.client.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    insert_sql,
                    str(chosen_id),
                    getattr(q, "resource_id", None),
                    getattr(q, "name_chunk", None),
                    str(getattr(q, "chunk_id", chosen_id)),
                    getattr(q, "chunk_size", None),
                    getattr(q, "raw_text", None),
                    vec_literal,
                    embedding_model_val,
                    metadata_json,
                    created_at_val
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
        ef_runtime = getattr(q, "ef_runtime", None)
        top_k = int(getattr(q, "top_k", 5))

        vec_literal = _serialize_vector(emb_vector)  # like "[0.1,0.2,...]"

        params: list[Any] = [vec_literal]
        next_idx = 2

        where_clauses: list[str] = []
        if model_val:
            model_val = str(model_val).strip()
            if model_val:
                where_clauses.append(f"embedding_model = ${next_idx}::text")
                params.append(model_val)
                next_idx += 1

        metadata_in = getattr(q, "metadata", None)
        if isinstance(metadata_in, dict):
            for key, val in metadata_in.items():
                # solo permitimos keys en allow_metadata
                if key not in allow_metadata:
                    continue
                if val is None:
                    # SKIP None => no filtrar por esta key (comportamiento "None = ignorar")
                    continue

                # created_at: preferimos filtrar por la columna created_at si existe
                if key == "created_at":
                    # support dict with min/max or scalar
                    if isinstance(val, dict):
                        # admite min/max/gte/lte
                        minv = val.get("min") or val.get("gte")
                        maxv = val.get("max") or val.get("lte")
                        if minv is not None:
                            dt_min = _parse_to_datetime(minv)
                            if dt_min:
                                where_clauses.append(f"created_at >= ${next_idx}::timestamptz")
                                params.append(dt_min)
                                next_idx += 1
                        if maxv is not None:
                            dt_max = _parse_to_datetime(maxv)
                            if dt_max:
                                where_clauses.append(f"created_at <= ${next_idx}::timestamptz")
                                params.append(dt_max)
                                next_idx += 1
                    else:
                        # single value -> equality (or date >= value)
                        dt = _parse_to_datetime(val)
                        if dt:
                            where_clauses.append(f"created_at >= ${next_idx}::timestamptz")
                            params.append(dt)
                            next_idx += 1
                    continue

                # si es lista/tuple -> buscamos ANY dentro del array JSONB (OR dentro del campo)
                if isinstance(val, (list, tuple, set)):
                    arr = [str(x) for x in val if x is not None]
                    if not arr:
                        continue
                # usamos jsonb_array_elements_text para comprobar si algún elemento coincide
                # evitamos interpolation de key usando el nombre literal (key está validado en allow_metadata)
                # generamos una subquery EXISTS con un parámetro (text[])
                    where_clauses.append(
                        f"EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'{key}') elem WHERE elem = ANY(${next_idx}::text[]))"
                    )
                    params.append(arr)
                    next_idx += 1
                    continue

                # si es dict para otros campos (por ejemplo numeric ranges) - intentamos rango
                if isinstance(val, dict):
                    # try numeric range keys min/max/gte/lte/gt/lt
                    minv = val.get("min") or val.get("gte")
                    maxv = val.get("max") or val.get("lte")
                    gt = val.get("gt")
                    lt = val.get("lt")
                    if any(v is not None for v in (minv, maxv, gt, lt)):
                        # extraemos numeric values y usamos (metadata->>key)::numeric
                        if minv is not None:
                            where_clauses.append(f"(metadata->>'{key}')::numeric >= ${next_idx}::numeric")
                            params.append(float(minv))
                            next_idx += 1
                        if maxv is not None:
                            where_clauses.append(f"(metadata->>'{key}')::numeric <= ${next_idx}::numeric")
                            params.append(float(maxv))
                            next_idx += 1
                        if gt is not None:
                            where_clauses.append(f"(metadata->>'{key}')::numeric > ${next_idx}::numeric")
                            params.append(float(gt))
                            next_idx += 1
                        if lt is not None:
                            where_clauses.append(f"(metadata->>'{key}')::numeric < ${next_idx}::numeric")
                            params.append(float(lt))
                            next_idx += 1
                        continue
                # si no es rango, caerá al fallback equality abajo

            # fallback: equality against metadata->>key (string)
                where_clauses.append(f"metadata->>'{key}' = ${next_idx}::text")
                params.append(str(val))
                next_idx += 1

        # finalmente el LIMIT param
        params.append(top_k)
        limit_placeholder = f"${next_idx}"
        # ahora construimos SQL
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query_sql = f"""
        SELECT name_chunk, chunk_id, chunk_size, raw_text, metadata, embedding_model,
                (embedding <=> $1::vector) AS distance
        FROM public.{t}
        {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT {limit_placeholder};
        """

        async with self.client.acquire() as conn:
            try:
                if ef_runtime is not None:
                    await conn.execute(f"SET hnsw.ef_search = {int(ef_runtime)};")

                rows = await conn.fetch(query_sql, *params)

                results = []
                for r in rows:
                    dist = r['distance']
                    similarity = None
                    try:
                        similarity = 1.0 - float(dist) if dist is not None else None
                    except Exception:
                        similarity = None

                    # metadata devuelta por asyncpg ya es dict (si lo insertaste como jsonb)
                    meta = r['metadata']
                    results.append({
                        "name_chunk": r['name_chunk'],
                        "chunk_id": str(r['chunk_id']) if r['chunk_id'] is not None else None,
                        "chunk_size": int(r['chunk_size']) if r['chunk_size'] is not None else None,
                        "raw_text": r['raw_text'],
                        "score": similarity,
                        "embedding_model": r['embedding_model'],
                        "metadata": meta
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

                return results[: top_k]

            except Exception as e:
                logging.exception("Search error")
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
        async with self.client.acquire() as conn:
            try:
                rows = await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'chunks__%';"
                )
                indices = [r['tablename'].replace("chunks__", "", 1) for r in rows]
                return indices
            except Exception as e:
                return []

    async def ready(self):
        async with self.client.acquire() as conn:
            try:
                await conn.fetchval("SELECT 1;")
                return True
            except Exception:
                return False