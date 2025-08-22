from fastapi import HTTPException
import redis
from redis.commands.search.index_definition import IndexDefinition, IndexType
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from aquiles.models import SendRAG, DropIndex, QueryRAG
from typing import Dict, Any
from aquiles.utils import _escape_tag
from redis.commands.search.query import Query
import redis.asyncio as redisA
from redis.asyncio.cluster import RedisCluster
from typing import Union
from aquiles.wrapper.basewrapper import BaseWrapper
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex

class RdsWr(BaseWrapper):
    def __init__(self, client: Union[redisA.Redis, RedisCluster]):
        self.client = client

    async def create_index(self, q: CreateIndex,
                            schema=None):
        index = self.client.ft(q.indexname)
        exists = True
        try:
            await index.info()  
        except redis.ResponseError:
            exists = False

        if exists and not q.delete_the_index_if_it_exists:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Index '{q.indexname}' already exists. Set delete_the_index_if_it_exists=true to overwrite."
            )

        if exists and q.delete_the_index_if_it_exists:
            await index.dropindex(delete_documents=False)

        definition = IndexDefinition(
            prefix=[f"{q.indexname}:"],
            index_type=IndexType.HASH
        )

        try:
            await self.client.ft(q.indexname).create_index(fields=schema, definition=definition)
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating index: {e}"
            )

    async def send(self, q: SendRAG, emb_bytes=None):
        new_id = await self.client.incr(f"{q.index}:next_id")

        key = f"{q.index}:{new_id}"

        mapping = {
            "name_chunk":   q.name_chunk,
            "chunk_id":     new_id,
            "chunk_size":   q.chunk_size,
            "raw_text":     q.raw_text,
            "embedding":    emb_bytes,
        }

        val = q.embedding_model
        try:
            val = None if val is None else str(val).strip()
        except Exception:
            val = None

        mapping["embedding_model"] = val or "__unknown__"

        try:
            print(f"[DEBUG] Guardando chunk → key={key}, size emb_bytes={len(emb_bytes)} bytes, embedding_model={q.embedding_model!r}")
            await self.client.hset(key, mapping=mapping)
            print(f"[DEBUG] HSET OK para key={key}")
        except Exception as e:
            raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving chunk: {e}")

        return key

    async def query(self, q: QueryRAG, emb_vector):
        model_val = getattr(q, "embedding_model", None)
        if model_val:
            model_val = str(model_val).strip()
            if model_val:  
                safe_tag = _escape_tag(model_val)
                filter_prefix = f"(@embedding_model:{{{safe_tag}}})"
            else:
                filter_prefix = "(*)"
        else:
            filter_prefix = "(*)"

        query_string = f"{filter_prefix}=>[KNN {q.top_k} @embedding $vec AS score]"

        print("[DEBUG] FT.SEARCH query_string:", query_string)

        knn_q = (
            Query(query_string)
            .return_fields("name_chunk", "chunk_id", "chunk_size", "raw_text", "score", "embedding_model")
            .dialect(2)
        )   

        try:
            res = await self.client.ft(q.index).search(knn_q, {"vec": emb_vector})
        except Exception as e:
            print(f"Search error: {e}")
            raise HTTPException(500, f"Search error: {e}")

        docs = res.docs or []

        if q.cosine_distance_threshold is not None:
            try:
                docs = [d for d in docs if float(getattr(d, "score", 0.0)) <= q.cosine_distance_threshold]
            except Exception:
                pass

        docs = docs[: q.top_k]

        results = []
        for doc in docs:
            embedding_model_val = getattr(doc, "embedding_model", None)
            if isinstance(embedding_model_val, (bytes, bytearray)):
                try:
                    embedding_model_val = embedding_model_val.decode("utf-8")
                except Exception:
                    embedding_model_val = None

            results.append({
                "name_chunk": getattr(doc, "name_chunk", None),
                "chunk_id":   int(getattr(doc, "chunk_id", 0)),
                "chunk_size": int(getattr(doc, "chunk_size", 0)),
                "raw_text":   getattr(doc, "raw_text", None),
                "score":      float(getattr(doc, "score", 0.0)),
                "embedding_model": embedding_model_val,
            })

        return results

    async def drop_index(self, q: DropIndex):
        if q.delete_docs:
            res = await self.client.ft(q.index_name).dropindex(True)
        else:
            res = await self.client.ft(q.index_name).dropindex(False)
        return {"status": res, "drop-index": q.index_name}

    async def get_ind(self):
        try:
            indices = await self.client.execute_command("FT._LIST")
            indices = [i.decode() if isinstance(i, bytes) else i for i in indices]
        except redis.RedisError:
            indices = []
        return indices
        

    async def get_status_ram(self):
        info = await self.client.info(section="memory")

        raw_stats = await self.client.memory_stats()
        stats = {
            key.decode() if isinstance(key, (bytes, bytearray)) else key: val
            for key, val in raw_stats.items()
        }

        used = info.get("used_memory", 0)
        maxm = info.get("maxmemory", 0)
        free_memory_mb = ((maxm - used) / 1024**2) if maxm and used else None

        redis_metrics: Dict[str, Any] = {
            "memory_info": info,
            "memory_stats": stats,
            "free_memory_mb": free_memory_mb,
        }

        return redis_metrics

    async def ready(self):
        await self.client.ping()