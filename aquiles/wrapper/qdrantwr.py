from uuid import uuid4
from qdrant_client.models import (
    VectorParams, Distance,
    HnswConfigDiff, PointStruct,
    PayloadSchemaType,
    Filter, FieldCondition, MatchValue, MatchAny, DatetimeRange, Range
)
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex, allow_metadata
import asyncio
from qdrant_client import AsyncQdrantClient
from aquiles.wrapper.basewrapper import BaseWrapper
from datetime import datetime
from datetime import timezone

def _to_datetime_if_needed(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(int(val), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    return None

def _build_qdrant_filter(metadata: dict | None, embedding_model: str | None, allow_metadata: set):
    if not metadata and not embedding_model:
        return None

    must_conditions = []

    if embedding_model:
        em = str(embedding_model).strip()
        if em:
            must_conditions.append(FieldCondition(key="embedding_model", match=MatchValue(value=em)))

    if not metadata:
        return Filter(must=must_conditions) if must_conditions else None

    for key, value in metadata.items():
        if key not in allow_metadata:
            continue
        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            vals = [v for v in value if v is not None]
            if not vals:
                continue
            must_conditions.append(FieldCondition(key=key, match=MatchAny(any=list(vals))))
            continue

        if isinstance(value, dict):
            if key == "created_at":
                gte = value.get("min") or value.get("gte")
                lte = value.get("max") or value.get("lte")
                dt_gte = _to_datetime_if_needed(gte)
                dt_lte = _to_datetime_if_needed(lte)
                if dt_gte or dt_lte:
                    must_conditions.append(FieldCondition(
                        key=key,
                        range=DatetimeRange(gte=dt_gte, lte=dt_lte)
                    ))
                    continue
            gte = value.get("min") or value.get("gte")
            lte = value.get("max") or value.get("lte")
            gt = value.get("gt")
            lt = value.get("lt")
            if any(v is not None for v in (gte, lte, gt, lt)):
                try:
                    rng = Range(
                        gte=(float(gte) if gte is not None else None),
                        lte=(float(lte) if lte is not None else None),
                        gt=(float(gt) if gt is not None else None),
                        lt=(float(lt) if lt is not None else None),
                    )
                    must_conditions.append(FieldCondition(key=key, range=rng))
                    continue
                except Exception:
                    pass

        must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=must_conditions) if must_conditions else None


class QdrantWr(BaseWrapper):
    def __init__(self, client: AsyncQdrantClient):

        self.client = client
        self._lock = asyncio.Lock()

    async def ensure_collection(self, c: CreateIndex):
        exists = await self.client.collection_exists(c.indexname)
        if not exists:
            try:
                await self.client.create_collection(
                    collection_name=c.indexname,
                    vectors_config=VectorParams(size=c.embeddings_dim, distance=Distance.COSINE),
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200))
            except Exception as e:
                print(f"Error 1 {e}")

        if exists and c.delete_the_index_if_it_exists:
            try:
                await self.client.delete_collection(collection_name=c.indexname, timeout=30)
            
                await self.client.create_collection(
                    collection_name=c.indexname,
                    vectors_config=VectorParams(size=c.embeddings_dim, distance=Distance.COSINE),
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200))
            except Exception as e:
                print(f"Error 2 {e}")

    async def ensure_payload_indexes(self, c: CreateIndex):
        try:
            await self.client.create_payload_index(c.indexname, "name_chunk", field_schema=PayloadSchemaType.TEXT)

            await self.client.create_payload_index(c.indexname, "chunk_id", field_schema=PayloadSchemaType.UUID)

            await self.client.create_payload_index(c.indexname, "chunk_size", field_schema=PayloadSchemaType.INTEGER)

            await self.client.create_payload_index(c.indexname, "embedding_model", field_schema=PayloadSchemaType.KEYWORD)

            await self.client.create_payload_index(c.indexname, "author", field_schema=PayloadSchemaType.KEYWORD)

            await self.client.create_payload_index(c.indexname, "language", field_schema=PayloadSchemaType.KEYWORD)

            await self.client.create_payload_index(c.indexname, "topics", field_schema=PayloadSchemaType.KEYWORD)

            await self.client.create_payload_index(c.indexname, "source", field_schema=PayloadSchemaType.KEYWORD)

            await self.client.create_payload_index(c.indexname, "created_at", field_schema=PayloadSchemaType.DATETIME)

            await self.client.create_payload_index(c.indexname, "extra", field_schema=PayloadSchemaType.KEYWORD)
        except Exception as e:
                print(f"Error 2 {e}")

    async def create_index(self, q: CreateIndex):
        await self.ensure_collection(q)
        await self.ensure_payload_indexes(q)

    async def send(self, q: SendRAG):
        new_id = uuid4()

        chosen_id = new_id    

        val = getattr(q, "embedding_model", None)
        try:
            val = None if val is None else str(val).strip()
        except Exception:
            val = None
        embedding_model_val = val or "__unknown__"

        payload = {
            "name_chunk": getattr(q, "name_chunk", None),
            "chunk_id": getattr(q, "chunk_id", chosen_id) if getattr(q, "chunk_id", None) is not None else chosen_id,
            "chunk_size": getattr(q, "chunk_size", None),
            "raw_text": getattr(q, "raw_text", None),
            "embedding_model": embedding_model_val,
        }

        if q.metadata:
            for key, value in q.metadata.items():
                if key in allow_metadata and value:
                    payload[key] = value
                else:
                    payload[key] = "__unknown__"

        vector = getattr(q, "embeddings", None)
        if vector is None:
            raise ValueError("No vector provided in q.embeddings")

        point = PointStruct(id=str(chosen_id), vector=vector, payload=payload)

        try:
            await self.client.upsert(collection_name=q.index, points=[point], wait=True)
            return chosen_id
        except Exception as e:
            print("Error upserting point to Qdrant: %s", e)
            raise

    async def query(self, q: QueryRAG, emb_vector):

        model_val = getattr(q, "embedding_model", None)
        query_filter = _build_qdrant_filter(getattr(q, "metadata", None), model_val, allow_metadata)

        score_threshold = None
        if getattr(q, "cosine_distance_threshold", None) is not None:
            try:
                dist_thr = float(q.cosine_distance_threshold)
                score_threshold = 1.0 - dist_thr
            except Exception:
                score_threshold = None

        try:
            hits = await self.client.search(
                collection_name=q.index,
                query_vector=emb_vector,
                limit=int(q.top_k) if getattr(q, "top_k", None) is not None else 5,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,  
        )
        except Exception as e:
            raise Exception(f"Search error: {e}")


        results = []
   
        for h in hits:
            payload = getattr(h, "payload", {}) or {}
            embedding_model_val = payload.get("embedding_model", None)
        
            if isinstance(embedding_model_val, (bytes, bytearray)):
                try:
                    embedding_model_val = embedding_model_val.decode("utf-8")
                except Exception:
                    embedding_model_val = None

        
            q_score = getattr(h, "score", None)
        
            results.append({
                "name_chunk": payload.get("name_chunk"),
                "chunk_id":   payload.get("chunk_id", 0) if payload.get("chunk_id") is not None else None,
                "chunk_size": int(payload.get("chunk_size", 0)) if payload.get("chunk_size") is not None else None,
                "raw_text":   payload.get("raw_text"),
                "score":      float(q_score) if q_score is not None else None,
                "embedding_model": embedding_model_val,
                "metadata": {"author": payload.get("author"),
                            "language": payload.get("language"),
                            "topics": payload.get("topics"),
                            "created_at": payload.get("created_at"),
                            "extra": payload.get("extra")}
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

        results = results[: int(getattr(q, "top_k", 5))]

        return results

    async def drop_index(self, q: DropIndex):
        if q.delete_docs:
            res = await self.client.delete_collection(q.index_name)
        else:
            res = await self.client.delete_collection(q.index_name)
        return {"status": res, "drop-index": q.index_name}

    async def get_ind(self):
        try:
            resp = await self.client.get_collections()
            indices = [c.name for c in getattr(resp, "collections", [])]
        except Exception as e:
            print(f"Error {e}")
            indices = []
        return indices
    
    

    async def ready(self):
        # si es perezoso esto, pero solo es para validar que este activo
        await self.client.get_collections()