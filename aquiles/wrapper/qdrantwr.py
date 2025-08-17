from qdrant_client.models import (
    VectorParams, Distance,
    HnswConfigDiff, PointStruct,
    PayloadSchemaType,
    Filter, FieldCondition, MatchValue, Range
)
from aquiles.models import CreateIndex

class QdrantWr:
    def __init__(self, client):

        self.client = client

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

            await self.client.create_payload_index(c.indexname, "chunk_id", field_schema=PayloadSchemaType.INTEGER)

            await self.client.create_payload_index(c.indexname, "chunk_size", field_schema=PayloadSchemaType.INTEGER)

            await self.client.create_payload_index(c.indexname, "embedding_model", field_schema=PayloadSchemaType.KEYWORD)
        except Exception as e:
                print(f"Error 2 {e}")

    async def create_index(self, q: CreateIndex):
        await self.ensure_collection(q)
        await self.ensure_payload_indexes(q)