import asyncpg
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex
from aquiles.wrapper.basewrapper import BaseWrapper

Pool = asyncpg.Pool

class PostgreSQLRAG(BaseWrapper):
    def __init__(self, client: Pool):
        self.client = client

    async def create_index(self, q: CreateIndex):
        return await super().create_index(q)

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