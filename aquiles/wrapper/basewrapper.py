from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex

class BaseWrapper:
    """
    Base class for future Wrappers to handle operations in RAG
    """
    def __init__(self, client) -> None:
        self.client = client

    async def create_index(self, q: CreateIndex):
        raise NotImplementedError("Index creation is not yet available in BaseWrapper.")

    async def create_index_multimodal(self):
        raise NotImplementedError("Index creation is not yet available in BaseWrapper.")

    async def send(self, q: SendRAG)  :
        raise NotImplementedError("Sending data to RAG in BaseWrapper is not yet available.")

    async def send_multimodal(self):
        raise NotImplementedError("Sending data to RAG in BaseWrapper is not yet available.")

    async def query(self, q: QueryRAG, emb_vector):
        raise NotImplementedError("Data queries to RAG are not yet available in BaseWrapper.")

    async def query_multimodal(self, emb_vector):
        raise NotImplementedError("Data queries to RAG are not yet available in BaseWrapper.")

    async def drop_index(self, q: DropIndex):
        raise NotImplementedError("Index deletion is not yet available in RAG in BaseWrapper")

    async def get_ind(self):
        raise NotImplementedError("Getting indexes on RAG in BaseWrapper is not yet available")

    async def ready(self):
        raise NotImplementedError("RAG ping is not yet available in BaseWrapper RAG")

    async def close(self):
        raise NotImplementedError("RAG close is not yet available in BaseWrapper RAG")