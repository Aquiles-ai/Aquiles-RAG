import httpx
from typing import List, Literal, Callable, Sequence
from aquiles.utils import chunk_text_by_words
import asyncio

EmbeddingFunc = Callable[[str], Sequence[float]]

class AsyncAquilesRAG:
    def __init__(self, host: str = "http://127.0.0.1:5500", api_key=None):
        """ 
        Asynchronous client to interact with the Aquiles-RAG service.

        Args:
            host (str): Base URL of the Aquiles-RAG server. Defaults to localhost.
            api_key (str, optional): API key for authenticated requests. If provided, included in headers.
        """
        self.base_url = host
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key} if api_key else {}

    async def create_index(self, index_name: str,
                           embeddings_dim: int = 768,
                           dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
                           delete_the_index_if_it_exists: bool = False) -> str:
        """
        Create or overwrite a vector index in the Aquiles-RAG backend.

        Args:
            index_name (str): Unique name for the index.
            embeddings_dim (int): Dimensionality of the embeddings to store.
            dtype (str): Numeric data type for index storage (e.g., FLOAT32).
            delete_the_index_if_it_exists (bool): If True, delete any existing index with the same name before creating.

        Returns:
            str: Server response text indicating success or details.
        """

        url = f'{self.base_url}/create/index'

        body = {"indexname" : index_name,
                "embeddings_dim": embeddings_dim,
                "dtype": dtype,
                "delete_the_index_if_it_exists": delete_the_index_if_it_exists}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, headers=self.headers)
            response.raise_for_status()
            return response.text

    async def query(self, index: str, embedding,
                    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
                    top_k: int = 5,
                    cosine_distance_threshold: float = 0.6) -> List[dict]:
        """
            Query the vector index for nearest neighbors based on cosine similarity.

            Args:
                index (str): Name of the index to search.
                embedding (Sequence[float]): Query embedding vector.
                dtype (str): Data type of the index (must match index creation).
                top_k (int): Number of top matches to return.
                cosine_distance_threshold (float): Maximum cosine distance for valid matches.

            Returns:
                List[dict]: Ordered list of match results with scores and metadata.
            """

        url = f"{self.base_url}/rag/query-rag"

        body = {
            "index": index,
            "embeddings": embedding,
            "dtype": dtype,
            "top_k": top_k,
            "cosine_distance_threshold": cosine_distance_threshold
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def _send_chunk(self, client, url, payload, idx):

        """
        Helper method to send a single chunk to the RAG server.

        Args:
            client (httpx.AsyncClient): The active HTTP client.
            url (str): Endpoint URL.
            payload (dict): Data to send.
            idx (int): Chunk index for tracking.

        Returns:
            dict: Server response or error dictionary.
        """
        try:
            resp = await client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"chunk_index": idx, "error": str(e)}

    async def send_rag(self, embedding_func: EmbeddingFunc, index: str, name_chunk: str,
                        raw_text: str, dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32") -> List[dict]:
        """
        Split raw text into chunks, compute embeddings using the provided function, and store them in the RAG index.

        Args:
            embedding_func (Callable[[str], Sequence[float]]): Function that takes a text chunk and returns its embedding vector.
            index (str): Name of the index to store the embedded documents.
            name_chunk (str): Prefix used to name each chunk (e.g., document name).
            raw_text (str): The full raw text to be split and embedded.
            dtype (str): Data type of the embeddings used in the index.

        Returns:
            List[dict]: Server responses or errors for each chunk upload.
        """

        url = f"{self.base_url}/rag/create"
        chunks = chunk_text_by_words(raw_text)

        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = []
            for idx, chunk in enumerate(chunks, start=1):
                emb = embedding_func(chunk)
                payload = {
                    "index": index,
                    "name_chunk": f"{name_chunk}_{idx}",
                    "dtype": dtype,
                    "chunk_size": 1024,
                    "raw_text": chunk,
                    "embeddings": emb,
                }
                tasks.append(self._send_chunk(client, url, payload, idx))

            return await asyncio.gather(*tasks)