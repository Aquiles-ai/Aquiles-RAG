import httpx
from typing import List, Literal, Callable, Sequence, Awaitable, Union, Dict, Any
from aquiles.utils import chunk_text_by_words, _extract_text_from_chunk
import asyncio
import inspect
from httpx import Timeout

EmbeddingFunc = Callable[[str], Union[Sequence[float], Awaitable[Sequence[float]]]]

timeout = Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)

class AsyncAquilesRAG:
    def __init__(self, host: str = "http://127.0.0.1:5500", api_key=None):
        """ 
        Asynchronous client to interact with the Aquiles-RAG service.

        Args
        ----
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

        Args
        ----
        index_name (str): Unique name for the index.
        embeddings_dim (int): Dimensionality of the embeddings to store.
        dtype (str): Numeric data type for index storage (e.g., FLOAT32).
        delete_the_index_if_it_exists (bool): If True, delete any existing index with the same name before creating.

        Returns
        -------
        str: Server response text indicating success or details.
        """

        url = f'{self.base_url}/create/index'

        body = {"indexname" : index_name,
                "embeddings_dim": embeddings_dim,
                "dtype": dtype,
                "delete_the_index_if_it_exists": delete_the_index_if_it_exists}
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=self.headers)
            response.raise_for_status()
            return response.text

    async def query(self, index: str, embedding,
                    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
                    top_k: int = 5,
                    cosine_distance_threshold: float = 0.6,
                    embedding_model: str | None = None,
                    metadata: Dict[str, Any] | None = None) -> List[dict]:
        """
        Query the vector index for nearest neighbors using cosine similarity.

        Metadata note
        -------------
        The optional `metadata` parameter filters results to chunks that share the
        provided metadata. Only the following metadata keys are accepted:

            ALLOW_METADATA = {
                "author",       # str: document author (e.g. "Xiaolu Zhang")
                "language",     # str: language code (e.g. "EN", "es"), prefer ISO 639-1
                "topics",       # list[str]: list of topic tags (e.g. ["Diffusion", "LLM"])
                "source",       # str: origin of the content (e.g. "arxiv")
                "created_at",   # str: ISO 8601 datetime recommended (e.g. "2024-08-31T12:34:56+00:00")
                "extra"         # dict: arbitrary additional metadata
            }

        - Keys not in ALLOW_METADATA may be ignored or cause rejection by the backend.
        - `topics` should be a list of strings.
        - `created_at` is recommended in ISO 8601 to enable date filtering/sorting.
        - `extra` may contain arbitrary key/value pairs.

        Args
        ----
        index (str): Name of the index to search.
        embedding (Sequence[float]): Query embedding vector.
        dtype (str): Numeric dtype of the index (must match index creation).
        top_k (int): Number of top matches to return.
        cosine_distance_threshold (float): Max cosine distance for matches.
        embedding_model (str | None, optional): Optional filter for the embedding model.
        metadata (Dict[str, Any] | None, optional): Metadata filter (see above).

        Returns
        -------
        List[dict]: Ordered list of results with scores and metadata.
        """

        url = f"{self.base_url}/rag/query-rag"

        
        body = {
            "index": index,
            "embeddings": embedding,
            "dtype": dtype,
            "top_k": top_k,
            "cosine_distance_threshold": cosine_distance_threshold
        }

        if embedding_model is not None:
                body["embedding_model"] = embedding_model

        if metadata is not None:
            body["metadata"] = metadata

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def _send_chunk(self, client, url, payload, idx):

        """
        Helper method to send a single chunk to the RAG server.

        Args
        ----
        client (httpx.AsyncClient): The active HTTP client.
        url (str): Endpoint URL.
        payload (dict): Data to send.
        idx (int): Chunk index for tracking.

        Returns
        -------
        dict: Server response or error dictionary.
        """
        try:
            resp = await client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"chunk_index": idx, "error": str(e)}

    async def send_rag(self, embedding_func: EmbeddingFunc, index: str, name_chunk: str,
                        raw_text: str, dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
                        embedding_model: str | None = None,
                        metadata: Dict[str, Any] | None = None) -> List[dict]:
        """
        Split raw_text into chunks, compute embeddings using the provided function,
        and store the chunks in the RAG index.

        Metadata note
        -------------
        Each chunk may be annotated with `metadata`. Only the following keys are allowed:

            ALLOW_METADATA = {
                "author", "language", "topics", "source", "created_at", "extra"
            }

        Recommended types and examples:
        - author: str
        - language: str (ISO 639-1 preferred)
        - topics: list[str], e.g. ["Diffusion", "LLM"]
        - source: str, e.g. "arxiv"
        - created_at: str in ISO 8601 or a datetime instance; ISO 8601 is recommended.
        - extra: dict with arbitrary key/value pairs

        Example:
            metadata = {
                "author": "Xiaolu Zhang",
                "language": "EN",
                "topics": list({"Diffusion", "LLM", "LLaDA"}),
                "source": "arxiv",
                "created_at": "2024-08-31T12:34:56+00:00",
                "extra": {"doi": "10.1234/abcd"}
            }

        Behavior:
        - The provided metadata will be attached to every chunk produced for `name_chunk`.
        - Backends that strictly validate metadata may reject keys not in ALLOW_METADATA.
        - Keep types consistent to ensure reliable filtering.

        Args
        ----
        embedding_func (Callable[[str], Union[Sequence[float], Awaitable[Sequence[float]]]]):
            Sync or async function that takes a text chunk and returns its embedding.
        index (str): Index name to persist embeddings.
        name_chunk (str): Base name/prefix for generated chunks (e.g., document filename).
        raw_text (str): Full text to split into chunks and embed.
        dtype (str): Numeric dtype for the index.
        embedding_model (str | None, optional): Embedding model identifier (recommended).
        metadata (Dict[str, Any] | None, optional): Metadata to associate with each chunk.

        Returns
        -------
        List[dict]: Server responses or error dicts for each uploaded chunk.
        """

        url = f"{self.base_url}/rag/create"
        chunks = chunk_text_by_words(raw_text)

        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = []
            for idx, chunk in enumerate(chunks, start=1):
                result = embedding_func(chunk)
                if inspect.isawaitable(result):
                    emb = await result    
                else:
                    emb = result 

                payload = {
                    "index": index,
                    "name_chunk": f"{name_chunk}_{idx}",
                    "dtype": dtype,
                    "chunk_size": 1024,
                    "raw_text": chunk,
                    "embeddings": emb,
                }

                if embedding_model is not None:
                    payload["embedding_model"] = embedding_model

                if metadata is not None:
                    payload["metadata"] = metadata


                tasks.append(self._send_chunk(client, url, payload, idx))

            return await asyncio.gather(*tasks)

    async def drop_index(self, index_name: str, delete_docs: bool = False) -> List[dict]:
        """
            Delete the index and documents if indicated.

            Args
            ----
            index_name (str): Name of the index to delete
            delete_docs (bool): If True, removes documents from the index, by default it is False

            Returns
            -------
            List[dict]: A JSON with the status and name of the deleted index
        """
        url = f'{self.base_url}/rag/drop_index'
        body = {
            "index_name": index_name,
            "delete_docs": delete_docs
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def reranker(self, query: str, docs: Union[List[dict], Dict]) -> List[dict]:
        """
        API to rerank information obtained from the RAG

        Args
        ----
        query (str): original query made to RAG
        docs: (Union[List[dict], Dict]): Results (Raw) from the query API to the RAG to rerank the results
        
        Returns
        -------
        List[dict]: Rerank result
        """

        url = f'{self.base_url}/v1/rerank'

        if isinstance(docs, dict) and "results" in docs:
            docs_list = docs["results"] or []
        elif isinstance(docs, list):
            docs_list = docs
        else:
            if isinstance(docs, dict):
                docs_list = [docs]
            else:
                raise ValueError("docs debe ser List[dict] o Dict con key 'results'")
        
        rag = []
        for d in docs_list:
            text = _extract_text_from_chunk(d)
            if not text:
                continue
            rag.append([query, text])
        payload = {"rerankerjson" : rag}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()