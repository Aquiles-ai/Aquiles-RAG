import requests as r
from typing import List, Literal, Callable, Sequence
from aquiles.utils import chunk_text_by_words

EmbeddingFunc = Callable[[str], Sequence[float]]

class AquilesRAG:
    def __init__(self, host: str = "http://127.0.0.1:5500", api_key = None):
        """ 
        Client for interacting with the Aquiles-RAG service.

        Args
        ----
        host (str): Base URL of the Aquiles-RAG server. Defaults to localhost.
        api_key (str, optional): API key for authenticated requests. If provided, included in headers.
        """
        self.base_url = host
        self.api_key = api_key
        if self.api_key:
            self.header = {"X-API-Key": api_key}

    def create_index(self, index_name: str, 
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
        try:
            if self.api_key:
                response = r.post(url=url, json=body, headers=self.header)
            else:
                response = r.post(url=url, json=body)

            response.raise_for_status()
            return response.text
        except r.RequestException as e:
            raise RuntimeError(f"Failed to create index '{index_name}': {e}")

    def query(self, index: str, embedding, 
            dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
            top_k: int = 5,
            cosine_distance_threshold: float = 0.6,
            embedding_model: str | None = None) -> List[dict]:
            """
            Query the vector index for nearest neighbors based on cosine similarity.

            Args
            ----
            index (str): Name of the index to search.

            embedding (Sequence[float]): Query embedding vector.

            dtype (str): Data type of the index (must match index creation).

            top_k (int): Number of top matches to return.

            cosine_distance_threshold (float): Maximum cosine distance for valid matches.

            embedding_model(str | None, optional): Optional filter to restrict results to embeddings created by a specific model (helps match embeddings produced by different models).

            Returns
            -------
            List[dict]: Ordered list of match results with scores and metadata.
            """

            url = f'{self.base_url}/rag/query-rag'
            if embedding_model:
                body ={
                    "index" : index,
                    "embeddings": embedding,
                    "dtype": dtype,
                    "top_k": top_k,
                    "cosine_distance_threshold": cosine_distance_threshold,
                    "embedding_model": embedding_model
                }
            else:
                body ={
                    "index" : index,
                    "embeddings": embedding,
                    "dtype": dtype,
                    "top_k": top_k,
                    "cosine_distance_threshold": cosine_distance_threshold
                }

            try:
                if self.api_key:
                    response = r.post(url=url, json=body, headers=self.header)
                else:
                    response = r.post(url=url, json=body)
                response.raise_for_status()
                return response.json()
            except r.RequestException as e:
                raise RuntimeError(f"Query failed on index '{index}': {e}")

    def send_rag(self,
                embedding_func: EmbeddingFunc,
                index: str,
                name_chunk: str,
                raw_text: str,
                dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
                embedding_model: str | None = None) -> List[dict]:
                """
                Split text into chunks, compute embeddings, and store them in the index.

                Args
                ----
                embedding_func (Callable[[str], Sequence[float]]): Function that takes a text chunk and returns its embedding vector.

                index (str): Name of the index to store documents.

                base_name (str): Prefix for chunk identifiers (e.g., document name).

                raw_text (str): Full text to be indexed.

                dtype (str): Data type of the index.

                chunk_size (int): Maximum number of words per chunk.
                
                embedding_model(str | None, optional): Embedding model used to compute vectors. Recommend providing this so retrieval can filter/weight by model provenance.

                Returns
                -------
                List[dict]: Server responses for each chunk upload.
                """
                url = f'{self.base_url}/rag/create'

                chunks = chunk_text_by_words(raw_text)
                responses = []

                for idx, chunk in enumerate(chunks, start=1):
                    emb = embedding_func(chunk)

                    if embedding_model:
                        payload = {
                            "index": index,
                            "name_chunk": f"{name_chunk}_{idx}",
                            "dtype": dtype,
                            "chunk_size": 1024,
                            "raw_text": chunk,
                            "embeddings": emb,
                            "embedding_model": embedding_model
                        }
                    else:
                        payload = {
                            "index": index,
                            "name_chunk": f"{name_chunk}_{idx}",
                            "dtype": dtype,
                            "chunk_size": 1024,
                            "raw_text": chunk,
                            "embeddings": emb,
                        }

                    try:
                        if self.api_key:
                            resp = r.post(url, json=payload, headers=self.header, timeout=10)
                        else:
                            resp = r.post(url, json=payload, timeout=10)
                        resp.raise_for_status()
                        responses.append(resp.json())
                    except Exception as e:
                        responses.append({"chunk_index": idx, "error": str(e)})

                return responses

    def drop_index(self, index_name: str, delete_docs: bool = False) -> List[dict]:
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
        try:
            if self.api_key:
                resp = r.post(url=url, json=body, headers=self.header)
            else:
                resp = r.post(url=url, json=body)
            resp.raise_for_status()
            return resp.json()
        except r.RequestException as e:
                raise RuntimeError(f"Error: {e}")