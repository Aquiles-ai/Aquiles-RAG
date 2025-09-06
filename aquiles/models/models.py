from pydantic import BaseModel, Field, PositiveInt
from typing import List, Literal, Optional, Any, Dict
from aquiles.configs import AllowedUser

class SendRAG(BaseModel):
    index: str = Field(..., description="Index name in Redis")
    name_chunk: str = Field(..., description="Human-readable chunk label or name")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    chunk_size: PositiveInt = Field(1024,
        gt=0,
        description="Number of tokens in each chunk")
    raw_text: str = Field(..., description="Full original text of the chunk")
    embeddings: List[float] = Field(..., description="Vector of embeddings associated with the chunk")
    embedding_model: str | None = Field(default=None, description="Optional metadata field for the embeddings model")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata (key-value)")

class QueryRAG(BaseModel):
    index: str = Field(..., description="Name of the index in which the query will be made")
    embeddings: List[float] = Field(..., description="Embeddings for the query")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    top_k: int = Field(5, description="Number of most similar results to return")
    cosine_distance_threshold: Optional[float] = Field(
        0.6,
        gt=0.0, lt=2.0,
        description="Max cosine distance (0–2) to accept; if omitted, no threshold"
    )
    embedding_model: str | None = Field(default=None, description="Optional metadata field for the embeddings model")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata (key-value)")

class CreateIndex(BaseModel):
    indexname: str = Field(..., description="Name of the index to create")
    embeddings_dim : int = Field(768, description="Dimension of embeddings")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    delete_the_index_if_it_exists: bool = Field(
        False,
        description="If true, will drop any existing index with the same name before creating."
    )

    concurrently: bool | None = Field(default=None, description="Option for postgresql")


class CreateIndexMultimodal(BaseModel):
    indexname: str = Field(..., description="Name of the index to create")
    embeddings_dim_text : int = Field(768, description="Dimension of text embeddings")
    embeddings_dim_image : int = Field(512, description="Image embedding dimensions")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    delete_the_index_if_it_exists: bool = Field(
        False,
        description="If true, will drop any existing index with the same name before creating."
    )

    concurrently: bool | None = Field(default=None, description="Option for postgresql")


class SendRAGMultimodal(BaseModel):
    index: str = Field(..., description="Index name in Redis")
    name_chunk: str = Field(..., description="Human-readable chunk label or name")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    chunk_size: PositiveInt = Field(1024,
        gt=0,
        description="Number of tokens in each chunk")
    caption: str = Field(..., description="Full original text of the chunk")
    embeddings_caption: List[float] = Field(..., description="Vector of embeddings associated with the chunk")
    image: str = Field(..., description="Image path")
    embeddings_image: List[float] = Field(..., description="Image embeddings")
    embedding_model_caption: str | None = Field(default=None, description="Optional metadata field for the embeddings model")
    embedding_model_image: str | None = Field(default=None, description="Optional metadata field for the embeddings model")


class EditsConfigsReds(BaseModel):
    local: Optional[bool] = Field(None, description="Redis standalone local")
    host: Optional[str] = Field(None, description="Redis Host")
    port: Optional[int] = Field(None, description="Redis Port")
    usernanme: Optional[str] = Field(None, description="If a username has been configured for Redis")
    password: Optional[str] = Field(None, description="If a password has been configured for Redis")
    cluster_mode: Optional[bool] = Field(None, description="Use Redis Cluster locally?")
    tls_mode: Optional[bool] = Field(None, description="Connect via SSL/TLS?")
    ssl_cert: Optional[str] = Field(None, description="Absolute path of the SSL Cert")
    ssl_key: Optional[str] = Field(None, description="Absolute path of the SSL Key")
    ssl_ca: Optional[str] = Field(None, description="Absolute path of the SSL CA")
    allows_api_keys: Optional[List[str]] = Field( None, description="New list of allowed API keys (replaces the previous one)")
    allows_users: Optional[List[AllowedUser]] = Field(None, description="New list of allowed users (replaces the previous one)")

class EditsConfigsQdrant(BaseModel):
    local: Optional[bool] = Field(None, description="Qdrant standalone local")
    host: Optional[str] = Field(None, description="Qdrant Host")
    port: Optional[int] = Field(None, description="Qdrant Port")
    prefer_grpc: Optional[bool] = Field(None, description="If you are going to use the gRPC connection, activate this")
    grpc_port: Optional[int] = Field(None, description="Port for gRPC connections")
    grpc_options: Optional[dict [str, Any]] = Field(None, description="Options for communication via gRPC")
    api_key: Optional[str] = Field(None, description="API KEY from your Qdrant provider in Cloud")
    auth_token_provider: Optional[str] = Field(None, description="Auth Token from your Qdrant provider in Cloud")
    allows_api_keys: Optional[List[str]] = Field( None, description="New list of allowed API keys (replaces the previous one)")
    allows_users: Optional[List[AllowedUser]] = Field(None, description="New list of allowed users (replaces the previous one)")

class DropIndex(BaseModel):
    index_name: str = Field(..., description="The name of the index to delete")
    delete_docs: bool = Field(False, description="Removes all documents from the index if true")

class EditsConfigsPostgreSQL(BaseModel):
    local: bool | None= Field(default=None, description="PostgreSQL standalone local")
    host: str | None = Field(default=None, description="PostgreSQL Host")
    port: int | None = Field(default=None, description="PostgreSQL Port")
    user: str | None = Field(default=None, description="")
    password: str | None = Field(default=None, description="")
    database: str | None = Field(default=None, description="")
    min_size: int | None = Field(default=None, description="")
    max_size: int | None = Field(default=None, description="")
    max_queries: int | None = Field(default=None, description="")
    timeout: float | None = Field(default=None, description="")
    allows_api_keys: List[str] | None = Field(default=None)
    allows_users: List[AllowedUser] | None = Field(default=None)


class RerankerInput(BaseModel):
    rerankerjson: List[tuple]

allow_metadata = {"author", "language", "topics", "source", "created_at", "extra"}