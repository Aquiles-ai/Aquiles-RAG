from qdrant_client import AsyncQdrantClient
from aquiles.configs import load_aquiles_config
from aquiles.connection import get_connection
import inspect

async def get_connectionAll():
    configs    = await load_aquiles_config()
    type_co = configs.get("type_co", "Redis")
    if type_co == "Redis":
        conn = get_connection()
        if inspect.isawaitable(conn):
            conn = await conn
        return conn
 
    local      = configs.get("local", True)
    host       = configs.get("host", "localhost")
    port       = configs.get("port", 6379)
    prefer_grpc = configs.get("prefer_grpc", False)
    grpc_port = configs.get("grpc_port", 6334)
    grpc_options = configs.get("grpc_options", None)
    api_key = configs.get("api_key", "")
    auth_token_provider = configs.get("auth_token_provider", "")

    if local and prefer_grpc:
        return AsyncQdrantClient(url=host, port=port, grpc_port=grpc_port, grpc_options=grpc_options)
    elif local:
        return AsyncQdrantClient(url=host, port=port)

    if not local and auth_token_provider:
        return AsyncQdrantClient(url=host, port=port, auth_token_provider=auth_token_provider)

    return AsyncQdrantClient(
        url=host,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        grpc_port=grpc_port,
        grpc_options=grpc_options,
    )
    