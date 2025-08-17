from qdrant_client import AsyncQdrantClient
from aquiles.configs import load_aquiles_config
from aquiles.connection import get_connection
import inspect

async def get_connectionAll():
    configs    = await load_aquiles_config()
    type_co = configs.get("type_c", "Redis")
    if type_co == "Redis":
        conn = get_connection()
        if inspect.isawaitable(conn):
            conn = await conn
        return conn
 
    local = configs.get("local", True)
    host = configs.get("host", "localhost")
    port = configs.get("port", 6333)
    prefer_grpc = configs.get("prefer_grpc", False)
    grpc_port = configs.get("grpc_port", 6334)
    grpc_options = configs.get("grpc_options", None)
    api_key = configs.get("api_key", "")
    auth_token_provider = configs.get("auth_token_provider", "")

    try:

        if local:
            if prefer_grpc:
                return AsyncQdrantClient(
                    host=host,
                    grpc_port=grpc_port,
                    prefer_grpc=True,
                    grpc_options=grpc_options,
                )
            client = AsyncQdrantClient(url=f"http://{host}:{port}")

        if not local:
            if prefer_grpc:
                client = AsyncQdrantClient(
                    host=host,
                    grpc_port=grpc_port,
                    api_key=api_key or None,
                    prefer_grpc=True,
                    tls=True, 
                    grpc_options=grpc_options,
                )
            else:
                client = AsyncQdrantClient(
                    url=f"https://{host}:{port}",
                    api_key=api_key or None,
                    https=True
                )

            await client.get_collections()
            return client
    except Exception as e:
        print("Error conectando a Qdrant:", repr(e))
        raise