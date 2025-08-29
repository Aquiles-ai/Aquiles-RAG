from qdrant_client import AsyncQdrantClient
from aquiles.configs import load_aquiles_config
from aquiles.connection import get_connection
import inspect
from urllib.parse import urlparse, unquote
import asyncpg

async def get_connectionAll():
    configs = await load_aquiles_config()
    type_co = configs.get("type_c", "Redis")
    if type_co == "Redis":
        conn = get_connection()
        if inspect.isawaitable(conn):
            conn = await conn
        return conn


    if type_co == "Qdrant":
        local = configs.get("local", True)
        raw_host = configs.get("host", "localhost")
        port = configs.get("port", 6333)
        prefer_grpc = configs.get("prefer_grpc", False)
        grpc_port = configs.get("grpc_port", 6334)
        grpc_options = configs.get("grpc_options", None)
        api_key = configs.get("api_key", "") or None
        auth_token_provider = configs.get("auth_token_provider", None)

        parsed = urlparse(raw_host)
        if parsed.scheme:
            host = parsed.hostname or parsed.path or "localhost"
            if parsed.port:
                port = parsed.port
        else:
            host = raw_host

        try:
            if local:
                if prefer_grpc:
                    return AsyncQdrantClient(
                        host=host,
                        grpc_port=grpc_port,
                        prefer_grpc=True,
                        grpc_options=grpc_options,
                    )

                client = AsyncQdrantClient(url=f"http://{host}:{port}", api_key=api_key)
                await client.get_collections()
                return client

            if prefer_grpc:
                client = AsyncQdrantClient(
                    host=host,
                    grpc_port=grpc_port,
                    api_key=api_key,
                    prefer_grpc=True,
                    tls=True,
                    grpc_options=grpc_options,
                )
            else:
                scheme = "https" if api_key is not None else "http"
                client = AsyncQdrantClient(
                    url=f"{scheme}://{host}:{port}",
                    api_key=api_key,
                    https=(scheme == "https")
                )
                
            return client

        except Exception as e:
            print("Error conectando a Qdrant:", repr(e))
            raise

    if type_co == "PostgreSQL":
        raw_host = configs.get("host", "localhost")
        port = configs.get("port", 5432)
        user = configs.get("user", None)
        password = configs.get("password", None)
        database = configs.get("database", None) or configs.get("dbname", configs.get("db", None)) or "postgres"
        min_size = int(configs.get("min_size", 1))
        max_size = int(configs.get("max_size", 10))
        timeout = float(configs.get("timeout", 60))
        max_queries = int(configs.get("max_queries", 50000))

        parsed = urlparse(raw_host)
        if parsed.scheme:
            url_user = parsed.username
            url_pass = parsed.password
            url_host = parsed.hostname or parsed.path
            url_port = parsed.port
            url_db = parsed.path[1:] if parsed.path and parsed.path.startswith("/") else parsed.path

            if url_user:
                user = unquote(url_user)
            if url_pass:
                password = unquote(url_pass)
            if url_host:
                raw_host = url_host
            if url_port:
                port = url_port
            if url_db:
                database = url_db

        dsn = None
        if user:
            dsn = f"postgresql://{user}:{password or ''}@{raw_host}:{port}/{database}"
        else:
            dsn = f"postgresql://{raw_host}:{port}/{database}"

        try:
            pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                max_queries=max_queries
            )
            async with pool.acquire() as conn:
                val = await conn.fetchval("SELECT 1;")
                if val != 1:
                    await pool.close()
                    raise Exception("Unexpected result validating Postgres connection.")
            return pool

        except Exception as e:
            print("Error creating Postgres pool:", repr(e))
            raise

    raise RuntimeError(f"Connection type '{type_co}' not supported by get_connectionAll.")