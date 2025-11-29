from fastmcp import FastMCP, Context
from collections.abc import AsyncIterator
from aquiles.configs import load_aquiles_config, save_aquiles_configs
from aquiles.connection import get_connectionAll
from aquiles.schemas import RedsSch
from aquiles.wrapper import RdsWr, QdrantWr, PostgreSQLRAG
from aquiles.mcp.midd import APIKeyMiddleware
import inspect
from contextlib import asynccontextmanager
from aquiles.models import CreateIndex, SendRAG, QueryRAG, DropIndex
from typing import Literal
import traceback
import numpy as np
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException

class AppState:
    def __init__(self):
        self.con = None
        self.aquiles_config = None
        self.reranker = None

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    state = AppState()
    
    state.con = await get_connectionAll()
    
    state.aquiles_config = await load_aquiles_config()
    
    type_co = state.aquiles_config.get("type_co", state.aquiles_config.get("type_c", "Redis"))
    
    try:
        configs = state.aquiles_config
        if configs.get("rerank", False):
            from aquiles.rerank import Reranker as RerankerClass
            import asyncio
            
            provider = configs.get("provider_re", None)
            model = configs.get("reranker_model", "Xenova/ms-marco-MiniLM-L-6-v2")
            max_re = configs.get("max_concurrent_request", 2)
            providers = provider if isinstance(provider, list) else ([provider] if provider else None)
            
            state.reranker = RerankerClass(model_name=model, providers=providers, max_concurrent=max_re)
            
            if configs.get("reranker_preload", False):
                asyncio.create_task(state.reranker.load_async())
                print("Reranker preload scheduled (background).")
        else:
            state.reranker = None
            print("Reranker disabled by config.")
    except Exception as e:
        print(f"Warning: failed to prepare reranker singleton: {e}")
        state.reranker = None
    
    try:
        yield state
    finally:
        con = state.con
        if con is None:
            return
        
        try:
            if type_co == "Redis":
                if hasattr(con, "aclose"):
                    if inspect.iscoroutinefunction(con.aclose):
                        await con.aclose()
                    else:
                        con.aclose()
                elif hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
                    else:
                        con.close()
            
            elif str(type_co) in ("PostgreSQL", "postgresql", "pg", "postgresql+asyncpg"):
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
                    else:
                        con.close()
            else:
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
                    else:
                        con.close()
        except Exception as e:
            print(f"❌ Error cerrando la conexión en shutdown: {e}")

mcp = FastMCP("Aquiles-RAG MCP Server", lifespan=lifespan)
mcp.add_middleware(APIKeyMiddleware())

@mcp.custom_route("/rag/create", methods=["POST"])
async def send_rag(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        q = SendRAG(**body)
    except Exception as e:
        print(f"X Error parsing request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request body: {str(e)}"
        )

    con = await get_connectionAll()
    aquiles_config = await load_aquiles_config()
    type_co = aquiles_config.get("type_c", "Redis")
    r = con

    try:
        if type_co == "Redis":
            dtype_map = {
                "FLOAT32": np.float32,
                "FLOAT16": np.float16,
                "FLOAT64": np.float64
            }
            
            dtype = dtype_map.get(q.dtype)
            if dtype is None:
                print(f"X dtype not supported: {q.dtype}")
                raise HTTPException(
                    status_code=400,
                    detail=f"dtype '{q.dtype}' not supported. Use: FLOAT32, FLOAT16, or FLOAT64"
                )

            try:
                emb_array = np.array(q.embeddings, dtype=dtype)
                emb_bytes = emb_array.tobytes()
                
                clientRd = RdsWr(r)
                key = await clientRd.send(q, emb_bytes)
                
                return JSONResponse({
                    "status": "ok",
                    "key": key,
                    "type": "Redis"
                })
                
            except Exception as e:
                print(f"X Error saving chunk to Redis: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save to Redis: {str(e)}"
                )

        elif type_co == "Qdrant":
            try:
                clientQdr = QdrantWr(r)
                key = await clientQdr.send(q)
                
                return JSONResponse({
                    "status": "ok",
                    "key": key,
                    "type": "Qdrant"
                })
                
            except Exception as e:
                print(f"X Error saving chunk to Qdrant: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save to Qdrant: {str(e)}"
                )

        elif type_co == "PostgreSQL":
            try:
                clientPg = PostgreSQLRAG(r)
                key = await clientPg.send(q)
                
                return JSONResponse({
                    "status": "ok",
                    "key": key,
                    "type": "PostgreSQL"
                })
                
            except Exception as e:
                print(f"X Error saving chunk to PostgreSQL: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save to PostgreSQL: {str(e)}"
                )
        
        else:
            print(f"X Unknown storage type: {type_co}")
            raise HTTPException(
                status_code=500,
                detail=f"Unknown storage type: {type_co}"
            )

    finally:
        try:
            if type_co == "Redis":
                if hasattr(con, "aclose"):
                    if inspect.iscoroutinefunction(con.aclose):
                        await con.aclose()
                    else:
                        con.aclose()
                elif hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            elif str(type_co) in ("PostgreSQL", "postgresql", "pg", "postgresql+asyncpg"):
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            else:
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
        except Exception as e:
            print(f"X Error closing the connection on shutdown: {e}")

@mcp.custom_route("/create/index", methods=["POST"])
async def create_index_api(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        q = CreateIndex(**body)
    except Exception as e:
        print(f"X Error parsing request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request body: {str(e)}"
        )

    con = await get_connectionAll()
    aquiles_config = await load_aquiles_config()
    type_co = aquiles_config.get("type_c", "Redis")
    r = con

    try:
        if type_co == "Redis":
            if not hasattr(r, "ft"):
                raise HTTPException(status_code=500, detail="Invalid or uninitialized Redis connection.")

            clientRd = RdsWr(r)

            schema = await RedsSch(q)
            try:
                await clientRd.create_index(q, schema=schema)
            except Exception as e:
                print(f"X Error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating index: {e}"
                )

            return JSONResponse({
                "status": "success",
                "index": q.indexname,
                "fields": [f.name for f in schema]
            })

        elif type_co == "Qdrant":
            clientQdr = QdrantWr(r)

            try:
                await clientQdr.create_index(q)
            except Exception as e:
                traceback.print_exc()
                print(f"X Error detallado creating index: {repr(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating index: {e}"
                )

            return JSONResponse({
                "status": "success",
                "index": q.indexname
            })

        elif type_co == "PostgreSQL":
            clientPg = PostgreSQLRAG(r)

            try:
                await clientPg.create_index(q)
            except Exception as e:
                traceback.print_exc()
                print(f"X Error detallado creating index: {repr(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating index: {e}"
                )

            return JSONResponse({
                "status": "success",
                "index": q.indexname
            })
        
        else:
            print(f"X Unknown storage type: {type_co}")
            raise HTTPException(
                status_code=500,
                detail=f"Unknown storage type: {type_co}"
            )

    finally:
        try:
            if type_co == "Redis":
                if hasattr(con, "aclose"):
                    if inspect.iscoroutinefunction(con.aclose):
                        await con.aclose()
                    else:
                        con.aclose()
                elif hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            elif str(type_co) in ("PostgreSQL", "postgresql", "pg", "postgresql+asyncpg"):
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            else:
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
        except Exception as e:
            print(f"X Error closing the connection on shutdown: {e}")


@mcp.custom_route("/rag/query-rag", methods=["POST"])
async def query_api(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        q = QueryRAG(**body)
    except Exception as e:
        print(f"X Error parsing request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request body: {str(e)}"
        )

    con = await get_connectionAll()
    aquiles_config = await load_aquiles_config()
    type_co = aquiles_config.get("type_c", "Redis")
    r = con
    try:
        if type_co == "Redis":
            if q.dtype == "FLOAT32":
                dtypev = np.float32
            elif q.dtype == "FLOAT16":
                dtypev = np.float16
            elif q.dtype == "FLOAT64":
                dtypev = np.float64
            else:
                raise HTTPException(status_code=500, detail="dtype not supported")

            emb_array = np.array(q.embeddings, dtype=dtypev)
            emb_bytes = emb_array.tobytes()

            clientRd = RdsWr(r)

            results = await clientRd.query(q, emb_bytes)

            return JSONResponse({"status": "ok", "total": len(results), "results": results})

        elif type_co == "Qdrant":
            clientQdr = QdrantWr(r)

            results = await clientQdr.query(q, q.embeddings)

            return JSONResponse({"status": "ok", "total": len(results), "results": results})

        elif type_co == "PostgreSQL":
            clientPg = PostgreSQLRAG(r)

            results = await clientPg.query(q, q.embeddings)

            return JSONResponse({"status": "ok", "total": len(results), "results": results})

    finally:
        try:
            if type_co == "Redis":
                if hasattr(con, "aclose"):
                    if inspect.iscoroutinefunction(con.aclose):
                        await con.aclose()
                    else:
                        con.aclose()
                elif hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            elif str(type_co) in ("PostgreSQL", "postgresql", "pg", "postgresql+asyncpg"):
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
            else:
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
        except Exception as e:
            print(f"X Error closing the connection on shutdown: {e}")

@mcp.tool()
async def readiness(ctx: Context) -> dict:
    """
    Check the status of database connections.
    
    Returns:
        Connection status information with detailed error messages if any
    """
    state = ctx.request_context.lifespan_context
    
    type_co = state.aquiles_config.get("type_co", state.aquiles_config.get("type_c", "Redis"))
    r = state.con

    print("Executing readiness()")

    if type_co == "Redis":
        try:
            clientRd = RdsWr(r)
            await clientRd.ready()
            return {"status": "ready", "connection_type": "Redis"}
        except Exception as e:
            print(f"Redis readiness check failed: {e}")
            return {
                "status": "Redis unavailable",
                "connection_type": "Redis",
                "error": str(e)
            }

    elif type_co == "Qdrant":
        try:
            clientQdr = QdrantWr(r)
            await clientQdr.ready()
            return {"status": "ready", "connection_type": "Qdrant"}
        except Exception as e:
            print(f"Qdrant readiness check failed: {e}")
            return {
                "status": "Qdrant unavailable",
                "connection_type": "Qdrant",
                "error": str(e)
            }

    elif type_co == "PostgreSQL":
        try:
            clientPg = PostgreSQLRAG(r)
            await clientPg.ready()
            return {"status": "ready", "connection_type": "PostgreSQL"}
        except Exception as e:
            print(f"PostgreSQL readiness check failed: {e}")
            return {
                "status": "PostgreSQL unavailable",
                "connection_type": "PostgreSQL",
                "error": str(e)
            }

    return {
        "status": "unknown connection type",
        "connection_type": type_co
    }

@mcp.tool()
async def create_index(indexname: str, embeddings_dim : int, dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"], delete_the_index_if_it_exists: bool, concurrently: bool, ctx: Context):
    """
    Create a new vector search index in the configured backend
    (Redis, Qdrant or PostgreSQL).

    Parameters
    ----------
    indexname : str
        Name of the index to create.

    embeddings_dim_text : int
        Dimension of the text embeddings.
        Default = 768.

    dtype : {"FLOAT32", "FLOAT64", "FLOAT16"}
        Embedding data type.
        FLOAT32 → default
        FLOAT64 → high precision, high memory
        FLOAT16 → low precision, low memory

    delete_the_index_if_it_exists : bool
        If true, drops the existing index before creating a new one.

    concurrently : bool | None
        PostgreSQL-only option.
        If true, performs concurrent index creation.
        If None → backend default is used.

    Notes
    -----
    - Redis: returns schema fields along with index info.

    Returns
    -------
    dict
        A dictionary describing success, index name, and in Redis also the
        generated schema.
    """

    state = ctx.request_context.lifespan_context
    type_co = state.aquiles_config.get("type_co", state.aquiles_config.get("type_c", "Redis"))
    r = state.con

    q = CreateIndex(indexname=indexname, embeddings_dim=embeddings_dim, dtype=dtype, delete_the_index_if_it_exists=delete_the_index_if_it_exists, concurrently=concurrently)

    print("Executing create_index()")

    if type_co == "Redis":
        if not hasattr(r, "ft"):
            return {
                "status": "Invalid or uninitialized Redis connection.",
                "connection_type": "Redis",
            }

        clientRd = RdsWr(r)

        schema = await RedsSch(q)
        try:
            await clientRd.create_index(q, schema=schema)
        except Exception as e:
            print(e)
            return {
                "status": "Redis was unable to create the index",
                "connection_type": "Redis",
                "error": str(e)
            }

        return {
            "status": "success",
            "index": indexname,
            "fields": [f.name for f in schema]
        }

    elif type_co == "Qdrant":
        clientQdr = QdrantWr(r)

        try:
            await clientQdr.create_index(q)
        except Exception as e:
            traceback.print_exc()
            print("Qdrant was unable to create the index", repr(e))
            return {
                "status": "Qdrant was unable to create the index",
                "connection_type": "Qdrant",
                "error": str(e)
            }

        return {
            "status": "success",
            "index": q.indexname}

    elif type_co == "PostgreSQL":
        clientPg = PostgreSQLRAG(r)

        try:
            await clientPg.create_index(q)
        except Exception as e:
            traceback.print_exc()
            print("PostgreSQL was unable to create the index:", repr(e))
            return {
                "status": "PostgreSQL was unable to create the index",
                "connection_type": "PostgreSQL",
                "error": str(e)
            }

        return {
            "status": "success",
            "index": q.indexname} 

@mcp.tool()
async def get_ind(ctx: Context):
    """
    It retrieves the created indexes

    Returns
    -------
    dict
        A dictionary that describes all the indexes created
    """

    state = ctx.request_context.lifespan_context
    type_co = state.aquiles_config.get("type_co", state.aquiles_config.get("type_c", "Redis"))
    r = state.con

    print("Executing get_ind()")

    if type_co == "Redis":
        clientRd = RdsWr(r)

        indices = await clientRd.get_ind()
        print(indices)

        return {"indices": indices}

    elif type_co == "Qdrant":
        clientQdr = QdrantWr(r)

        indices = await clientQdr.get_ind()

        return {"indices": indices}


    elif type_co == "PostgreSQL":
        clientPg = PostgreSQLRAG(r)

        indices = await clientPg.get_ind()

        return {"indices": indices}

@mcp.tool()
async def delete_index(index_name: str, delete_docs: bool, ctx: Context):
    """
    Remove an index

    Parameters
    ----------
    indexname : str
        Name of the index to delete.

    delete_docs : bool
        Remove documents from the index

    Returns
    -------
    dict
        A dictionary describing the deleted index
    """
    state = ctx.request_context.lifespan_context
    type_co = state.aquiles_config.get("type_co", state.aquiles_config.get("type_c", "Redis"))
    r = state.con
    q = DropIndex(index_name=index_name, delete_docs=delete_docs)

    print("Executing delete_index()")

    if type_co == "Redis":
        clientRd = RdsWr(r)

        result = await clientRd.drop_index(q)

        return result

    elif type_co == "Qdrant":
        clientQdr = QdrantWr(r)

        result = await clientQdr.drop_index(q)

        return result


    elif type_co == "PostgreSQL":
        clientPg = PostgreSQLRAG(r)

        result = await clientPg.drop_index(q)

        return result



#if __name__ == "__main__":
#    mcp.run()