from fastmcp import FastMCP, Context
from collections.abc import AsyncIterator
from aquiles.configs import load_aquiles_config, save_aquiles_configs
from aquiles.connection import get_connectionAll
from aquiles.schemas import RedsSch
from aquiles.wrapper import RdsWr, QdrantWr, PostgreSQLRAG
from aquiles.mcp.midd import APIKeyMiddleware
import inspect
from contextlib import asynccontextmanager

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

"""
I know it's basic, but I just want to validate that the AI ​​calls the tools and the authentication works.
"""

@mcp.tool()
async def sum_numbers(a: float, b: float, ctx: Context) -> dict:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Dictionary with the result and operation details
    """
    result = a + b
    api_key = ctx.get_state("api_key")
    
    return {
        "operation": "sum",
        "a": a,
        "b": b,
        "result": result,
        "message": f"{a} + {b} = {result}",
        "authenticated_with": api_key if api_key else "no-auth"
    }

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


#if __name__ == "__main__":
#    mcp.run()