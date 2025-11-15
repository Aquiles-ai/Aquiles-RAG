from fastmcp import FastMCP, Context
from aquiles.configs import load_aquiles_config, save_aquiles_configs
from aquiles.connection import get_connectionAll
from aquiles.schemas import RedsSch
from aquiles.wrapper import RdsWr, QdrantWr, PostgreSQLRAG
from aquiles.mcp.midd import APIKeyMiddleware

mcp = FastMCP("Aquiles-RAG MCP Server")
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

if __name__ == "__main__":
    mcp.run()