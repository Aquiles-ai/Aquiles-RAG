import asyncio
import os
import openai
from agents import Agent, Runner
from agents.mcp import MCPServerSse

async def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    mcp_server = MCPServerSse({"url": "http://127.0.0.1:5500/sse", "headers": {
        "X-API-Key": os.getenv("AQUILES_API_KEY", "dummy-api-key")
    }})
    await mcp_server.connect()

    agent = Agent(
        name="Aquiles Assistant",
        instructions="""
        You are a helpful assistant with access to tools on the MCP server.
        Use the tools to answer user queries.
        """,
        mcp_servers=[mcp_server]
    )

    prompt = """You can see what tools are available and, if there's one to check the connection to 
    the vector database, try it. If it's successful, try creating an index with a 
    random name and then (if successful) retrieve all the created indexes, and tell me what result you get."""

    result = await Runner.run(agent, prompt)
    print(result.final_output)

    await mcp_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
