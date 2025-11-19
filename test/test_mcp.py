import asyncio
import os
import openai
from openai import AsyncOpenAI as OpenAI
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerSse

openai.api_key = os.getenv("OPENAI_API_KEY")

async def get_emb(text: str):
    client = OpenAI()

    resp = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return resp.data[0].embedding

async def main():

    mcp_server = MCPServerSse({"url": "http://127.0.0.1:5500/sse", "headers": {
        "X-API-Key": os.getenv("AQUILES_API_KEY", "dummy-api-key")
    }})
    await mcp_server.connect()

    agent = Agent(
        name="Aquiles Assistant",
        instructions="""
        You are a helpful assistant with access to tools on the MCP server.
        Use the tools to answer user queries.
        You have access to a tool called **`embedding_gen`**.
        Use it whenever you need to obtain the embedding of a text.
        The tool takes a single parameter (`text`), which contains the content to be vectorized, and returns the corresponding embedding.
        Whenever a task requires generating vector representations, invoke this tool with the appropriate text.
        """,
        mcp_servers=[mcp_server],
        tools=[function_tool(get_emb, name_override="embedding_gen")],
        model="gpt-5.1"
    )

    prompt = """You can see what tools are available and, if there's one to check the connection to 
    the vector database, try it. If it's successful, try creating an index with a 
    random name and then (if successful) retrieve all the created indexes, and tell me what result you get."""

    result = await Runner.run(agent, prompt)
    print(result.final_output)

    await mcp_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
