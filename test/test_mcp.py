import asyncio
import os
import openai
from openai import AsyncOpenAI as OpenAI
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerSse
from aquiles.client import AsyncAquilesRAG
from typing import Literal

openai.api_key = os.getenv("OPENAI_API_KEY")

async def get_emb(text: str):
    client = OpenAI()

    resp = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return resp.data[0].embedding

async def send_info(index: str, name_chunk: str, raw_text: str, dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"]):
    client = AsyncAquilesRAG(host="https://aquiles-deploy.onrender.com",api_key=os.getenv("AQUILES_API_KEY", "dummy-api-key")) # MCP server can now be deployed in Render

    result = await client.send_rag(get_emb, index, name_chunk, raw_text, dtype)

    return result

async def query(index: str, text: str, dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32", 
                top_k: int = 5, cosine_distance_threshold: float = 0.6,):
    client = AsyncAquilesRAG(host="https://aquiles-deploy.onrender.com", api_key=os.getenv("AQUILES_API_KEY", "dummy-api-key")) # MCP server can now be deployed in Render
    embedding = await get_emb(text)
    result = await client.query(index=index, embedding=embedding, dtype=dtype, top_k=top_k, cosine_distance_threshold=cosine_distance_threshold)
    return result

async def main():

    mcp_server = MCPServerSse({"url": "https://aquiles-deploy.onrender.com/sse", "headers": { # MCP server can now be deployed in Render
        "X-API-Key": os.getenv("AQUILES_API_KEY", "dummy-api-key")
    }})
    await mcp_server.connect()

    agent = Agent(
        name="Aquiles Assistant",
        instructions="""
        You are a helpful assistant with access to tools on the MCP server for managing and querying a vector database (Aquiles RAG).
        Use the tools to answer user queries efficiently.

        ## Available Tools:

        ### 1. **send_info** - Store text in vector database
        Use this tool whenever you need to add text+embedding to the vector store.
        
        **Signature:** `send_info(index: str, name_chunk: str, raw_text: str, dtype: Literal["FLOAT32","FLOAT64","FLOAT16"])`
        
        **Parameters:**
        - `index`: Target index name where the chunk will be stored
        - `name_chunk`: A short identifier or name for the chunk being stored
        - `raw_text`: The text content to be vectorized and stored
        - `dtype`: Numeric dtype for storage (one of "FLOAT32", "FLOAT64", "FLOAT16")
        
        **Behavior:**
        - This tool internally computes the embedding (via the internal `get_emb` helper) and sends the chunk to Aquiles
        - IMPORTANT: When a task requires generating or storing vector representations, invoke **send_info** with the appropriate parameters
        - If **send_info** fails at any point (embedding computation or sending to Aquiles), stop execution and report exactly where it failed and why

        ### 2. **query_rag** - Search/query the vector database
        Use this tool to search for relevant information stored in the vector database based on semantic similarity.
        
        **Signature:** `query_rag(index: str, text: str, dtype: Literal["FLOAT32","FLOAT64","FLOAT16"] = "FLOAT32", top_k: int = 5, cosine_distance_threshold: float = 0.6)`
        
        **Parameters:**
        - `index`: The index name to search in (must be the same index where data was stored)
        - `text`: The query text to search for (will be converted to an embedding internally)
        - `dtype`: Numeric dtype used when the index was created (must match the storage dtype, default: "FLOAT32")
        - `top_k`: Number of most similar results to return (default: 5)
        - `cosine_distance_threshold`: Minimum similarity score threshold (0.0 to 1.0, default: 0.6). Results below this threshold are filtered out
        
        **Behavior:**
        - This tool internally computes the query embedding and searches for the most similar vectors
        - Returns results sorted by similarity (highest first)
        - Only returns results above the cosine_distance_threshold
        - IMPORTANT: Always use the same `dtype` that was used when storing the data with `send_info`
        
        **Usage Examples:**
        - To search for information: `query_rag(index="my_docs", text="What is machine learning?")`
        - To get more results: `query_rag(index="my_docs", text="AI applications", top_k=10)`
        - To filter low-quality matches: `query_rag(index="my_docs", text="query", cosine_distance_threshold=0.8)`

        ## Workflow Guidelines:

        1. **Storing Data:** Use `send_info` to add documents/chunks to the vector database
        2. **Searching Data:** Use `query_rag` to find relevant information based on user queries
        3. **Consistency:** Always use the same `dtype` for both storing and querying within the same index
        4. **Error Handling:** If any tool fails, report the exact error and suggest corrective actions
        
        ## Important Notes:
        - The embedding model used is `text-embedding-3-small` (1536 dimensions)
        - Ensure the `index` parameter is consistent between `send_info` and `query_rag` operations
        - The `dtype` must match between storage and query operations for the same index
        """,
        mcp_servers=[mcp_server],
        tools=[function_tool(send_info, name_override="send_info"),
                function_tool(query, name_override="query_rag")],
        model="gpt-5"
    )

    prompt = """Execute this test step by step. After EACH step, immediately proceed to the next:

        STEP 1: Test database connection
        STEP 2: Create 2 indexes with random names, when creating the indexes, set the embeddings dimension to 1536.
        STEP 3: List all indexes (then IMMEDIATELY continue)
        STEP 4: Add 8 sentences using send_info (2 per topic: cars, food, sports, tech, music)
        STEP 5: Query RAG with one topic (Create one query that is similar to one of the sentences you sent to the RAG and another that is on the same topic, but does not resemble the sentences you sent)
        STEP 6: Delete 1 index
        STEP 7: Report all results

    IMPORTANT: Do NOT wait after step 3. Continue immediately to step 4.
    Stop only if a step fails."""

    result = await Runner.run(agent, prompt)
    print(result.final_output)

    await mcp_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
