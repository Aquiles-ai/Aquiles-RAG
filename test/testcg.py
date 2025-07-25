"""
RAG API Load Test (create / add / search)

This script performs a load test against a RAG API exposing the following endpoints:
- `/create/index`: creates a vector index for embeddings
- `/rag/create`: adds embedded chunks to the index
- `/rag/query-rag`: queries the index using embeddings

📌 What does this script do?
1. Generates an embedding pool using `OpenAI.text-embedding-3-small` from `.md` files in a local directory.
2. Executes a ramp-up test to:
   - Create the index (once)
   - Insert embedded documents
   - Perform top-k searches
3. Logs latency and success rate per endpoint.

⚙️ Configuration:
- Data source: Markdown (`.md`) files ej. from `/home/fredy/projects/sd`
- Embedding model: `text-embedding-3-small` (OpenAI API)
- Target index: `md_test_index`
- Simulates increasing concurrency every 2 seconds (1→10→100 workers) per phase

📈 Output:
At the end, it prints performance stats per endpoint:
- Total number of requests
- Success rate
- Latency (average, min, max)

You can easily extend this script for sustained tests (e.g., 50 RPS for 30 minutes) or integrate it with tools like `k6` or `Locust`.

Requirements:
- Python 3.9+
- Packages: `aiohttp`, `openai`, `aquiles-rag` (utils)
"""

import os
import asyncio
import aiohttp
import time
from openai import OpenAI
from aquiles.utils import chunk_text_by_words

API_HOST = "http://localhost:5500"
INDEX_NAME = "md_test_index"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5
HEADER = {"X-API-Key": "dummy-api-key"}

# Embedding pool generation
async def generate_pool(md_folder: str, model: str, pool_size: int):
    client = OpenAI()
    pool = []
    for fname in os.listdir(md_folder):
        if not fname.endswith(".md"): continue
        text = open(os.path.join(md_folder, fname), encoding="utf-8").read()
        for chunk in chunk_text_by_words(text):
            emb_resp = client.embeddings.create(input=chunk, model=model)
            pool.append({
                "index": INDEX_NAME,
                "name_chunk": f"{fname}_c{len(pool)+1}",
                "dtype": "FLOAT32",
                "chunk_size": len(chunk.split()),
                "raw_text": chunk,
                "embeddings": emb_resp.data[0].embedding,
            })
            if len(pool) >= pool_size:
                return pool
    return pool

def new_metrics():
    return {ep: {"total":0, "success":0, "latencies":[]} for ep in ["create_index","rag_create","rag_query_rag"]}
metrics = new_metrics()

def record(endpoint: str, status: int, latency_ms: float):
    m = metrics[endpoint]
    m["total"] += 1
    if 200 <= status < 300:
        m["success"] += 1
    m["latencies"].append(latency_ms)

def print_metrics():
    print("\n=== Final metrics per endpoint ===")
    for ep, m in metrics.items():
        if m["total"] == 0: continue
        lat = m["latencies"]
        print(f"\n- {ep}:")
        print(f"  total requests : {m['total']}")
        print(f"  successes      : {m['success']} ({m['success']/m['total']:.1%})")
        print(f"  avg latency ms : {sum(lat)/len(lat):.1f}")
        print(f"  min latency ms : {min(lat):.1f}")
        print(f"  max latency ms : {max(lat):.1f}")

# Phase 1: Creating a unique index
async def create_index_once(session: aiohttp.ClientSession):
    payload = {"indexname": INDEX_NAME, "embeddings_dim": 1536, "dtype": "FLOAT32"}
    start = time.perf_counter()
    try:
        async with session.post(f"{API_HOST}/create/index", json=payload, headers=HEADER) as resp:
            lat = (time.perf_counter() - start) * 1000
            record("create_index", resp.status, lat)
            print(f"[CREATE] {INDEX_NAME} → {resp.status} [{lat:.1f}ms]")
    except aiohttp.ClientError as e:
        print(f"[CREATE] error: {e}")

async def worker_add(session: aiohttp.ClientSession, item: dict):
    start = time.perf_counter()
    try:
        async with session.post(f"{API_HOST}/rag/create", json=item, headers=HEADER) as resp:
            lat = (time.perf_counter() - start) * 1000
            record("rag_create", resp.status, lat)
    except aiohttp.ClientError as e:
        print(f"[ADD] error: {e}")

async def worker_search(session: aiohttp.ClientSession, item: dict):
    q = {"index": item["index"], "embeddings": item["embeddings"], "dtype": "FLOAT32", "top_k": TOP_K}
    start = time.perf_counter()
    try:
        async with session.post(f"{API_HOST}/rag/query-rag", json=q, headers=HEADER) as resp:
            lat = (time.perf_counter() - start) * 1000
            record("rag_query_rag", resp.status, lat)
    except aiohttp.ClientError as e:
        print(f"[SEARCH] error: {e}")


async def ramp_phase(func, session: aiohttp.ClientSession, pool: list, duration: int, endpoint_name: str):
    end = time.time() + duration
    concurrency = 1
    round_num = 1
    while time.time() < end:
        tasks = [asyncio.create_task(func(session, pool[(round_num-1) % len(pool)])) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"{endpoint_name} round {round_num} x{concurrency} done")
        ep = endpoint_name
        succ = metrics[ep]["success"]
        total = metrics[ep]["total"]
        if total > 0 and succ / total < 0.5:
            concurrency = max(concurrency // 10, 1)
        else:
            concurrency = min(concurrency * 10, 100)
        round_num += 1
        await asyncio.sleep(2)

# Orquestador principal
async def main():
    md_folder = "/home/fredy/projects/sd"
    pool_size = 20
    durations = {"create":10, "add":10, "search":10}

    print("🌱 Generating embedding pools…")
    pool = await generate_pool(md_folder, EMBEDDING_MODEL, pool_size)
    print(f"Pool ready: {len(pool)} items")

    # limit the number of simultaneous connections
    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print("\n>> CREATE INDEX")
        await create_index_once(session)

        print("\n>> ADD PHASE (RAG)")
        await ramp_phase(worker_add, session, pool, durations["add"], endpoint_name="rag_create")

        print("\n>> FASE SEARCH")
        await ramp_phase(worker_search, session, pool, durations["search"], endpoint_name="rag_query_rag")

    print_metrics()

if __name__ == "__main__":
    asyncio.run(main())
