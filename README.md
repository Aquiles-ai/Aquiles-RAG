<h1 align="center">Aquiles-RAG</h1>

<div align="center">
  <img width="2560" height="1200" alt="Aquiles-RAG" src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1763680389/aquiles_rag_idhjga.png"/>
</div>

<p align="center">
  <strong>Self-hosted RAG infrastructure with MCP Server support</strong><br/>
  🚀 FastAPI • Redis / Qdrant / PostgreSQL • Async • Embedding-agnostic • MCP Ready
</p>

<p align="center">
  <a href="https://pypi.org/project/aquiles-rag/"><img src="https://img.shields.io/pypi/v/aquiles-rag.svg" alt="PyPI Version"></a>
  <a href="https://aquiles-ai.github.io/aqRAG-docs/"><img src="https://img.shields.io/badge/Docs-Read%20the%20Docs-brightgreen.svg" alt="Documentation"></a>
  <a href="https://pypi.org/project/aquiles-rag/"><img src="https://img.shields.io/pypi/dm/aquiles-rag" alt="PyPI Downloads"></a>
  <a href="https://deepwiki.com/Aquiles-ai/Aquiles-RAG"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>


## 🎯 What is Aquiles-RAG?

**Aquiles-RAG** is a production-ready RAG (Retrieval-Augmented Generation) API server that brings high-performance vector search to your applications. Choose your backend (Redis, Qdrant, or PostgreSQL), connect your embedding model, and start building intelligent search systems in minutes.

### Why Aquiles-RAG?

| Challenge | Aquiles-RAG Solution |
|-----------|----------------------|
| 💸 **Expensive vector databases** | Use Redis, Qdrant, or PostgreSQL you already have |
| 🔒 **Data leaves your infrastructure** | Everything runs on your servers |
| 🔧 **Complex RAG setup** | Interactive wizard configures everything |
| 🐌 **Slow integrations** | Async clients, batch operations, optimized pipelines |
| 🚫 **Vendor lock-in** | Switch backends without changing code |

### Key Features

- **🔌 Backend Flexibility** - Redis HNSW, Qdrant, or PostgreSQL pgvector
- **⚡ High Performance** - Async operations, batch processing, optimized search
- **🤖 MCP Server Built-in** - Native Model Context Protocol support for AI assistants
- **🛠️ Interactive Setup** - CLI wizard configures your entire stack
- **🔄 Sync & Async Clients** - Python and TypeScript/JavaScript SDKs included
- **📊 Optional Re-ranking** - Improve results with semantic re-scoring


## 🚀 Quick Start

### Installation

```bash
pip install aquiles-rag
```

### Interactive Setup

Configure your vector database in seconds:

```bash
aquiles-rag configs
```

The wizard guides you through:
- Backend selection (Redis, Qdrant, or PostgreSQL)
- Connection settings (host, port, credentials)
- TLS/gRPC options
- Optional re-ranker configuration

### Start Server

```bash
aquiles-rag serve --host "0.0.0.0" --port 5500
```

### Your First RAG Query

```python
from aquiles.client import AquilesRAG

client = AquilesRAG(host="http://127.0.0.1:5500", api_key="YOUR_API_KEY")

# Create index
client.create_index("documents", embeddings_dim=768, dtype="FLOAT32")

# Store document with your embedding function
def get_embedding(text):
    return your_embedding_model.encode(text)

client.send_rag(
    embedding_func=get_embedding,
    index="documents",
    name_chunk="intro",
    raw_text="Your document text here..."
)

# Query
results = client.query("documents", query_embedding, top_k=5)
print(results)
```

That's it! You now have a working RAG system.

## 🎨 Supported Backends

| Backend | Features | Best For |
|---------|----------|----------|
| **Redis** | HNSW indexing, fast in-memory search | Speed-critical applications |
| **Qdrant** | HTTP/gRPC, collections, filters | Scalable production systems |
| **PostgreSQL** | pgvector extension, SQL integration | Existing Postgres infrastructure |

All backends support:
- Vector similarity search (cosine, inner product)
- Metadata filtering
- Batch operations
- Optional re-ranking

## 🤖 MCP Server Integration

Aquiles-RAG includes a built-in Model Context Protocol server for seamless AI assistant integration.

### Start MCP Server

```bash
aquiles-rag mcp-serve --host "0.0.0.0" --port 5500 --transport "sse"
```

### Example with OpenAI Agent

```python
from agents import Agent, Runner
from agents.mcp import MCPServerSse

# Connect to MCP server
mcp_server = MCPServerSse({
    "url": "http://localhost:5500/sse",
    "headers": {"X-API-Key": "YOUR_API_KEY"}
})
await mcp_server.connect()

# Create agent with RAG tools
agent = Agent(
    name="RAG Assistant",
    instructions="You can store and query documents using the vector database.",
    mcp_servers=[mcp_server],
    model="gpt-4"
)

# Agent now has access to:
# - create_index
# - send_info (store documents)
# - query_rag (semantic search)
# - list_indexes
# - delete_index

result = await Runner.run(agent, "Store this document and find similar content")
```

**MCP Tools Available:**
- Index management (create, list, delete)
- Document ingestion with automatic chunking
- Semantic search with configurable parameters
- Metadata filtering

## 💻 Client SDKs

### Python - Async Client

```python
from aquiles.client import AsyncAquilesRAG

client = AsyncAquilesRAG(host="http://127.0.0.1:5500", api_key="YOUR_API_KEY")

async def main():
    # Create index
    await client.create_index("docs", embeddings_dim=1536)
    
    # Store documents (parallel chunking)
    await client.send_rag(
        embedding_func=async_get_embedding,
        index="docs",
        name_chunk="document_1",
        raw_text=long_text,
        metadata={
            "author": "John Doe",
            "source": "documentation"
        }
    )
    
    # Query
    results = await client.query("docs", query_embedding, top_k=5)
    print(results)

asyncio.run(main())
```

### TypeScript/JavaScript

```bash
npm install @aquiles-ai/aquiles-rag-client
```

```typescript
import { AsyncAquilesRAG } from '@aquiles-ai/aquiles-rag-client';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function getEmbedding(text: string): Promise<number[]> {
    const resp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
    });
    return resp.data[0].embedding;
}

const client = new AsyncAquilesRAG({
    host: 'http://127.0.0.1:5500',
    apiKey: 'your-api-key',
});

// Create index (1536 dimensions for text-embedding-3-small)
await client.createIndex('my_docs', 1536, 'FLOAT32');

// Store document
await client.sendRAG(
    getEmbedding,
    'my_docs',
    'doc_1',
    'Your document text...',
    {
        embeddingModel: 'text-embedding-3-small',
        metadata: { author: 'John Doe' }
    }
);

// Query
const queryEmb = await getEmbedding('What is this about?');
const results = await client.query('my_docs', queryEmb, { topK: 5 });
console.log(results);
```

## 🛠️ Advanced Features

### Optional Re-ranking

Improve search results with semantic re-scoring:

```bash
# Enable during setup wizard
aquiles-rag configs
```

Re-ranking refines results after vector search by scoring `(query, document)` pairs for better relevance.

### Web UI Playground

Access the interactive UI:

```
http://localhost:5500/ui
```

**Features:**
- Test index creation and queries
- Inspect live configurations
- Protected Swagger UI documentation
- Real-time request/response monitoring


## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                              │
│  HTTP/HTTPS • Python SDK • TypeScript SDK • MCP Server       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Server                            │
│  • Request validation                                        │
│  • Business logic orchestration                              │
│  • Optional re-ranking                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Vector Store                               │
│  Redis HNSW  •  Qdrant Collections  •  PostgreSQL pgvector  │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Client sends embedding + query parameters
2. Server validates and routes to vector store
3. Vector store returns top-k candidates
4. Optional re-ranker refines results
5. Formatted response returned to client

## 🎯 Use Cases

| Who | What |
|-----|------|
| 🚀 **AI Startups** | Build RAG features without vendor costs |
| 👨‍💻 **Developers** | Prototype semantic search quickly |
| 🏢 **Enterprises** | Private, scalable document search |
| 🔬 **Researchers** | Experiment with embeddings and retrieval |

## 📋 Requirements

- Python 3.9+
- One of: Redis, Qdrant, or PostgreSQL with pgvector
- pip or uv

**Quick Redis Setup (Docker):**
```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

**PostgreSQL Note:** Aquiles-RAG doesn't run automatic migrations. Create the `pgvector` extension and required tables manually before use.

## 🛠️ Tech Stack

- **FastAPI** - High-performance async API framework
- **Redis / Qdrant / PostgreSQL** - Vector storage backends
- **NumPy** - Efficient array operations
- **Pydantic** - Request/response validation
- **HTTPX** - Async HTTP client
- **Click** - CLI framework

## 📚 REST API Examples

### Create Index

```bash
curl -X POST http://localhost:5500/create/index \
  -H "X-API-Key: YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "indexname": "documents",
    "embeddings_dim": 768,
    "dtype": "FLOAT32"
  }'
```

### Insert Document

```bash
curl -X POST http://localhost:5500/rag/create \
  -H "X-API-Key: YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "index": "documents",
    "name_chunk": "doc1_part1",
    "raw_text": "Document content...",
    "embeddings": [0.12, 0.34, ...]
  }'
```

### Query

```bash
curl -X POST http://localhost:5500/rag/query-rag \
  -H "X-API-Key: YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "index": "documents",
    "embeddings": [0.78, 0.90, ...],
    "top_k": 5,
    "cosine_distance_threshold": 0.6
  }'
```

## ⚠️ Backend Notes

**Redis:**
- Fast in-memory HNSW indexing
- Full metrics via `/status/ram`
- Supports HASH storage with COSINE search

**Qdrant:**
- HTTP or gRPC connections
- Collection-based organization
- Limited metrics compared to Redis

**PostgreSQL:**
- Requires manual pgvector setup
- No automatic migrations
- SQL-native filtering and joins
- Check Postgres monitoring for metrics

## 📖 Documentation

- [**Full Documentation**](https://aquiles-ai.github.io/aqRAG-docs/)


## 🤝 Contributing

We welcome contributions! See the test suite in `test/` for examples:
- Client SDK tests
- API endpoint tests
- Deployment validation

## 📄 License

[Apache License](LICENSE)

<div align="center">

**[⭐ Star this project](https://github.com/Aquiles-ai/Aquiles-RAG)** • **[🐛 Report issues](https://github.com/Aquiles-ai/Aquiles-RAG/issues)**

*Built with ❤️ for the AI community*

</div>