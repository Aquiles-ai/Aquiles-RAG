<h1 align="center">Aquiles-RAG</h1>

<div align="center">
  <img src="aquiles/static/aq-rag2.png" alt="Llada Logo" width="200"/>
</div>

### Description
Aquiles-RAG is a high-performance Retrieval-Augmented Generation (RAG) solution built on Redis. It offers a high-level interface through FastAPI REST APIs to:

* Create RAG indexes in Redis.
* Send raw text alongside its embeddings (the client must chunk the text and compute embeddings before submission).
* Query the index to retrieve the most relevant chunks.

## Features

* **Optimized Performance:** Uses Redis as a vector search engine.
* **Simple API:** Endpoints for index creation, insertion, and querying.
* **Extensible:** Basic implementation ready for enhancements and integration into ML pipelines.

## High-Level Architecture

Here's a diagram illustrating how Aquiles-RAG connects clients to Redis using an asynchronous FastAPI server:

![diagram](aquiles/static/diagram.png)

## Usage

### Create Index

```bash
curl -X POST http://localhost:5500/create/index \
     -H 'Content-Type: application/json' \
     -d '{"indexname": "my_index"}'
```

### Send RAG

```bash
curl -X POST http://localhost:5500/rag/create \
     -H 'Content-Type: application/json' \
     -d '{
           "index": "my_index",
           "raw_text": "Full text goes here...",
           "embeddings": [0.12, 0.34, ...]
         }'
```

### Query RAG

```bash
curl -X POST http://localhost:5500/rag/query-rag \
     -H 'Content-Type: application/json' \
     -d '{
           "index": "my_index",
           "embeddings": [0.56, 0.78, ...],
           "top_k": 5
         }'
```

## Command-Line Interface (CLI)

### Usage Examples

```bash
# Save configuration
aquiles-rag configs --local True --host redis.local --port 6380

# Start server
aquiles-rag serve
```