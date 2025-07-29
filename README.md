<h1 align="center">Aquiles‑RAG</h1>

<div align="center">
  <img src="aquiles/static/aq-rag2.png" alt="Aquiles‑RAG Logo" width="200"/>
</div>

<p align="center">
  <strong>High‑performance Retrieval‑Augmented Generation (RAG) on Redis</strong><br/>
  🚀 FastAPI • Redis Vector Search • Async • Embedding‑agnostic
</p>

<p align="center">
  <a href="https://aquiles-ai.github.io/aqRAG-docs/">📖 Documentation</a>
</p>



## 📑 Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration & Connection Options](#configuration--connection-options)
6. [Usage](#usage)

   * [CLI](#cli)
   * [REST API](#rest-api)
   * [Python Client](#python-client)
   * [UI Playground](#ui-playground)
7. [Architecture](#architecture)
8. [License](#license)


## ⭐ Features

* 📈 **High Performance**: Redis-powered vector search using HNSW.
* 🛠️ **Simple API**: Endpoints for index creation, insertion, and querying.
* 🔌 **Embedding‑agnostic**: Works with any embedding model (OpenAI, Llama 3, etc.).
* 💻 **Integrated CLI**: Configure and serve with built‑in commands.
* 🧩 **Extensible**: Ready to integrate into ML pipelines or microservices.


## 🛠 Tech Stack

* **Python 3.9+**
* [FastAPI](https://fastapi.tiangolo.com/)
* [Redis](https://redis.io/) + \[`redis-py` async / cluster]
* [NumPy](https://numpy.org/)
* [Pydantic](https://pydantic-docs.helpmanual.io/)
* [Jinja2](https://jinja.palletsprojects.com/)
* [Click](https://click.palletsprojects.com/) (CLI)
* [Requests](https://docs.python-requests.org/) (Python client)
* [Platformdirs](https://github.com/platformdirs/platformdirs) (config management)


## ⚙️ Requirements

1. **Redis** (standalone or cluster)
2. **Python 3.9+**
3. **pip**

> **Optional**: Run Redis with Docker:
>
> ```bash
> docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
> ```


## 🚀 Installation

### Via PyPI

The easiest way is to install directly from PyPI:

```bash
pip install aquiles-rag
```

### From Source (optional)

If you’d like to work from the latest code or contribute:

1. Clone the repository and navigate into it:

   ```bash
   git clone https://github.com/Aquiles-ai/Aquiles-RAG.git
   cd Aquiles-RAG
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. (Optional) Install in editable/development mode:

   ```bash
   pip install -e .
   ```


## 🔧 Configuration & Connection Options

Aquiles‑RAG stores its configuration in:

```
~/.local/share/aquiles/aquiles_config.json
```

By default, it uses:

```json
{
  "local": true,
  "host": "localhost",
  "port": 6379,
  "username": "",
  "password": "",
  "cluster_mode": false,
  "tls_mode": false,
  "ssl_certfile": "",
  "ssl_keyfile": "",
  "ssl_ca_certs": "",
  "allows_api_keys": [""],
  "allows_users": [{"username": "root", "password": "root"}]
}
```

You can modify the config file manually or use the CLI:

```bash
aquiles-rag configs --host redis.example.com --port 6380 --username user --password pass
```

### Redis Connection Modes

Aquiles‑RAG supports four modes to connect to Redis, based on your config:

1. **Local Cluster** (`local=true` & `cluster_mode=true`)

   ```python
   RedisCluster(host=host, port=port, decode_responses=True)
   ```

2. **Standalone Local** (`local=true`)

   ```python
   redis.Redis(host=host, port=port, decode_responses=True)
   ```

3. **Remote with TLS/SSL** (`local=false`, `tls_mode=true`)

   ```python
   redis.Redis(
     host=host,
     port=port,
     username=username or None,
     password=password or None,
     ssl=True,
     decode_responses=True,
     ssl_certfile=ssl_certfile,  # if provided
     ssl_keyfile=ssl_keyfile,    # if provided
     ssl_ca_certs=ssl_ca_certs   # if provided
   )
   ```

4. **Remote without TLS/SSL** (`local=false`, `tls_mode=false`)

   ```python
   redis.Redis(
     host=host,
     port=port,
     username=username or None,
     password=password or None,
     decode_responses=True
   )
   ```

These options give full flexibility to connect to any Redis topology securely.

## 📖 Usage

### CLI

* **Save configs**

  ```bash
  aquiles-rag configs --host "127.0.0.1" --port 6379
  ```

* **Serve the API**

  ```bash
  aquiles-rag serve --host "0.0.0.0" --port 5500
  ```

* **Deploy custom config**

  ```bash
  aquiles-rag deploy --host "0.0.0.0" --port 5500 --workers 4 my_config.py
  ```

### REST API

1. **Create Index**

   ```bash
   curl -X POST http://localhost:5500/create/index \
     -H "X-API-Key: YOUR_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{
       "indexname": "documents",
       "embeddings_dim": 768,
       "dtype": "FLOAT32",
       "delete_the_index_if_it_exists": false
     }'
   ```

2. **Insert Chunk**

   ```bash
   curl -X POST http://localhost:5500/rag/create \
     -H "X-API-Key: YOUR_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{
       "index": "documents",
       "name_chunk": "doc1_part1",
       "dtype": "FLOAT32",
       "chunk_size": 1024,
       "raw_text": "Text of the chunk...",
       "embeddings": [0.12, 0.34, 0.56, ...]
     }'
   ```

3. **Query Top‑K**

   ```bash
   curl -X POST http://localhost:5500/rag/query-rag \
     -H "X-API-Key: YOUR_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{
       "index": "documents",
       "embeddings": [0.78, 0.90, ...],
       "dtype": "FLOAT32",
       "top_k": 5,
       "cosine_distance_threshold": 0.6
     }'
   ```

### Python Client

```python
from aquiles.client import AquilesRAG

client = AquilesRAG(host="http://127.0.0.1:5500", api_key="YOUR_API_KEY")

# Create an index
client.create_index("documents", embeddings_dim=768, dtype="FLOAT32")

# Insert chunks using your embedding function
def get_embedding(text):
    # e.g. call OpenAI, Llama3, etc.
    return embedding_model.encode(text)

responses = client.send_rag(
    embedding_func=get_embedding,
    index="documents",
    name_chunk="doc1",
    raw_text=full_text
)

# Query the index
results = client.query("documents", query_embedding, top_k=5)
print(results)
```

### UI Playground

Access the web UI (with basic auth) at:

```
http://localhost:5500/ui
```


Use it to:

* Edit configurations live
* Test `/create/index`, `/rag/create`, `/rag/query-rag`
* Explore protected Swagger UI & ReDoc docs

#### 🚀 Screenshots

1. **Playground Home**  
   ![Playground Home](aquiles/static/playground.png)

2. **Live Configurations**  
   ![Live Configurations](aquiles/static/config.png)

3. **Creating an Index**  
   ![Creating an Index](aquiles/static/create.png)

4. **Adding Data to RAG**  
   ![Adding Data to RAG](aquiles/static/add.png)

5. **Querying RAG Results**  
   ![Querying RAG Results](aquiles/static/query.png)


## 🏗 Architecture

The following diagram shows the high‑level architecture of Aquiles‑RAG:

![Architecture](aquiles/static/diagram.png)

1. **Clients** (HTTP/HTTPS, Python SDK, or UI Playground) make asynchronous HTTP requests.
2. **FastAPI Server** acts as the orchestration and business‑logic layer, validating requests and translating them to vector store commands.
3. **Redis / RedisCluster** serves as the RAG vector store (HASH + HNSW/COSINE search).

> ***Test Suite***\*: See the **`test/`** direct\*ory for automated tests:
>
> * **client tests** for the Python SDK
> * **API tests** for endpoint behavior
> * **test\_deploy.py** for deployment configuration and startup validation


## 📄 License

[MIT License](LICENSE)
