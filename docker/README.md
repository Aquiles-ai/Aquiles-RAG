# Docker Deployment Guide for Aquiles-RAG

This guide will help you run Aquiles-RAG with Redis using Docker easily.

> IMPORTANT NOTE: Although this example uses Redis, Aquiles-RAG can also be used with Qdrant and PostgreSQL (pgvector), in addition to Redis.

## 📋 File Structure

```
project/
├── Dockerfile.redis          # For normal API
├── Dockerfile.mcp           # For MCP Server
├── docker-compose.yml       # Complete orchestration
├── requirements.txt         # Python dependencies
├── deploy_redis.py         # Shared configuration for API and MCP
├── .env                    # Environment variables (create from .env.example)
└── .env.example            # Variables template
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

Run everything (Redis + API + MCP) with a single command:

```bash
# 1. Copy the environment variables example file
cp .env.example .env

# 2. Edit .env with your credentials
nano .env

# 3. Start all services
docker-compose up -d

# 4. View logs
docker-compose logs -f
```

**Available services:**
- Redis: `localhost:6379`
- Normal API: `http://localhost:5500`
- MCP Server: `http://localhost:5501`

### Option 2: Individual Docker Build

#### Normal API (without MCP)

```bash
# Build
docker build -f Dockerfile.redis -t aquiles-rag:latest .

# Run
docker run -d \
  --name aquiles-api \
  -p 5500:5500 \
  --env-file .env \
  -e HOST=0.0.0.0 \
  -e PORT=5500 \
  -e WORKERS=2 \
  aquiles-rag:latest
```

#### MCP Server

```bash
# Build
docker build -f Dockerfile.mcp -t aquiles-mcp:latest .

# Run
docker run -d \
  --name aquiles-mcp \
  -p 5501:5500 \
  --env-file .env \
  -e HOST=0.0.0.0 \
  -e PORT=5500 \
  aquiles-mcp:latest
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Redis
REDIS_HOST=localhost          # Use 'redis' if using docker-compose
REDIS_PORT=6379
REDIS_USERNAME=default
REDIS_PASS=
```

### Using External Redis

If you already have an external Redis (not the one from docker-compose):

```bash
# In your .env
REDIS_HOST=your-external-redis.com
REDIS_PORT=12345
REDIS_USERNAME=your-username
REDIS_PASS=your-password
```

Then run only the service you need:

```bash
# Only API
docker-compose up -d aquiles-api

# Only MCP
docker-compose up -d aquiles-mcp
```

## 🔍 Verification

### Test Normal API

```bash
# Health check
curl http://localhost:5500/health

# Create index
curl -X POST http://localhost:5500/create/index \
  -H "X-API-Key: your-api-key" \
  -H 'Content-Type: application/json' \
  -d '{
    "indexname": "test",
    "embeddings_dim": 768,
    "dtype": "FLOAT32"
  }'
```

### Test MCP Server

```bash
# SSE endpoint is available at
http://localhost:5501/sse
```

## 🛠️ Useful Commands

```bash
# View logs
docker-compose logs -f aquiles-api
docker-compose logs -f aquiles-mcp

# Restart services
docker-compose restart

# Stop everything
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View status
docker-compose ps
```

## 📦 Production Deployment

### Render / Railway / Fly.io

1. Copy `deploy_redis.py` to your project
2. Create a `requirements.txt` with:
   ```
   aquiles-rag
   python-dotenv
   ```
3. Configure environment variables in your platform
4. Start command:
   - **Normal API**: `aquiles-rag deploy --host 0.0.0.0 --port 5500 --workers 2 deploy_redis.py`
   - **MCP**: `aquiles-rag deploy-mcp --host 0.0.0.0 --port 5500 deploy_redis.py`

**Note**: Both modes (API and MCP) use the same `deploy_redis.py` file for configuration.

### Key Differences: API vs MCP

| Feature | Normal API | MCP Server |
|---------|-----------|------------|
| Command | `deploy` | `deploy-mcp` |
| Workers | Configurable (--workers) | Not applicable |
| Transport | HTTP/REST | SSE only |
| Endpoint | Multiple REST endpoints | `/sse` |
| Usage | REST Client/Python SDK | AI Assistants Integration |

## ⚠️ Troubleshooting

### X Error: Cannot connect to Redis

```bash
# Verify Redis is running
docker-compose ps redis

# Check Redis logs
docker-compose logs redis

# Verify connectivity
docker-compose exec aquiles-api ping redis
```

### X Error: Port already in use

```bash
# Change port in docker-compose.yml or use variable
PORT=5502 docker-compose up -d
```

### X Error: Environment variables not loading

```bash
# Make sure .env exists
ls -la .env

# Rebuild images
docker-compose up -d --build
```

## 📚 Additional Resources

- [Complete Documentation](https://aquiles-ai.github.io/aqRAG-docs/)
- [Usage Examples](../README.md#usage)
- [GitHub Repository](https://github.com/Aquiles-ai/Aquiles-RAG)