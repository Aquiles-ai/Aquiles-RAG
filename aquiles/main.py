from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import redis
from configs import load_aquiles_config
from starlette import status

# For now we will leave a silly implementation, while we set up the configs, and everything necessary for Redis

app = FastAPI()

class SendRAG(BaseModel):
    index: str
    raw_text: str
    embeddings: List[float]

class QueryRAG(BaseModel):
    index: str
    embeddings: List[float]
    top_k: int = 5

class CreateIndex(BaseModel):
    indexname: str

@app.post("/create/index")
async def create_index(q: CreateIndex):
    # Basic connection to Redis, I will expand this soon
    configs = load_aquiles_config()
    host = configs.get("host", "localhost")
    port = configs.get("port", 6379)
    r = redis.Redis(host=host, port=port, decode_responses=True)
    try:
        r.ft(q.indexname).dropindex(True)
    except redis.ResponseError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"There was an error creating the index: {e}")

@app.post("/rag/create")
async def send_rag(q: SendRAG):
    return {"status": "ok",
            "response": "dummy response"}

@app.post("/rag/query-rag")
async def query_rag(q: QueryRAG):
    return {"status": "ok",
            "response": "dummy response"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=5500)
