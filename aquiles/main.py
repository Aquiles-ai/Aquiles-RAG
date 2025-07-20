from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PositiveInt
from typing import List
import redis
from aquiles.configs import load_aquiles_config
from starlette import status
import os
import pathlib

# For now we will leave a silly implementation, while we set up the configs, and everything necessary for Redis

app = FastAPI()

package_dir = pathlib.Path(__file__).parent.absolute()
static_dir = os.path.join(package_dir, "static")
templates_dir = os.path.join(package_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# This is for automatic documentation of Swagger UI.
class SendRAG(BaseModel):
    index: str = Field(..., description="Index name in Redis")
    name_chunk: str = Field(..., description="Human-readable chunk label or name")
    chunk_id: PositiveInt = Field(1, description="Sequential ID of the chunk within the index")
    chunk_size: PositiveInt = Field(1024,
        gt=0,
        description="Number of tokens in each chunk")
    raw_text: str = Field(..., description="Full original text of the chunk")
    embeddings: List[float] = Field(..., description="Vector of embeddings associated with the chunk")

class QueryRAG(BaseModel):
    index: str = Field(..., description="Name of the index in which the query will be made")
    embeddings: List[float] = Field(..., description="Embeddings for the query")
    top_k: int = Field(5, description="Number of most similar results to return")

class CreateIndex(BaseModel):
    indexname: str = Field(..., description="Name of the index to create")

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

@app.get("/ui", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=5500)
