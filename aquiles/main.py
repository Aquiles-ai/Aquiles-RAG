from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PositiveInt
from typing import List
import redis
from aquiles.configs import load_aquiles_config, save_aquiles_configs, init_aquiles_config
from aquiles.connection import get_connection
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

init_aquiles_config()

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

class EditsConfigs(BaseModel):
    local: bool = Field(True, description="Redis standalone local")
    host: str = Field("localhost", description="Redis Host")
    port: int = Field(6379, description="Redis Port")
    usernanme: str = Field("", description="If a username has been configured for Redis, configure it here, by default it is not necessary")
    password: str = Field("", description="If a password has been configured for Redis, configure it here, by default it is not necessary")
    cluster_mode: bool = Field(False, description="Option that if you have a Redis Cluster locally, activate it, if you do not have a local cluster leave it as False")
    tls_mode: bool = Field(False, description="Option to connect via SSL/TLS, only leave it as True if you are going to connect via SSL/TLS")
    ssl_cert: str = Field("", description="Absolute path of the SSL Cert")
    ssl_key: str = Field("", description="Absolute path of the SSL Key")
    ssl_ca: str = Field("", description="Absolute path of the SSL CA")

@app.post("/create/index")
async def create_index(q: CreateIndex):
    r = get_connection()
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

@app.get("/ui/configs")
async def get_configs():
    configs = load_aquiles_config()
    return {"local": configs["local"],
            "host": configs["host"],
            "port": configs["port"],
            "usernanme": configs["usernanme"],
            "password": configs["password"],
            "cluster_mode": configs["cluster_mode"],
            "ssl_cert": configs["ssl_cert"], 
            "ssl_key": configs["ssl_key"],
            "ssl_ca": configs["ssl_ca"]}

@app.post("/ui/configs")
async def ui_configs(q: EditsConfigs):
    configs = load_aquiles_config()   
    return {"status": "ok"}

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
