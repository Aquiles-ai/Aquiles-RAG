from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from datetime import timedelta
from pydantic import BaseModel, Field, PositiveInt
from typing import List, Optional, Literal, Union, Dict, Any
import redis
import numpy as np
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from aquiles.configs import load_aquiles_config, save_aquiles_configs, init_aquiles_config
from aquiles.connection import get_connection
from aquiles.configs import AllowedUser
from aquiles.utils import verify_api_key
from aquiles.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_401_UNAUTHORIZED
import os
import pathlib
from contextlib import asynccontextmanager
import psutil

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await get_connection()

    yield

    await app.state.redis.aclose()

app = FastAPI(title="Aquiles-RAG", debug=True, lifespan=lifespan)

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
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    chunk_size: PositiveInt = Field(1024,
        gt=0,
        description="Number of tokens in each chunk")
    raw_text: str = Field(..., description="Full original text of the chunk")
    embeddings: List[float] = Field(..., description="Vector of embeddings associated with the chunk")

class QueryRAG(BaseModel):
    index: str = Field(..., description="Name of the index in which the query will be made")
    embeddings: List[float] = Field(..., description="Embeddings for the query")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    top_k: int = Field(5, description="Number of most similar results to return")
    cosine_distance_threshold: Optional[float] = Field(
        0.6,
        gt=0.0, lt=2.0,
        description="Max cosine distance (0–2) to accept; if omitted, no threshold"
    )

class CreateIndex(BaseModel):
    indexname: str = Field(..., description="Name of the index to create")
    embeddings_dim : int = Field(768, description="Dimension of embeddings")
    dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = Field(
        "FLOAT32",
        description="Embedding data type. Choose from FLOAT32, FLOAT64, or FLOAT16"
    )
    delete_the_index_if_it_exists: bool = Field(
        False,
        description="If true, will drop any existing index with the same name before creating."
    )



class EditsConfigs(BaseModel):
    local: Optional[bool] = Field(None, description="Redis standalone local")
    host: Optional[str] = Field(None, description="Redis Host")
    port: Optional[int] = Field(None, description="Redis Port")
    usernanme: Optional[str] = Field(None, description="If a username has been configured for Redis")
    password: Optional[str] = Field(None, description="If a password has been configured for Redis")
    cluster_mode: Optional[bool] = Field(None, description="Use Redis Cluster locally?")
    tls_mode: Optional[bool] = Field(None, description="Connect via SSL/TLS?")
    ssl_cert: Optional[str] = Field(None, description="Absolute path of the SSL Cert")
    ssl_key: Optional[str] = Field(None, description="Absolute path of the SSL Key")
    ssl_ca: Optional[str] = Field(None, description="Absolute path of the SSL CA")
    allows_api_keys: Optional[List[str]] = Field( None, description="New list of allowed API keys (replaces the previous one)")
    allows_users: Optional[List[AllowedUser]] = Field(None, description="New list of allowed users (replaces the previous one)")

class DropIndex(BaseModel):
    index_name: str = Field(..., description="The name of the index to delete")
    delete_docs: bool = Field(False, description="Removes all documents from the index if true")
    

@app.post("/create/index", dependencies=[Depends(verify_api_key)])
async def create_index(q: CreateIndex, request: Request):
    r: Union[Redis, RedisCluster] = request.app.state.redis

    index = r.ft(q.indexname)
    exists = True
    try:
        await index.info()  
    except redis.ResponseError:
        exists = False

    if exists and not q.delete_the_index_if_it_exists:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Index '{q.indexname}' already exists. Set delete_the_index_if_it_exists=true to overwrite."
        )

    if exists and q.delete_the_index_if_it_exists:
        await index.dropindex(delete_documents=False)

    schema = (
        TextField("name_chunk"),
        NumericField("chunk_id", sortable=True),
        NumericField("chunk_size", sortable=True),
        TextField("raw_text"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": q.dtype,
                "DIM": q.embeddings_dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": 400,
                "M": 16,
                "EF_CONSTRUCTION": 200,
                "EF_RUNTIME": 100,
            }
        )
    )

    definition = IndexDefinition(
        prefix=[f"{q.indexname}:"],
        index_type=IndexType.HASH
    )

    try:
        await r.ft(q.indexname).create_index(fields=schema, definition=definition)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating index: {e}"
        )

    return {
        "status": "success",
        "index": q.indexname,
        "fields": [f.name for f in schema]
    }

@app.post("/rag/create", dependencies=[Depends(verify_api_key)])
async def send_rag(q: SendRAG, request: Request):
    r: Union[Redis, RedisCluster] = request.app.state.redis

    if q.dtype == "FLOAT32":
        dtype = np.float32
    elif q.dtype == "FLOAT16":
        dtype = np.float16
    elif q.dtype == "FLOAT64":
        dtype = np.float64
    else:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"dtype not supported"
        )

    emb_array = np.array(q.embeddings, dtype=dtype)
    emb_bytes = emb_array.tobytes()

    new_id = await r.incr(f"{q.index}:next_id")

    key = f"{q.index}:{new_id}"

    mapping = {
        "name_chunk":   q.name_chunk,
        "chunk_id":     new_id,
        "chunk_size":   q.chunk_size,
        "raw_text":     q.raw_text,
        "embedding":    emb_bytes,
    }

    try:
        print(f"[DEBUG] Guardando chunk → key={key}, size emb_bytes={len(emb_bytes)} bytes")
        await r.hset(key, mapping=mapping)
        print(f"[DEBUG] HSET OK para key={key}")
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving chunk: {e}")

    return {"status": "ok", "key": key}

@app.post("/rag/query-rag", dependencies=[Depends(verify_api_key)])
async def query_rag(q: QueryRAG, request: Request):
    r: Union[Redis, RedisCluster] = request.app.state.redis

    if q.dtype == "FLOAT32":
        dtype = np.float32
    elif q.dtype == "FLOAT16":
        dtype = np.float16
    elif q.dtype == "FLOAT64":
        dtype = np.float64
    else:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"dtype not supported"
        )

    emb_array = np.array(q.embeddings, dtype=dtype)
    emb_bytes = emb_array.tobytes()
    
    knn_q = (
        Query(f"*=>[KNN {q.top_k} @embedding $vec AS score]")
        .return_fields("name_chunk", "chunk_id", "chunk_size", "raw_text", "score")
        .dialect(2)
    )

    try:
        res = await r.ft(q.index).search(knn_q, {"vec": emb_bytes})
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(500, f"Search error: {e}")

    print(res)
    docs = res.docs
    print(docs)
    if q.cosine_distance_threshold is not None:
        docs = [
            d for d in docs
            if float(d.score) <= q.cosine_distance_threshold
        ]

    # Limitamos al top_k real
    docs = docs[: q.top_k]
    
    return {
        "status": "ok",
        "total": len(docs),
        "results": [
            {
                "name_chunk": doc.name_chunk,
                "chunk_id":   int(doc.chunk_id),
                "chunk_size": int(doc.chunk_size),
                "raw_text":   doc.raw_text,
                "score":      float(doc.score),
            }
            for doc in docs
        ]
    }

@app.post("/rag/drop_index", dependencies=[Depends(verify_api_key)])
async def drop_index(q: DropIndex, request: Request):
    r: Union[Redis, RedisCluster] = request.app.state.redis
    try:
        if q.delete_docs:
            res = await r.ft(q.index_name).dropindex(True)
        else:
            res = await r.ft(q.index_name).dropindex(False)
        return {"status": res, "drop-index": q.index_name}
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(500, f"Delete error: {e}")

# All of these are routes for the UI. I'm going to try to make them as minimal as possible so as not to affect performance.

@app.exception_handler(HTTPException)
async def auth_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == HTTP_401_UNAUTHORIZED:
        login_url = f"/login/ui?next={request.url.path}"
        return RedirectResponse(url=login_url, status_code=302)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED,
                            detail="Usuario o contraseña inválidos")
    token = create_access_token(
        username=form_data.username,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    response = RedirectResponse(url="/ui", status_code=302)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@app.get("/ui", response_class=HTMLResponse)
async def home(request: Request, user: str = Depends(get_current_user)):
    try:
        return templates.TemplateResponse("ui.html", {"request": request})
    except HTTPException:
        return RedirectResponse(url="/login/ui", status_code=302)

@app.get("/login/ui", response_class=HTMLResponse)
async def login_ui(request: Request):
    return templates.TemplateResponse("login_ui.html", {"request": request})

@app.get("/ui/configs")
async def get_configs(request: Request, user: str = Depends(get_current_user)):
    try:
        r: Union[Redis, RedisCluster] = request.app.state.redis

        try:
            indices = await r.execute_command("FT._LIST")
            indices = [i.decode() if isinstance(i, bytes) else i for i in indices]
        except redis.RedisError:
            indices = []

        configs = load_aquiles_config()
        return {"local": configs["local"],
                "host": configs["host"],
                "port": configs["port"],
                "usernanme": configs["usernanme"],
                "password": configs["password"],
                "cluster_mode": configs["cluster_mode"],
                "ssl_cert": configs["ssl_cert"], 
                "ssl_key": configs["ssl_key"],
                "ssl_ca": configs["ssl_ca"],
                "allows_api_keys": configs["allows_api_keys"],
                "allows_users": configs["allows_users"],
                "indices": indices
                }
    except HTTPException:
        return RedirectResponse(url="/login/ui", status_code=302)

@app.post("/ui/configs")
async def ui_configs(update: EditsConfigs, user: str = Depends(get_current_user)):
    try:
        configs = load_aquiles_config()

        partial = update.model_dump(exclude_unset=True, exclude_none=True)

        if not partial:
            raise HTTPException(
                status_code=400,
                detail="No fields were sent for update."
            )

        configs.update(partial)

        save_aquiles_configs(configs)

        return {"status": "ok", "updated": partial}
    except HTTPException:
        return RedirectResponse(url="/login/ui", status_code=302)

@app.get(app.openapi_url, include_in_schema=False)
async def protected_openapi(user: str = Depends(get_current_user)):
    return JSONResponse(app.openapi())

@app.get("/docs", include_in_schema=False)
async def protected_swagger_ui(request: Request, user: str = Depends(get_current_user)):
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} – Docs",
        swagger_ui_parameters=app.swagger_ui_parameters, 
    )

@app.get("/redoc", include_in_schema=False)
async def protected_redoc_ui(request: Request, user: str = Depends(get_current_user)):
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} – ReDoc",
    )

@app.get("/status/ram")
async def get_status_ram(request: Request) -> Dict[str, Any]:

    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    app_metrics = {
        "process_memory_mb": round(mem_info.rss / 1024**2, 2),
        "process_cpu_percent": proc.cpu_percent(interval=0.1),
    }

    try:
        r: Union[Redis, RedisCluster] = request.app.state.redis

        info = await r.info(section="memory")

        raw_stats = await r.memory_stats()
        stats = {
            key.decode() if isinstance(key, (bytes, bytearray)) else key: val
            for key, val in raw_stats.items()
        }

        used = info.get("used_memory", 0)
        maxm = info.get("maxmemory", 0)
        free_memory_mb = ((maxm - used) / 1024**2) if maxm and used else None

        redis_metrics: Dict[str, Any] = {
            "memory_info": info,
            "memory_stats": stats,
            "free_memory_mb": free_memory_mb,
        }

    except Exception as e:
        redis_metrics = {
            "error": f"Failed to get Redis metrics: {e}"
        }

    return {
        "redis": redis_metrics,
        "app_process": app_metrics,
    }

@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    return templates.TemplateResponse("status.html", {"request": request})

@app.get("/health/live", tags=["health"])
async def liveness():
    return {"status": "alive"}

@app.get("/health/ready", tags=["health"])
async def readiness(request: Request):
    r: Union[Redis, RedisCluster] = request.app.state.redis
    try:
        await r.ping()
        return {"status": "ready"}
    except:
        raise HTTPException(503, "Redis unavailable")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#if __name__ == "__main__":
#    import uvicorn

#    uvicorn.run(app=app, host="0.0.0.0", port=5500)
