from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from datetime import timedelta
from typing import Dict, Any
import numpy as np
from aquiles.configs import load_aquiles_config, save_aquiles_configs, init_aquiles_config
from aquiles.connection import get_connectionAll
from aquiles.schemas import RedsSch
from aquiles.wrapper import RdsWr
from aquiles.models import QueryRAG, SendRAG, CreateIndex, EditsConfigs, DropIndex
from aquiles.utils import verify_api_key
from aquiles.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_401_UNAUTHORIZED
import os
import pathlib
from contextlib import asynccontextmanager
import psutil
import inspect


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.con = await get_connectionAll()

    app.state.aquiles_config = await load_aquiles_config()
    
    type_co = app.state.aquiles_config.get("type_co", "Redis")
    
    try:
        yield
    finally:
        con = getattr(app.state, "con", None)
        if con is None:
            return
        try:
            if type_co == "Redis":
                if hasattr(con, "aclose"):
                    if inspect.iscoroutinefunction(con.aclose):
                        await con.aclose()
                    else:
                        con.aclose()
                elif hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
                    else:
                        con.close()
            else:
                if hasattr(con, "close"):
                    if inspect.iscoroutinefunction(con.close):
                        await con.close()
                    else:
                        con.close()
        except Exception:
            print("Error cerrando la conexión en shutdown")

app = FastAPI(title="Aquiles-RAG", debug=True, lifespan=lifespan, docs_url=None, redoc_url=None)

package_dir = pathlib.Path(__file__).parent.absolute()
static_dir = os.path.join(package_dir, "static")
templates_dir = os.path.join(package_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

init_aquiles_config()


@app.post("/create/index", dependencies=[Depends(verify_api_key)])
async def create_index(q: CreateIndex, request: Request):
    conf = getattr(request.app.state, "aquiles_config", {}) or {}
    type_co = conf.get("type_co", "Redis")

    if type_co != "Redis":
        raise HTTPException(status_code=400, detail="Index creation with Qdrant has not been implemented yet :(")

    r = request.app.state.con  
    if not hasattr(r, "ft"):
        raise HTTPException(status_code=500, detail="Invalid or uninitialized Redis connection.")

    clientRd = RdsWr(r)

    schema = await RedsSch(q)
    try:
        await clientRd.create_index(index_name=q.indexname, delete_the_index_if_it_exists=q.delete_the_index_if_it_exists,
        schema=schema)
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
    r = request.app.state.con

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

    clientRd = RdsWr(r)

    key = None
    try:
        key = await clientRd.send(q, emb_bytes)
    except Exception as e:
            print(f"Error saving chunk: {e}")

    return {"status": "ok", "key": key}

@app.post("/rag/query-rag", dependencies=[Depends(verify_api_key)])
async def query_rag(q: QueryRAG, request: Request):
    r = request.app.state.con

    if q.dtype == "FLOAT32":
        dtype = np.float32
    elif q.dtype == "FLOAT16":
        dtype = np.float16
    elif q.dtype == "FLOAT64":
        dtype = np.float64
    else:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="dtype not supported")

    emb_array = np.array(q.embeddings, dtype=dtype)
    emb_bytes = emb_array.tobytes()

    clientRd = RdsWr(r)

    results = await clientRd.query(q, emb_bytes)

    return {"status": "ok", "total": len(results), "results": results}


@app.post("/rag/drop_index", dependencies=[Depends(verify_api_key)])
async def drop_index(q: DropIndex, request: Request):
    r = request.app.state.con
    try:

        clientRd = RdsWr(r)
        r = await clientRd.drop_index(q)
        return r
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
        r = request.app.state.con

        clientRd = RdsWr(r)

        indices = await clientRd.get_ind()

        configs = app.state.aquiles_config
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
        configs = app.state.aquiles_config

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
        r = request.app.state.con

        clientRd = RdsWr(r)

        redis_metrics = await clientRd.get_status_ram()

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
    r = request.app.state.con
    try:
        clientRd = RdsWr(r)

        await clientRd.ready()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=5500)
