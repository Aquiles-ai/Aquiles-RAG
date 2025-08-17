import os
from typing import List, Dict, Any, Optional, Union
from platformdirs import user_data_dir
import json
from pydantic import BaseModel, Field
import aiofiles
import asyncio
from pathlib import Path

_load_lock = asyncio.Lock()

data_dir = user_data_dir("aquiles", "AquilesRAG")
os.makedirs(data_dir, exist_ok=True)

AQUILES_CONFIG = os.path.join(data_dir, "aquiles_cofig.json")

class AllowedUser(BaseModel):
    username: str = Field(..., description="Allowed username")
    password: str = Field(..., description="Associated password")

class InitConfigsRedis(BaseModel):
    type_c: str = "Redis"
    local: bool = Field(True, description="Redis standalone local")
    host: str = Field("localhost", description="Redis Host")
    port: int = Field(6379, description="Redis Port")
    username: str = Field("", description="If a username has been configured for Redis, configure it here, by default it is not necessary")
    password: str = Field("", description="If a password has been configured for Redis, configure it here, by default it is not necessary")
    cluster_mode: bool = Field(False, description="Option that if you have a Redis Cluster locally, activate it, if you do not have a local cluster leave it as False")
    tls_mode: bool = Field(False, description="Option to connect via SSL/TLS, only leave it as True if you are going to connect via SSL/TLS")
    ssl_cert: str = Field("", description="Absolute path of the SSL Cert")
    ssl_key: str = Field("", description="Absolute path of the SSL Key")
    ssl_ca: str = Field("", description="Absolute path of the SSL CA")
    allows_api_keys: List[str] = Field( default_factory=lambda: [""], description="API KEYS allowed to make requests")
    allows_users: List[AllowedUser] = Field( default_factory=lambda: [AllowedUser(username="root", password="root")],
        description="Users allowed to access the mini-UI and docs"
    )
    initial_cap: int = Field(400)

class InitConfigsQdrant(BaseModel):
    type_c: str = "Qdrant"
    local: bool = Field(True, description="Qdrant standalone local")
    host: str = Field("localhost", description="Qdrant Host")
    port: int = Field(6333, description="Qdrant Port")
    prefer_grpc: bool = Field(False, description="If you are going to use the gRPC connection, activate this")
    grpc_port: int = Field(6334, description="Port for gRPC connections")
    grpc_options: Optional[dict [str, Any]] = Field(default=None, description="Options for communication via gRPC")
    api_key: str = Field(default="", description="API KEY from your Qdrant provider in Cloud")
    auth_token_provider: str = Field(default="", description="Auth Token from your Qdrant provider in Cloud")
    allows_api_keys: List[str] = Field( default_factory=lambda: [""], description="API KEYS allowed to make requests")
    allows_users: List[AllowedUser] = Field( default_factory=lambda: [AllowedUser(username="root", password="root")],
        description="Users allowed to access the mini-UI and docs"
    )

def init_aquiles_config() -> None:
    if not os.path.exists(AQUILES_CONFIG):

        default_configs = InitConfigsRedis().dict()

        with open(AQUILES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(default_configs, f, ensure_ascii=False, indent=2)

def init_aquiles_config_v2(cfg: Union[InitConfigsRedis, InitConfigsQdrant], force: bool = False) -> None:
    if not isinstance(cfg, (InitConfigsRedis, InitConfigsQdrant)):
        raise TypeError("An instance of InitConfigsRedis or InitConfigsQdrant must be passed")

    try:
        conf = cfg.dict()
    except Exception:
        conf = cfg.model_dump()

    config_path = Path(AQUILES_CONFIG)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and not force:
        return

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

#def load_aquiles_config():
#    if os.path.exists(AQUILES_CONFIG):
#        try:
#            with open(AQUILES_CONFIG, "r") as f:
#                return json.load(f)
#        except:
#            return {}
#    return {}

async def load_aquiles_config() -> Dict[str, Any]:
    async with _load_lock:  
        try:
            async with aiofiles.open(AQUILES_CONFIG, "r", encoding="utf-8") as f:
                s = await f.read()
        except FileNotFoundError:
            return {}
        except Exception as exc:
            return {}

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}

async def save_aquiles_configs(configs: Union[dict, InitConfigsRedis, InitConfigsQdrant, BaseModel]) -> None:
    async with _load_lock:
        if isinstance(configs, (InitConfigsRedis, InitConfigsQdrant, BaseModel)):
            try:
                conf = configs.dict()
            except Exception:
                conf = configs.model_dump()
        else:
            conf = configs

        config_path = Path(AQUILES_CONFIG)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = config_path.with_suffix(".tmp")

        json_str = json.dumps(conf, ensure_ascii=False, indent=2)
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(json_str)

        await asyncio.to_thread(os.replace, str(tmp_path), str(config_path))