import os
from typing import List
from platformdirs import user_data_dir
import json
from pydantic import BaseModel, Field

data_dir = user_data_dir("aquiles", "AquilesRAG")
os.makedirs(data_dir, exist_ok=True)

AQUILES_CONFIG = os.path.join(data_dir, "aquiles_cofig.json")

class AllowedUser(BaseModel):
    username: str = Field(..., description="Allowed username")
    password: str = Field(..., description="Associated password")

class InitConfigs(BaseModel):
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

def init_aquiles_config() -> None:
    """
    Creates achilles config.json with the default values from InitConfigs if it doesn't exist. Does nothing if it's already present.
    """
    if not os.path.exists(AQUILES_CONFIG):
        # Instancia EditsConfigs con sus valores por defecto
        default_configs = InitConfigs().dict()
        # Guarda el JSON formateado
        with open(AQUILES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(default_configs, f, ensure_ascii=False, indent=2)

def load_aquiles_config():
    if os.path.exists(AQUILES_CONFIG):
        try:
            with open(AQUILES_CONFIG, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_aquiles_configs(configs):
    with open(AQUILES_CONFIG, "w") as f:
        json.dump(configs, f)