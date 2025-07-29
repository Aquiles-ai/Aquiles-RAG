from aquiles.configs import InitConfigs
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import secrets
import os
import json
from platformdirs import user_data_dir

data_dir = user_data_dir("aquiles", "AquilesRAG")
os.makedirs(data_dir, exist_ok=True)
AQUILES_CONFIG = os.path.join(data_dir, "aquiles_cofig.json")

class DeployConfig(InitConfigs, BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    JWT_SECRET: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key to sign JWT"
    )
    ALGORITHM: str = Field("HS256", description="JWT signature algorithm")

def gen_configs_file(config: DeployConfig) -> None:
    if not os.path.exists(AQUILES_CONFIG):
        default_configs = config.dict()
        with open(AQUILES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(default_configs, f, ensure_ascii=False, indent=2)