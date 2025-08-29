from aquiles.configs import InitConfigsRedis, InitConfigsQdrant, InitConfigsPostgreSQL
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import secrets
import os
import json
from platformdirs import user_data_dir
from typing import Union

data_dir = user_data_dir("aquiles", "AquilesRAG")
os.makedirs(data_dir, exist_ok=True)
AQUILES_CONFIG = os.path.join(data_dir, "aquiles_cofig.json")

class DeployConfigRd(InitConfigsRedis, BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    JWT_SECRET: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key to sign JWT"
    )
    ALGORITHM: str = Field("HS256", description="JWT signature algorithm")

class DeployConfigQdrant(InitConfigsQdrant, BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    JWT_SECRET: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key to sign JWT"
    )
    ALGORITHM: str = Field("HS256", description="JWT signature algorithm")

class DeployConfigPostgreSQL(InitConfigsPostgreSQL, BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    JWT_SECRET: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key to sign JWT"
    )
    ALGORITHM: str = Field("HS256", description="JWT signature algorithm")


def gen_configs_file(config: Union[DeployConfigRd, DeployConfigQdrant, DeployConfigPostgreSQL], force: bool = False) -> None:
    """
    Creates the configuration file `aquiles_config.json` in the user's data directory
    (for example: ~/.local/share/aquiles/) **only if it doesn't exist**, or overwrites it
    when `force=True` is passed.

    Purpose
    -------
    Ensure that a deployment configuration file (Redis, API keys, allowed users, JWT, etc.)
    is present before the server starts.

    Args
    ----
    config : DeployConfigRd (Redis) or DeployConfigQdrant (Qdrant)
        A Pydantic configuration instance that contains all required keys.
    force : bool, optional
        If True, overwrites the existing file with the values from `config`. Defaults to False.

    Returns
    -------
    None

    Example
    -------
    >>> cfg = DeployConfigQdrant()
    >>> gen_configs_file(cfg, force=False)
    """

    if not os.path.exists(AQUILES_CONFIG):
        default_configs = config.dict()
        with open(AQUILES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(default_configs, f, ensure_ascii=False, indent=2)
    elif os.path.exists(AQUILES_CONFIG) and force:
        default_configs = config.dict()
        with open(AQUILES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(default_configs, f, ensure_ascii=False, indent=2)