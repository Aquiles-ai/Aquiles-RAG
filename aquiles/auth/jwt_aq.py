# auth.py
from aquiles.configs import load_aquiles_config
from fastapi import Depends, HTTPException, Cookie
from datetime import datetime, timedelta
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import jwt
import secrets

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    JWT_SECRET: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key to sign JWT"
    )
    ALGORITHM: str = Field("HS256", description="JWT signature algorithm")

settings = Settings(ALGORITHM="HS256")

#cfg = load_aquiles_config_sync()
#users_db = {u["username"]: u["password"] for u in cfg.get("allows_users", [])}

SECRET_KEY = settings.JWT_SECRET
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = 30

async def authenticate_user(username: str, password: str) -> bool:
    cfg = await load_aquiles_config()
    users_db = {u["username"]: u["password"] for u in cfg.get("allows_users", [])}
    pwd = users_db.get(username)
    return bool(pwd and pwd == password)

async def create_access_token(username: str, expires_delta: timedelta) -> str:
    to_encode = {"sub": username}
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token_from_cookie(access_token: str = Cookie(None)) -> str:
    """
    Extrae 'access_token' de la cookie, espera formato 'Bearer <token>'.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="No token cookie")
    scheme, _, token = access_token.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Esquema de autenticación inválido")
    return token

async def get_current_user(token: str = Depends(get_token_from_cookie)) -> str:
    """
    Decodifica el JWT y devuelve el 'username' en 'sub'.
    Lanza HTTPException(401) si algo falla.
    """
    try:
        cfg = await load_aquiles_config()
        users_db = {u["username"]: u["password"] for u in cfg.get("allows_users", [])}
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="Usuario no autenticado")
    return username

# Dependencia auxiliar, si la necesitas en otras rutas
async def require_user(user: str = Depends(get_current_user)):
    return user
