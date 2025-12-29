from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from starlette import status
from typing import Optional
from aquiles.utils.rate_limit_manager import get_api_key_manager
from aquiles.configs import load_aquiles_config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    request: Request = None,
    operation: Optional[str] = None
) -> str:
    configs = await load_aquiles_config()
    valid_keys = [k for k in configs.get("allows_api_keys", []) if k and k.strip()]
    
    if not valid_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key missing",
        )
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    if operation is None:
        return api_key
    
    manager = get_api_key_manager()
    manager.load_from_config(configs)
    
    is_valid, error = await manager.validate_request(
        api_key=api_key,
        operation=operation,
        allowed_keys=valid_keys
    )
    
    if not is_valid:
        if "Rate limit" in error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error,
            )
    
    return api_key

def require_operation(operation: str):
    async def dependency(
        api_key: Optional[str] = Security(api_key_header),
        request: Request = None
    ) -> str:
        return await verify_api_key(api_key, request, operation)
    return dependency