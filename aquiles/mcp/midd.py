from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from typing import Optional
from aquiles.configs import load_aquiles_config

class APIKeyMiddleware(Middleware):
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        request = get_http_request()

        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")

        configs = await load_aquiles_config()
        valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]

        if not valid_keys:
            return await call_next(context)

        if not api_key:
            raise ToolError("API key missing")

        if api_key not in valid_keys:
            raise ToolError("Invalid API key")
        
        context.fastmcp_context.set_state("api_key", api_key)
        
        return await call_next(context)
    
    async def on_list_resources(self, context: MiddlewareContext, call_next):
        return await self._validate_and_continue(context, call_next)
    
    async def on_list_prompts(self, context: MiddlewareContext, call_next):
        return await self._validate_and_continue(context, call_next)
    
    async def _validate_and_continue(self, context: MiddlewareContext, call_next):
        request = get_http_request()
        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        
        configs = await load_aquiles_config()
        valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]
        
        if valid_keys:
            if not api_key or api_key not in valid_keys:
                raise ToolError("Invalid or missing API key")
        
        return await call_next(context)