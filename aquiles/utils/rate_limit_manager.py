import asyncio
from typing import Optional, Tuple
from datetime import date
from collections import defaultdict
from aquiles.models import ApiKeysRegistry

class DailyRateLimiter:
    def __init__(self):
        # {api_key: {date: counter}}
        self._usage: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()

    async def check_and_increment(
        self, 
        api_key: str,
        limit: int
    ) -> Tuple[bool, Optional[str]]:
        async with self._lock:
            today = date.today()
            current_count = self._usage[api_key][today]
            
            if current_count >= limit:
                return False, f"Rate limit exceeded: {current_count}/{limit} requests per day"
            
            self._usage[api_key][today] = current_count + 1
            return True, None
    
    async def get_usage(self, api_key: str) -> int:
        async with self._lock:
            today = date.today()
            return self._usage[api_key][today]
    
    def reset(self, api_key: str) -> None:
        if api_key in self._usage:
            del self._usage[api_key]

class ApiKeyManager:    
    def __init__(self):
        self._registry: Optional[ApiKeysRegistry] = None
        self._rate_limiter = DailyRateLimiter()
        self._lock = asyncio.Lock()
    
    def load_from_config(self, aquiles_config: dict) -> None:
        api_keys_config = aquiles_config.get("api_keys_config", {})
        
        if api_keys_config:
            try:
                self._registry = ApiKeysRegistry.model_validate({"configs": api_keys_config})
            except Exception as e:
                print(f"X Error loading api_keys_config: {e}")
                self._registry = ApiKeysRegistry()
        else:
            self._registry = ApiKeysRegistry()
    
    async def validate_request(
        self,
        api_key: str,
        operation: str,  # "create_index", "query", "send", "delete_index"
        allowed_keys: list[str]
    ) -> Tuple[bool, Optional[str]]:
        if api_key not in allowed_keys:
            return False, "Invalid API Key"

        if self._registry is None or len(self._registry.configs) == 0:
            return True, None

        if not self._registry.has_permission(api_key, operation):
            config = self._registry.get_config(api_key)
            if config and not config.enabled:
                return False, "API Key disabled"
            return False, f"Insufficient permission to {operation}"

        config = self._registry.get_config(api_key)
        if config and config.rate_limit:
            limit = config.rate_limit.requests_per_day
            can_proceed, error = await self._rate_limiter.check_and_increment(api_key, limit)
            if not can_proceed:
                return False, error
        
        return True, None
    
    async def get_usage_stats(self, api_key: str) -> dict:
        usage_today = await self._rate_limiter.get_usage(api_key)
        
        config = self._registry.get_config(api_key) if self._registry else None
        
        return {
            "requests_today": usage_today,
            "limit": config.rate_limit.requests_per_day if config and config.rate_limit else None,
            "level": config.level.value if config else "unlimited",
            "enabled": config.enabled if config else True
        }
    
    def get_level(self, api_key: str) -> str:
        if not self._registry:
            return "unlimited"
        
        config = self._registry.get_config(api_key)
        return config.level.value if config else "unlimited"


_global_simple_manager: Optional[ApiKeyManager] = None


def get_api_key_manager() -> ApiKeyManager:
    global _global_simple_manager
    
    if _global_simple_manager is None:
        _global_simple_manager = ApiKeyManager()
    
    return _global_simple_manager