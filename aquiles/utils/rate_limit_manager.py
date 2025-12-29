import asyncio
from typing import Optional, Tuple
from datetime import date
from collections import defaultdict
from aquiles.models import ApiKeyLevel, ApiKeyConfig, ApiKeysRegistry

class DailyRateLimiter:
    def __init__(self):
        # {api_key: {date: counter}}
        self._usage: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()