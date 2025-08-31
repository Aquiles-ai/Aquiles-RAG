import asyncio
from typing import Optional, List

class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
                 providers: Optional[List[str]] = None,
                 max_concurrent: int = 2):
        """
        providers: None (default) o e.g. ["CUDAExecutionProvider"] si usas fastembed-gpu
        """
        self.model_name = model_name
        self.providers = providers
        self.max_concurrent = max_concurrent

        self._encoder = None
        self._sem: Optional[asyncio.Semaphore] = None
        self._load_lock: Optional[asyncio.Lock] = None

    def is_loaded(self) -> bool:
        return self._encoder is not None

    def _blocking_load(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        if self.providers:
            try:
                self._encoder = TextCrossEncoder(model_name=self.model_name, providers=self.providers)
            except TypeError:
                self._encoder = TextCrossEncoder(model_name=self.model_name)
        else:
            self._encoder = TextCrossEncoder(model_name=self.model_name)
        self._sem = asyncio.Semaphore(self.max_concurrent)

    async def load_async(self):
        if self.is_loaded():
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            if self.is_loaded():
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._blocking_load)

    async def ensure_loaded(self):
        await self.load_async()