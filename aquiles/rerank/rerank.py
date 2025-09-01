import asyncio
from typing import Optional, List, Union

class Reranker:
    def __init__(self,
                 model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
                 providers: Optional[Union[str, List[str]]] = None,
                 max_concurrent: int = 2):
        self.model_name = model_name
        if isinstance(providers, str):
            self.providers = [providers]
        else:
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

    async def score_pairs(self, pairs: List[tuple]) -> List[float]:
        if not self.is_loaded():
            raise RuntimeError("Reranker not loaded. Call ensure_loaded() first or schedule preload.")

        assert self._sem is not None
        async with self._sem:
            loop = asyncio.get_running_loop()
            def blocking_rerank():
                try:
                    return list(self._encoder.rerank_pairs(pairs))
                except AttributeError:
                    out = []
                    for q, d in pairs:
                        out.extend(list(self._encoder.rerank(q, [d])))
                    return out

            raw_scores = await loop.run_in_executor(None, blocking_rerank)
            return [float(s) for s in raw_scores]
