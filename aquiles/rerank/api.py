import asyncio
from fastapi import Depends, HTTPException, APIRouter, status, Request
from fastapi.responses import JSONResponse
from aquiles.utils import verify_api_key
from aquiles.models import RerankerInput

router = APIRouter(prefix="/v1", tags=["Reranker API"])

@router.post("/rerank", dependencies=[Depends(verify_api_key)])
async def rerank_endpoint(r: RerankerInput, request: Request):
    app = request.app
    configs = getattr(app.state, "aquiles_config", {}) or {}
    reranker = getattr(app.state, "reranker", None)

    if not configs.get("rerank", False) or reranker is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Reranker disabled by config")

    preload_flag = configs.get("reranker_preload", False)

    if not reranker.is_loaded():
        if preload_flag:
            try:
                await reranker.ensure_loaded()
            except Exception as e:
                print("Reranker load error:", e)
                raise HTTPException(status_code=500, detail=f"Failed to load reranker: {e}")
        else:
            asyncio.create_task(reranker.load_async())
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "loading", "detail": "Reranker is loading in background. Retry later."}
            )

    try:
        pairs = r.rerankerjson  
        scores = await reranker.score_pairs(pairs)
        return [{"query": q, "doc": d, "score": float(s)} for (q, d), s in zip(pairs, scores)]
    except Exception as e:
        print("Rerank processing error:", e)
        raise HTTPException(status_code=500, detail=f"Reranker error: {e}")