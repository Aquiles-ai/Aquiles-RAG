from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from aquiles.configs import load_aquiles_config
from starlette import status
from typing import Optional

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
):
    configs = load_aquiles_config()
    valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]
    
    if not valid_keys:
        return None

    if configs["allows_api_keys"]:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key missing",
            )
        if api_key not in configs["allows_api_keys"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return api_key

def chunk_text_by_words(text: str, chunk_size: int = 600) -> list[str]:
    """
    Splits a text into chunks of up to chunk_size words.
    We will use an average of 600 words equivalent to 1024 tokens

    Args:
        text (str): Input text.
        chunk_size (int): Maximum number of words per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks
