from fastapi import HTTPException, Security, Request, Depends
from fastapi.security import APIKeyHeader
from starlette import status
from typing import Optional