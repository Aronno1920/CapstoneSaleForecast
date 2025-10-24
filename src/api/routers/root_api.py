import time
from typing import Dict, Any
from fastapi import APIRouter

router = APIRouter(
    tags=["Health Check"],
    responses={404: {"description": "Not found"}}
    )


@router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "Sales Forecast API",
        "version": "1.0.0",
        "description": "Sales forecasting over Region/Area/Territory",
        "routers": ["/health", "/status", "/train", "/forecast"],
        "docs": "/docs",
        "timestamp": time.time(),
    }


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/status")
async def status():
    return {"models_ready": True}

