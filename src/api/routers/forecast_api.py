import asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter(
    tags=["forecast"],
    responses={404: {"description": "Not found"}}
    )


class ForecastRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    horizon: int = 6
    filters: Dict[str, Any] = {}


@router.post("/forecast")
async def forecast(req: ForecastRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        from src.services.forecast_service import forecast_pipeline
        with SessionFactory() as session:
            return forecast_pipeline(
                session=session,
                config=config,
                scope=req.scope,
                horizon=req.horizon,
                filters=req.filters or {},
            )

    result = await asyncio.to_thread(_run)
    return result
