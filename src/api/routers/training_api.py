import asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.services.training_service import train_lightgbm_pipeline, train_prophet_pipeline

router = APIRouter(
    tags=["Model Train"],
    responses={404: {"description": "Not found"}}
    )


class TrainRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    horizon: int = 6


class ProphetTrainRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    seasonality_mode: str = "additive"  # additive|multiplicative


####################################################
@router.post("/train-lightgbm")
async def train_lightgbm(req: TrainRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return train_lightgbm_pipeline(session=session, config=config, scope=req.scope, horizon=req.horizon)

    result = await asyncio.to_thread(_run)
    return result


@router.post("/train-prophet")
async def train_prophet(req: ProphetTrainRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return train_prophet_pipeline(
                session=session,
                config=config,
                scope=req.scope,
                seasonality_mode=req.seasonality_mode,
            )

    result = await asyncio.to_thread(_run)
    return result
####################################################