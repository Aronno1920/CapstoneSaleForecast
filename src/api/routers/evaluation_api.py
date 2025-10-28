import asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.services.evaluation_service import (
    evaluate_prophet_pipeline,
    evaluate_lightgbm_pipeline,
    evaluate_xgboost_pipeline,
    compare_models_pipeline,
)

router = APIRouter(
    tags=["Model Evaluation"],
    responses={404: {"description": "Not found"}},
)


class EvalRequest(BaseModel):
    scope: str = "territory"  # territory|area|region


##############################################
@router.post("/evaluate-prophet")
async def evaluate_prophet(req: EvalRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return evaluate_prophet_pipeline(session=session, config=config, scope=req.scope)

    result = await asyncio.to_thread(_run)
    return result


@router.post("/evaluate-xgboost")
async def evaluate_xgboost(req: EvalRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return evaluate_xgboost_pipeline(session=session, config=config, scope=req.scope)

    result = await asyncio.to_thread(_run)
    return result


@router.post("/compare-models")
async def compare_models(req: EvalRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return compare_models_pipeline(session=session, config=config, scope=req.scope)

    result = await asyncio.to_thread(_run)
    return result


@router.post("/evaluate-lightgbm")
async def evaluate_lightgbm(req: EvalRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        with SessionFactory() as session:
            return evaluate_lightgbm_pipeline(session=session, config=config, scope=req.scope)

    result = await asyncio.to_thread(_run)
    return result
##############################################