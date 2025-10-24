import asyncio
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(
    tags=["train"],
    responses={404: {"description": "Not found"}}
    )


class TrainRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    horizon: int = 6


@router.post("/train")
async def train(req: TrainRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        from src.services.training_service import train_pipeline
        with SessionFactory() as session:
            return train_pipeline(session=session, config=config, scope=req.scope, horizon=req.horizon)

    result = await asyncio.to_thread(_run)
    return result
