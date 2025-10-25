import asyncio
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter(
    tags=["Sales Forecast"],
    responses={404: {"description": "Not found"}}
    )


class ForecastRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    horizon: int = 6
    filters: Dict[str, Any] = {}

class ForecastGridRequest(BaseModel):
    scope: str = "territory"  # territory|area|region
    horizon: int = 6
    dimensions: Optional[List[str]] = None  # e.g., ["Region","Area","Territory","Product"]

class ForecastViewsRequest(BaseModel):
    scope: str = "territory"
    horizon: int = 6
    views: List[str] = []

###########################################
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


@router.post("/forecast-grid")
async def forecast_grid_endpoint(req: ForecastGridRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        from src.services.forecast_grid_service import forecast_grid
        with SessionFactory() as session:
            return forecast_grid(
                session=session, 
                config=config, 
                scope=req.scope, 
                horizon=req.horizon, 
                dimensions=req.dimensions
                )

    result = await asyncio.to_thread(_run)
    return result


@router.post("/forecast-views")
async def forecast_views(req: ForecastViewsRequest, request: Request):
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        from src.services.forecast_grid_service import forecast_grid
        with SessionFactory() as session:
            dims_union = set()
            view_map = {
                "year": ["Year"],
                "year_month": ["Year", "Month"],
                "year_product": ["Year", "Product"],
                "year_month_product": ["Year", "Month", "Product"],
                "year_month_territory": ["Year", "Month", "Territory"],
                "year_month_area": ["Year", "Month", "Area"],
                "year_month_region": ["Year", "Month", "Region"],
                "year_month_product_territory": ["Year", "Month", "Product", "Territory"],
                "year_month_product_area": ["Year", "Month", "Product", "Area"],
                "year_month_product_region": ["Year", "Month", "Product", "Region"],
            }
            for v in req.views:
                dims_union.update(view_map.get(v, []))
            dims_req = list(dims_union) if dims_union else ["Year", "Month"]
            fg = forecast_grid(session=session, config=config, scope=req.scope, horizon=req.horizon, dimensions=dims_req)
            if fg.get("status") != "ok":
                return fg
            rows: List[Dict[str, Any]] = []
            for item in fg.get("results", []):
                key_dims = item.get("key_dims", [])
                key_vals = list(item.get("key", []))
                kv = {k: v for k, v in zip(key_dims, key_vals)}
                for rec in item.get("forecast", []):
                    r = {**kv, **rec}
                    rows.append(r)
            if not rows:
                return {"status": "no_data", "message": "No forecast rows"}
            df = pd.DataFrame(rows)
            out: Dict[str, Any] = {"status": "ok", "views": {}}
            def agg_for(dims: List[str]):
                missing = [d for d in dims if d not in df.columns]
                if missing:
                    return {"status": "unavailable", "missing_dims": missing}
                grp = df.groupby(dims, dropna=False)[["yhat", "yhat_lower", "yhat_upper"]].sum().reset_index()
                return {"status": "ok", "dimensions": dims, "data": grp.to_dict(orient="records")}
            for name, dims in view_map.items():
                if not req.views or name in req.views:
                    out["views"][name] = agg_for(dims)
            return out

    result = await asyncio.to_thread(_run)
    return result