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
async def forecast_views(req: ForecastViewsRequest, request: Request):
    """
        1) Yearly total sales (forecasted amount) --> "yearly_total": ["Year"],
        2) Yearly selected product sales (requires Product dimension) --> "yearly_selected_product": ["Year", "Product"],
        3) Yearly product-wise sales --> "yearly_product_wise": ["Year", "Product"],
        4) Year/Month total sales --> "year_month_total": ["Year", "Month"],
        5) Year/Month selected product sales --> "year_month_selected_product": ["Year", "Month", "Product"],
        6) Year/Month and product-wise sales --> "year_month_product_wise": ["Year", "Month", "Product"],
        7) Year/Month and Territory-wise total sales --> "year_month_territory_total": ["Year", "Month", "Territory"],
        8) Year/Month and Territory-wise selected product sales --> "year_month_territory_selected_product": ["Year", "Month", "Territory", "Product"],
        9) Year/Month and Area-wise total sales --> "year_month_area_total": ["Year", "Month", "Area"],
        10) Year/Month and Area-wise selected product sales --> "year_month_area_selected_product": ["Year", "Month", "Area", "Product"]
    """
    config = request.app.state.config
    SessionFactory = request.app.state.SessionFactory

    def _run():
        from src.services.forecast_grid_service import forecast_grid
        with SessionFactory() as session:
            # Map the 10 requested conditions to explicit view names and dimensions
            view_map = {
                # 1) Yearly total sales (forecasted amount)
                "yearly_total": ["Year"],
                # 2) Yearly selected product sales (requires Product dimension)
                "yearly_selected_product": ["Year", "Product"],
                # 3) Yearly product-wise sales
                "yearly_product_wise": ["Year", "Product"],
                # 4) Year/Month total sales
                "year_month_total": ["Year", "Month"],
                # 5) Year/Month selected product sales
                "year_month_selected_product": ["Year", "Month", "Product"],
                # 6) Year/Month and product-wise sales
                "year_month_product_wise": ["Year", "Month", "Product"],
                # 7) Year/Month and Territory-wise total sales
                "year_month_territory_total": ["Year", "Month", "Territory"],
                # 8) Year/Month and Territory-wise selected product sales
                "year_month_territory_selected_product": ["Year", "Month", "Territory", "Product"],
                # 9) Year/Month and Area-wise total sales
                "year_month_area_total": ["Year", "Month", "Area"],
                # 10) Year/Month and Area-wise selected product sales
                "year_month_area_selected_product": ["Year", "Month", "Area", "Product"],
            }

            # Determine all dimensions needed for requested views
            req_names = req.views or list(view_map.keys())
            dims_union = set()
            for v in req_names:
                dims_union.update(view_map.get(v, []))
            dims_req = list(dims_union) if dims_union else ["Year", "Month"]

            # Get forecast grid with required dimensions
            fg = forecast_grid(
                session=session,
                config=config,
                scope=req.scope,
                horizon=req.horizon,
                dimensions=dims_req,
            )
            if fg.get("status") != "ok":
                return fg

            # Flatten results into a DataFrame
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

            # Aggregation helper: sum forecasted amount columns
            def agg_for(dims: List[str]):
                missing = [d for d in dims if d not in df.columns]
                if missing:
                    return {"status": "unavailable", "missing_dims": missing}
                grp = (
                    df.groupby(dims, dropna=False)[["yhat", "yhat_lower", "yhat_upper"]]
                    .sum()
                    .reset_index()
                )
                return {
                    "status": "ok",
                    "dimensions": dims,
                    "data": grp.to_dict(orient="records"),
                    "measure": "sales_amount",
                    "note": "Quantity and unit price forecasts are not available; aggregated forecasted sales amount provided.",
                }

            # Build output views
            out: Dict[str, Any] = {"status": "ok", "views": {}}
            for name, dims in view_map.items():
                if name in req_names:
                    out["views"][name] = agg_for(dims)
            return out

    result = await asyncio.to_thread(_run)
    return result