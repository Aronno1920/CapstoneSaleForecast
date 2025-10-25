"""
FastAPI REST API for Sales Forecast System with organized routers
"""
import time
import logging
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..utils.config import AppConfig
from ..database.connection import create_session_factory
from .routers import training_api, forecast_api, eda_sales_api, eda_product_api, eda_market_api


# Configure logging (basic)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Sales Forecast API",
    description="REST API for sales forecasting (Prophet/LightGBM)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize config and DB session factory
config = AppConfig.from_env()
SessionFactory = create_session_factory(config)

# Inject shared state
app.state.config = config
app.state.SessionFactory = SessionFactory

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root API Documentation
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "Sales Forecast API",
        "version": "1.0.0",
        "description": "Sales forecasting over Region/Area/Territory/Year/Month",
        "routers": ["/eda-market", "/eda-product", "/eda-sales", "/train-lightgbm", "/train-prophet"],
        "docs": "/docs",
        "redoc": "/redoc",
        "timestamp": time.time(),
    }


# Include routers
app.include_router(eda_product_api.router)
app.include_router(eda_market_api.router)
app.include_router(eda_sales_api.router)
app.include_router(training_api.router)
app.include_router(forecast_api.router)
