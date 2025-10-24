"""
FastAPI REST API for Sales Forecast System with organized routers
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..utils.config import AppConfig
from ..database.connection import create_session_factory
from .routers import root_api, training_api, forecast_api, eda_api


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


# Include routers
app.include_router(root_api.router)
app.include_router(eda_api.router)
app.include_router(training_api.router)
app.include_router(forecast_api.router)
