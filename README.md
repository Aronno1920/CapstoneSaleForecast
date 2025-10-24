# CapstoneSaleForecast

Sales Forecasting System that predicts future sales using historical sales data categorized by Region → Area → Territory at Year–Month granularity.

## Architecture

```
src/
  sales_forecast_app/
    app.py                 # Flask app with /health, /status, /train, /forecast
    wsgi.py                # WSGI entrypoint for Gunicorn
    config.py              # Env-driven configuration
    database/
      connection.py        # SQLAlchemy session factory (SQL Server via pyodbc)
      queries.py           # Sales aggregation queries
    services/
      training_service.py  # Training pipeline (Prophet per key)
      forecast_service.py  # Forecasting pipeline
    models/
      prophet_model.py     # Prophet wrapper
      lightgbm_model.py    # LightGBM wrapper (for hybrid/feature models)
    utils/
      preprocessing.py     # Data prep to ds/y schema
      features.py          # Lag/time features (for boosting models)
      evaluation.py        # RMSE/MAE/MAPE
requirements.txt
.env.example
Dockerfile
gunicorn.conf.py
```

## Data Expectations

- Table: `SalesHistory`
- Columns: `Region`, `Area`, `Territory`, `Year`, `Month`, `SalesAmount`
- If your schema differs, update `src/sales_forecast_app/database/queries.py` accordingly.

## Setup (Local)

1. Create and activate a virtual environment.
   - Windows (PowerShell):
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Install dependencies:
     ```powershell
     pip install -r requirements.txt
     ```
2. Configure environment variables:
   - Copy `.env.example` to `.env` and set values:
     - `SQLSERVER_SERVER`, `SQLSERVER_DATABASE`
     - Either `SQLSERVER_USER`/`SQLSERVER_PASSWORD` or leave empty for Windows Integrated Auth
     - `SQLSERVER_DRIVER` e.g. "ODBC Driver 17 for SQL Server" (Windows) or 18 in Docker
3. Ensure Microsoft ODBC Driver is installed on your machine.

## Run the API (Local Dev)

- Option A (module):
  ```powershell
  $env:PYTHONPATH = "src"
  python -m sales_forecast_app.app
  ```
- Option B (script):
  ```powershell
  python .\src\sales_forecast_app\app.py
  ```

The API starts on http://localhost:8000

## Endpoints

- `GET /health` → `{ status: "ok" }`
- `GET /status` → model/artifact readiness
- `POST /train`
  - Body: `{ "scope": "territory|area|region", "horizon": 6 }`
  - Trains Prophet models per key and stores artifacts under `ARTIFACTS_DIR`
- `POST /forecast`
  - Body: `{ "scope": "territory|area|region", "horizon": 6, "filters": { "Region": "R1", "Area": "A1", "Territory": "T1" } }`
  - Uses saved artifacts to forecast the next N months

### Example Requests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"scope":"territory","horizon":6}'

curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"scope":"territory","horizon":6,"filters":{"Region":"North"}}'
```

## Docker (Production-style)

```bash
docker build -t capstone-sales-forecast .
docker run --rm -p 8000:8000 --env-file .env capstone-sales-forecast
```

The container uses `gunicorn` with `sales_forecast_app.wsgi:app`.

## Notes and Next Steps

- Prophet models are trained per key group; LightGBM/XGBoost scaffolding exists for hybrid modeling.
- Add cross-validation and error logging as needed.
- Schedule retraining (cron/Airflow) by calling `POST /train` periodically.