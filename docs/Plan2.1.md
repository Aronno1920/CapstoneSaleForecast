# Sales Forecast Product — Python + FastAPI + Prophet + XGBoost/LightGBM + SQL Server

> Complete implementation guide, architecture, folder structure, and sample code to build a territory/area/region-wise sales forecasting product using Python, FastAPI, Prophet, and tree-based models (LightGBM/XGBoost) with SQL Server as the datastore.

---

## Overview
This document guides you through building a production-ready Sales Forecasting system. It covers:
- Data model and SQL Server schema
- ETL & preprocessing pipeline
- Baseline forecasting with Prophet
- Gradient-boosted models (LightGBM / XGBoost)
- Model training, evaluation, and persistence
- FastAPI service (endpoints: train, predict, status)
- Containerization (Docker), scheduling, and MLOps suggestions


---

## Requirements
- Python 3.10+
- SQL Server (express or full)
- Docker & Docker Compose (optional but recommended)

Python packages (sample `requirements.txt`):
```
fastapi
uvicorn[standard]
pandas
numpy
sqlalchemy
pyodbc
pymssql
prophet
lightgbm
xgboost
scikit-learn
pydantic
joblib
mlflow
alembic
python-dotenv
```

> Note: Prophet installs as `prophet` (or `fbprophet` in older setups). Use the variant that matches your environment.


---

## High-level Architecture

```
+--------------------+      +-----------------+      +----------------+
| SQL Server (Raw)   | ---> | ETL / Features  | ---> | Model Training |
+--------------------+      +-----------------+      +----------------+
        |                                           |
        v                                           v
+--------------------+                      +----------------+
| Forecast DB (SQL)  | <--- FastAPI --->    | Model Store    |
+--------------------+                      +----------------+
        |
        v
+--------------------+
| Dashboard / BI     |
+--------------------+
```

Components:
- **Extraction/ETL**: Pull data from SQL Server, create features and aggregates (region/territory/month).
- **Training module**: Train Prophet (time-series) and LightGBM/XGBoost (tabular with lags & exogenous features).
- **API**: FastAPI exposes endpoints for forecast queries and retraining.
- **Model store**: Filesystem, SQL table, or MLflow to store trained artifacts and metadata.
- **Dashboard**: PowerBI / Streamlit / Metabase for visualization.


---

## Data Model & SQL Schema (recommended)

Tables:
1. `sales_raw` — raw transactional or aggregated monthly sales
```sql
CREATE TABLE sales_raw (
    id INT IDENTITY PRIMARY KEY,
    sale_date DATE NOT NULL,
    region VARCHAR(100),
    territory VARCHAR(100),
    area VARCHAR(100),
    product_id INT,
    sales_amount FLOAT,
    sales_qty FLOAT,
    channel VARCHAR(50)
);
```

2. `sales_agg_monthly` — pre-aggregated per month for faster training
```sql
CREATE TABLE sales_agg_monthly (
    id INT IDENTITY PRIMARY KEY,
    year INT,
    month INT,
    yearmonth CHAR(7), -- e.g. '2023-01'
    region VARCHAR(100),
    territory VARCHAR(100),
    area VARCHAR(100),
    product_id INT,
    total_sales FLOAT,
    total_qty FLOAT
);
```

3. `model_registry` — store metadata for models
```sql
CREATE TABLE model_registry (
    id INT IDENTITY PRIMARY KEY,
    model_name VARCHAR(100),
    model_type VARCHAR(50), -- prophet, lgbm, xgb
    version VARCHAR(50),
    path VARCHAR(500),
    metrics JSON, -- store RMSE, MAE
    trained_at DATETIME
);
```


---

## Folder Structure
```
sales-forecast/
├─ app/
│  ├─ main.py                # FastAPI app
│  ├─ api/
│  │  ├─ endpoints.py       # endpoints (train, predict, status)
│  ├─ db/
│  │  ├─ db.py              # SQLAlchemy engine + session
│  │  ├─ models.py          # ORM models (optional)
│  ├─ services/
│  │  ├─ etl.py             # ETL and feature engineering
│  │  ├─ prophet_train.py   # Prophet training utilities
│  │  ├─ tree_train.py      # LGBM/XGB training utilities
│  │  ├─ inference.py       # prediction helpers
│  ├─ utils/
│  │  ├─ metrics.py         # evaluation metrics
│  │  ├─ persistence.py     # save/load model
├─ notebooks/
├─ scripts/
│  ├─ run_etl.py
│  ├─ train_all.py
├─ docker/
│  ├─ Dockerfile
│  ├─ docker-compose.yml
├─ requirements.txt
├─ README.md
```


---

## ETL & Feature Engineering (app/services/etl.py)
Key steps:
- Pull data from `sales_raw` or `sales_agg_monthly`.
- Ensure continuous months per series (fill missing with zero).
- Create lags: `lag_1`, `lag_3`, `lag_6`, rolling means.
- Add time features: month, quarter, is_month_start, seasonality flags.
- Optionally add external regressors: promotions, holidays, weather, macro.

```python
# app/services/etl.py
import pandas as pd
from sqlalchemy import text

def load_monthly_sales(engine, start_date=None, end_date=None):
    q = "SELECT yearmonth, region, territory, area, product_id, total_sales FROM sales_agg_monthly"
    df = pd.read_sql(q, engine)
    df['ds'] = pd.to_datetime(df['yearmonth'] + '-01')
    df = df.sort_values(['region','territory','area','product_id','ds'])
    return df


def create_lags(df, group_cols, target_col='total_sales', lags=[1,3,6]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    df['rolling_3'] = df.groupby(group_cols)[target_col].shift(1).rolling(3).mean().reset_index(level=group_cols, drop=True)
    return df
```


---

## Prophet (Time-Series) — Baseline
Prophet expects a DataFrame with `ds` (date) and `y` (value). For region/territory-wise forecasts you'll run one Prophet model per series (e.g., per territory or per (region,product) pair).

Example training loop:

```python
# app/services/prophet_train.py
from prophet import Prophet
import pandas as pd

def train_prophet(df_series, yearly_seasonality=True, weekly_seasonality=False):
    m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality)
    # Add holidays or regressors if available
    m.fit(df_series)
    return m


def forecast_prophet(m, periods=6, freq='MS'):
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]
```

**Notes**:
- If you have thousands of time-series, training a Prophet model per series may be costly. Consider grouping or using global models (TFT or deep learning) for scale.


---

## LightGBM / XGBoost (Tabular Approach)
Instead of one model per time series, train a single model that takes series identifiers (region/territory) as categorical features plus lags and time features. This scales much better.

Key ideas:
- Input: `ds, region, territory, area, product_id, month, year, lag_1, lag_3, rolling_3, is_holiday, promo_flag`.
- Target: `total_sales` for the next month.

Sample training code (LightGBM):

```python
# app/services/tree_train.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def train_lgb(train_df, features, target='total_sales'):
    X = train_df[features]
    y = train_df[target]
    # Use a time-based split or custom split
    tscv = TimeSeriesSplit(n_splits=3)
    models = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
        bst = lgb.train(params, dtrain, valid_sets=[dtrain,dval], early_stopping_rounds=50, num_boost_round=1000)
        preds = bst.predict(X_val)
        print('Fold RMSE:', mean_squared_error(y_val, preds, squared=False))
        models.append(bst)
    return models
```

Inference: use recent lags to forecast next month(s). For multi-step forecasting, either re-generate lags iteratively or train direct multi-output models.


---

## Model Persistence & Registry
- Save models with `joblib` or `pickle` to a model directory, and store metadata in `model_registry` table.
- For production-grade, use MLflow: register models, track experiments, and serve models.

```python
# app/utils/persistence.py
import joblib
from pathlib import Path

def save_model(obj, name, version='v1'):
    path = Path('models') / f"{name}_{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return str(path)


def load_model(path):
    return joblib.load(path)
```


---

## FastAPI App
Endpoints to include:
- `POST /train` — triggers a training job (prophet or lgbm)
- `GET /predict?region=&territory=&area=&product_id=&periods=` — returns forecasts
- `GET /models` — list available models and metadata
- `GET /health` — health check

Sample `main.py`:

```python
# app/main.py
from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title='Sales Forecast API')
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
```

Sample endpoints (simplified):

```python
# app/api/endpoints.py
from fastapi import APIRouter, BackgroundTasks
from app.services.etl import load_monthly_sales, create_lags
from app.services.tree_train import train_lgb
from app.services.prophet_train import train_prophet, forecast_prophet
from app.utils.persistence import save_model, load_model
from sqlalchemy import create_engine

router = APIRouter()

@router.get('/health')
async def health():
    return {'status':'ok'}

@router.post('/train')
async def train(model_type: str = 'lgbm', background_tasks: BackgroundTasks = None):
    # Start background training task or run sync
    # For simplicity run sync here (but prefer background tasks or external scheduler)
    engine = create_engine('mssql+pyodbc://...')
    df = load_monthly_sales(engine)
    df = create_lags(df, ['region','territory','area','product_id'])
    if model_type == 'prophet':
        # example train for single series
        sub = df[(df.region=='North') & (df.product_id==1)][['ds','total_sales']].rename(columns={'total_sales':'y'})
        m = train_prophet(sub)
        path = save_model(m, 'prophet_north_p1')
        return {'model_path': path}
    else:
        features = ['month','year','lag_1','lag_3','rolling_3','region','territory']
        models = train_lgb(df.dropna(), features)
        path = save_model(models[-1], 'lgbm_global')
        return {'model_path': path}

@router.get('/predict')
async def predict(region: str, territory: str = None, periods: int = 3):
    # Load latest model and run inference
    model = load_model('models/lgbm_global_v1.pkl')
    # build feature row for last known month and generate predictions
    return {'status':'ok', 'predictions': []}
```

**Important**: For long-running training, use BackgroundTasks, Celery, or an external scheduler. Avoid running heavy training inside the main FastAPI process.


---

## Evaluation & Metrics
- Use time-series cross-validation (rolling origin) for robust evaluation.
- Metrics: RMSE, MAE, MAPE (with caution when near-zero values).
- Keep baseline models (last-year same month, naive average) for benchmarking.


---

## Containerization & Docker
Sample `Dockerfile` (simplified):

```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml` should include your SQL Server container (mcr.microsoft.com/mssql/server) and the app service.


---

## Scheduling & Retraining
- Automate ETL + training using Airflow, Prefect, or simple cron.
- Retrain cadence: monthly/weekly depending on business volatility.
- Implement model drift monitoring: track performance on holdout set and raise alerts.


---

## Monitoring & Observability
- Store predictions and actuals in `forecast_results` table for backtesting.
- Monitor prediction errors per territory/region.
- Log model versions used for each prediction for traceability.


---

## Example: Multi-step forecasting strategy (practical)
1. **One-step direct**: Train model to predict next-month sales (works well, simpler). For forecasting multiple months, iteratively predict month+1, append to features, then predict month+2.
2. **Direct multi-output**: Train models that predict 3/6 month horizons directly using appropriate targets.


---

## Quick Tips & Best Practices
- Always use time-based validation, not random train/test split.
- Keep deterministic preprocessing pipelines (use scikit-learn `Pipeline` if possible).
- Make categorical encodings consistent between training and inference (store encoders).
- When using Prophet with many series, parallelize training using multiprocessing.
- Use LightGBM with categorical features properly (pass `categorical_feature`).


---

## Next Steps I can do for you
1. Create a runnable Git repo with the above structure and example data.
2. Implement the FastAPI endpoints fully (train, predict) with working SQL Server connection string template.
3. Build a notebook that trains Prophet and LightGBM on a sample of your data (you can upload a CSV).
4. Add MLflow tracking and a simple Streamlit dashboard.

Tell me which of the above you'd like me to build next — I can scaffold the code and provide runnable examples.

---

*End of document.*

