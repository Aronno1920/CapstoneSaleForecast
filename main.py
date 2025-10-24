import os
import uvicorn
from src.api.API import app


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
    reload = os.getenv("API_RELOAD", "true").lower() in ("1", "true", "yes")
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    if reload:
        # Use import string for autoreload
        uvicorn.run("main:app", host=host, port=port, reload=True, log_level=log_level)
    else:
        uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
