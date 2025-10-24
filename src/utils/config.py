import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class AppConfig:
    sqlserver_server: str
    sqlserver_database: str
    sqlserver_user: str | None
    sqlserver_password: str | None
    sqlserver_driver: str
    sqlalchemy_echo: bool
    artifacts_dir: str

    @staticmethod
    def from_env() -> "AppConfig":
        # Load .env values into process environment (no-op if already loaded)
        load_dotenv()
        return AppConfig(
            sqlserver_server=os.getenv("SQLSERVER_SERVER", "localhost"),
            sqlserver_database=os.getenv("SQLSERVER_DATABASE", "salesdb"),
            sqlserver_user=os.getenv("SQLSERVER_USER"),
            sqlserver_password=os.getenv("SQLSERVER_PASSWORD"),
            sqlserver_driver=os.getenv("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server"),
            sqlalchemy_echo=os.getenv("SQLALCHEMY_ECHO", "0") == "1",
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        )
