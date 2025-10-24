"""
SQLAlchemy Database Manager for MSSQL Server Integration (raw SQL usage)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

try:
    # Optional type import to avoid circulars at runtime
    from utils.config import AppConfig  # type: ignore
except Exception:  # pragma: no cover
    AppConfig = None  # type: ignore


# Database connection and session management
class DatabaseManager:

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(
                self.connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self):
        """Get a database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()


# Global database manager instance (will be initialized in config)
db_manager: DatabaseManager = None # type: ignore


def create_session_factory(config: "AppConfig"):
    """Create and return a SQLAlchemy sessionmaker based on AppConfig.

    Uses Windows Integrated Authentication when username/password are absent.
    """
    driver = quote_plus(config.sqlserver_driver)
    if config.sqlserver_user and config.sqlserver_password:
        user = quote_plus(config.sqlserver_user)
        pwd = quote_plus(config.sqlserver_password)
        db_url = (
            f"mssql+pyodbc://{user}:{pwd}@{config.sqlserver_server}/{config.sqlserver_database}?driver={driver}"
        )
    else:
        db_url = (
            f"mssql+pyodbc://@{config.sqlserver_server}/{config.sqlserver_database}?driver={driver}&trusted_connection=yes"
        )

    engine = create_engine(
        db_url,
        echo=bool(getattr(config, "sqlalchemy_echo", False)),
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
