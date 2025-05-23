from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import os
from urllib.parse import quote_plus

_engine = None
_Session = None

def load_db_credentials():
    return {
        "username": os.getenv("DB_USERNAME", "default_user"),
        "password": os.getenv("DB_PASSWORD", "default_password"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "1433"),
        "database_name": os.getenv("DB_NAME", "Car Store DB")
    }

def initialize_engine_session():
    global _engine, _Session
    creds = load_db_credentials()
    encoded_password = quote_plus(creds["password"])
    sql_server_uri = (
        f"mssql+pymssql://{creds['username']}:{encoded_password}@"
        f"{creds['host']}:{creds['port']}/{creds['database_name']}"
    )

    engine = create_engine(
        sql_server_uri,
        poolclass=QueuePool,
        pool_size=100,
        max_overflow=100,
        pool_timeout=20,
        pool_recycle=1800,
        pool_pre_ping=True
    )
    
    _engine = engine
    _Session = scoped_session(sessionmaker(bind=engine))

def get_engine():
    global _engine
    if _engine is None:
        initialize_engine_session()
    return _engine

def get_session():
    global _Session
    if _Session is None:
        initialize_engine_session()
    return _Session()
