# config.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

# PostgreSQL settings
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "firestream_db")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Elasticsearch settings
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_SCHEME = os.getenv("ES_SCHEME", "http")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    """
    Dependency to get a DB session.
    Usage (e.g. in FastAPI): 
        db = next(get_db())
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Elasticsearch client
es_client = Elasticsearch(
    hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": ES_SCHEME}],
    timeout=30,
    max_retries=3,
    retry_on_timeout=True,
)
