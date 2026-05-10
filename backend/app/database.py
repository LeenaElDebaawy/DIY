from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# SQLite database file stored inside the backend folder
DATABASE_URL = "sqlite:///./fixly.db"

# Create the database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}, # needed for SQLite with FastAPI
)
# Create a session factory for database operations
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base class that all database models will inherit from
class Base(DeclarativeBase):
    pass

# Dependency function that gives a database session to route functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()