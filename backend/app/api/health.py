from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db

# Create a small router just for the health related routes
router = APIRouter()


@router.get("/health")
def health_check():
    # Simple route to confirm the backend is running
    return {
        "status": "ok",
        "message": "Fixly backend is running",
    }


@router.get("/health/db")
def db_check(db: Session = Depends(get_db)):
    # Run a very small SQL query to confirm the database connection works
    db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "message": "Database connection working",
    }