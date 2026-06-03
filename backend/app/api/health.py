from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import get_db

# Create a small router just for the health related routes
router = APIRouter()


@router.get("/health")
def health_check():
    # Simple route to confirm the backend is running
    return {
        "status": "ok",
        "message": "Knitra backend is running",
    }


@router.get("/health/db")
async def db_check(db: AsyncSession = Depends(get_db)):
    # Actually await the query so we know the DB connection is real.
    # The previous version was sync and dropped the coroutine silently -
    # it always returned "ok" even when the database was unreachable.
    await db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "message": "Database connection working",
    }