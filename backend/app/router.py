from fastapi import APIRouter
from app.api import health

router = APIRouter()

router.include_router(health.router)