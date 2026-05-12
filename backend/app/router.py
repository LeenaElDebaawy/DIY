from fastapi import APIRouter

from app.api import auth, health

# Health routes stay at the root level
router = APIRouter()
router.include_router(health.router)

# All main API routes go under /api/v1
v1 = APIRouter(prefix="/api/v1")
v1.include_router(auth.router)

router.include_router(v1)