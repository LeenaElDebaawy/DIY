from fastapi import APIRouter

from app.api import analysis, auth, cases, chat, health, transcribe, uploads

# Health routes stay at the root level
router = APIRouter()
router.include_router(health.router)

# All main API routes go under /api/v1
v1 = APIRouter(prefix="/api/v1")
v1.include_router(auth.router)      # /api/v1/auth/...
v1.include_router(cases.router)     # /api/v1/cases/...
v1.include_router(uploads.router)   # /api/v1/cases/{case_id}/uploads/...
v1.include_router(analysis.router)  # /api/v1/cases/{case_id}/analyse  (POST trigger)
                                     # /api/v1/cases/{case_id}/analysis (GET result)
v1.include_router(chat.router)       # /api/v1/cases/{case_id}/chat     (GET + POST)
v1.include_router(transcribe.router) # /api/v1/transcribe               (POST audio→text)

router.include_router(v1)
