from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import _bearer, get_current_user
from app.database import get_db
from app.models import User
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserResponse
from app.services.auth_service import login_user, logout_user, register_user

# This router is mounted under /api/v1 in router.py,
# so the full paths become /api/v1/auth/register etc.
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user, token = await register_user(db, body.email, body.name, body.password)
    return AuthResponse(user=UserResponse.model_validate(user), token=token)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user, token = await login_user(db, body.email, body.password)
    return AuthResponse(user=UserResponse.model_validate(user), token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.post("/logout", status_code=204)
async def logout(
    credentials=Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    if credentials:
        await logout_user(db, credentials.credentials)
    return Response(status_code=204)
