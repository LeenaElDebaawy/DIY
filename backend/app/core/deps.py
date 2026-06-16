from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.services.auth_service import get_user_by_token

# HTTPBearer reads the Authorization: Bearer <token> header.
# auto_error=False means we handle missing headers ourselves below
# instead of letting FastAPI return a generic 403.
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    user = await get_user_by_token(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    return user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Allow only allow-listed admins (User.is_admin). Server-side enforcement —
    the frontend nav/redirect is UX only. Returns 403 for authenticated
    non-admins (401 for missing/invalid auth is handled by get_current_user)."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )
    return current_user
