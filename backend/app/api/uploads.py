from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.uploads import UploadResponse
from app.services.uploads_service import list_uploads, save_upload

router = APIRouter(prefix="/cases/{case_id}/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_file(
    case_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a photo to a case. The case must belong to the logged-in user."""
    return await save_upload(db, case_id, current_user.id, file)


@router.get("", response_model=list[UploadResponse])
async def list_case_uploads(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all uploads for a case owned by the logged-in user."""
    return await list_uploads(db, case_id, current_user.id)
