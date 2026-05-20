"""
Chat API routes.

GET  /api/v1/cases/{case_id}/chat
    Return the full message history for a case, oldest first.
    Used on initial page load to re-hydrate the thread.

POST /api/v1/cases/{case_id}/chat
    Send a user message and receive the AI reply.
    Both messages are persisted to the DB.
    Returns only the assistant reply (the user message is implied from the request body).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.chat import MessageRequest, MessageResponse
from app.services.chat_service import get_messages, send_message

router = APIRouter(prefix="/cases/{case_id}/chat", tags=["chat"])


@router.get("", response_model=list[MessageResponse])
async def list_messages(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all messages for this case, oldest first."""
    return await get_messages(db, case_id, current_user.id)


@router.post("", response_model=MessageResponse)
async def post_message(
    case_id: int,
    body: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message in the case chat thread.

    Persists the user message, calls the AI with full case context,
    persists the reply, and returns the assistant message.
    """
    return await send_message(db, case_id, current_user.id, body.content)
