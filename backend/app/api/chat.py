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
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.chat import MessageRequest, MessageResponse, SpawnCaseRequest, SpawnCaseResponse
from app.services.chat_service import (
    get_messages,
    send_message,
    spawn_case_from_image,
    stream_message,
)

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
    return await send_message(db, case_id, current_user.id, body.content, body.image_ids)


@router.post("/stream")
async def post_message_stream(
    case_id: int,
    body: MessageRequest,
    current_user: User = Depends(get_current_user),
):
    """Stream the assistant reply token-by-token as plain text for a live 'typing'
    effect. The stream ends with a '\\x1e<json>' sentinel carrying the follow-up
    verdict metadata. stream_message manages its own DB session so persistence
    survives the whole stream."""
    return StreamingResponse(
        stream_message(case_id, current_user.id, body.content, body.image_ids),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/spawn-case", response_model=SpawnCaseResponse)
async def spawn_case(
    case_id: int,
    body: SpawnCaseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a NEW case from a photo attached in chat (copies the photo + kicks off analysis)."""
    new_id = await spawn_case_from_image(
        db, case_id, current_user.id, body.image_ids, body.apparent_trade
    )
    return SpawnCaseResponse(case_id=new_id)
