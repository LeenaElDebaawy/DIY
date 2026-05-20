import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CaseCreate(BaseModel):
    """Fields the client must send when opening a new repair case.

    title is optional so the quick-start flow can create a case before
    the AI has generated a proper title from the uploaded photos.
    The pipeline overwrites title and category automatically after Stage 1.
    """

    title: Optional[str] = "Diagnosing…"
    category: Optional[str] = None
    description: Optional[str] = None


class CasePatch(BaseModel):
    """Partial update — only fields that are not None are written."""

    title: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None


class CaseResponse(BaseModel):
    """Shape of a case returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: Optional[str]
    category: Optional[str]
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
