import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    mime_type: str
    file_size_bytes: int
    storage_path: str           # filename only, e.g. "abc123.jpg" or "abc123.mp4"
    source_type: str = "image"  # "image" | "video"
    width: Optional[int]
    height: Optional[int]
    duration_seconds: Optional[float]
    thumbnail_paths: Optional[list[str]]
    uploaded_at: datetime.datetime
