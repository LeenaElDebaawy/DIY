import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Email is the login identifier and must be unique
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Display name for the user
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Password is stored only as a hash, never as plain text
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Account plan, default is free
    plan: Mapped[str] = mapped_column(String(50), default="free", nullable=False)

    # Timestamp created automatically by the database
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # One user can have many active tokens
    tokens: Mapped[list["UserToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    # One user can have many sessions
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


# ---------------------------------------------------------------------------
# UserToken
# ---------------------------------------------------------------------------

class UserToken(Base):
    __tablename__ = "user_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Bearer token stored in the database
    token: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    # Link token to its user
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Timestamp created automatically by the database
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to the user
    user: Mapped["User"] = relationship(back_populates="tokens")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Tracks where the session is in the flow
    status: Mapped[str] = mapped_column(
        String(50), default="created", nullable=False
    )

    # Counts how many clarification rounds happened
    clarification_round: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Null for anonymous sessions, set for authenticated users
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Filled later from AI result
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Useful for debugging or rate limiting later
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp created automatically by the database
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Timestamp updated whenever the row changes
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Optional relationship back to user
    user: Mapped[Optional["User"]] = relationship(back_populates="sessions")

    # Delete child rows automatically if the session is deleted
    uploads: Mapped[list["Upload"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Link upload to its session
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    # Original file name from the user
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # MIME type such as image/jpeg
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # File size in bytes
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # File path on disk
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Image dimensions
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamp created automatically by the database
    uploaded_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to session
    session: Mapped["Session"] = relationship(back_populates="uploads")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Link analysis to its session
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    # Tracks analysis state
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )

    # Optional user description
    text_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Which provider and model generated this analysis
    provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Raw AI response before parsing
    raw_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Final parsed result shown to the frontend
    structured_result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timing and token tracking
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Retry and validation tracking
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    validation_issues: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Raw outputs from stage 1, 2, and 3
    stage_outputs: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Current pipeline stage for frontend polling
    pipeline_stage: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Timestamp created automatically by the database
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to session
    session: Mapped["Session"] = relationship(back_populates="analyses")


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Link message to its session
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    # user or assistant
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    # Message text
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # text or analysis_summary
    message_type: Mapped[str] = mapped_column(
        String(50), default="text", nullable=False
    )

    # Optional extra metadata
    msg_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Timestamp created automatically by the database
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to session
    session: Mapped["Session"] = relationship(back_populates="messages")