import datetime
import os
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _admin_emails() -> set[str]:
    """Allow-listed admin emails from the ADMIN_EMAILS env var (comma-separated).
    Read at call time so changing the env does not require a process restart in
    dev. Lower-cased to match the normalised emails stored on users."""
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


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

    # Account plan, default is free.
    # server_default sets the default at the PostgreSQL level so the column
    # can never be NULL even if the ORM omits the value on insert.
    plan: Mapped[str] = mapped_column(
        String(50), default="free", server_default="free", nullable=False
    )

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

    @property
    def is_admin(self) -> bool:
        """True when this user's email is in the ADMIN_EMAILS allow-list.

        Computed (no DB column, no migration) so admin access is config-driven
        for the MVP. Mirrors the computed-property pattern used elsewhere
        (Session.output_mode, Analysis.hazard_ack). Enforced server-side via the
        require_admin dependency; the frontend only uses it to show/hide the nav.
        """
        return bool(self.email and self.email.strip().lower() in _admin_emails())


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

    # Tracks where the session is in the flow.
    # "open" is the canonical starting state used everywhere in the app.
    # The old default "created" is retired, kept consistent with the service layer.
    status: Mapped[str] = mapped_column(
        String(50), default="open", nullable=False
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

    # ── Hazardous-case acknowledgement ────────────────────────────────────────
    # Set to True (with timestamp) when the user clicks "I Understand" on the
    # hazard acknowledgement card shown for assessment_first cases.
    # Stored permanently — not session-scoped — so it survives page reloads
    # and forms the seed event for the P5 hazard_events audit trail.
    hazard_ack: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    hazard_ack_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
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

    @property
    def output_mode(self) -> Optional[str]:
        """
        Output mode of the latest completed analysis ('diy', 'assessment_first',
        'clarification', 'care', 'verify_first').

        Derived at read-time from Analysis.structured_result JSONB so no extra
        column is needed.  Returns None for cases with no completed analysis.

        Requires the analyses relationship to be pre-loaded (selectinload).
        Returns None gracefully if the relationship is not loaded.
        """
        try:
            rows = list(getattr(self, "analyses", None) or [])
        except Exception:
            return None
        for analysis in sorted(rows, key=lambda a: a.id, reverse=True):
            if analysis.status == "complete" and analysis.structured_result:
                v = (analysis.structured_result or {}).get("output_mode")
                if isinstance(v, str):
                    return v
        return None

    @property
    def confidence(self) -> Optional[int]:
        """
        AI confidence score (0-100) extracted from the latest completed analysis.

        Derived at read-time from Analysis.structured_result JSONB so no extra
        column is needed.  Returns None for cases that have no completed analysis
        yet (e.g. still running, failed, or never analysed).

        Requires the `analyses` relationship to be pre-loaded (selectinload) before
        this property is accessed.  If the relationship is not loaded (e.g. on routes
        that return a freshly created or patched case without a selectinload), the
        try/except ensures we return None gracefully instead of raising MissingGreenlet.
        """
        try:
            rows = list(getattr(self, "analyses", None) or [])
        except Exception:
            # SQLAlchemy raises MissingGreenlet when a lazy-load is attempted inside
            # an async context.  Return None rather than propagating the error.
            return None
        for analysis in sorted(rows, key=lambda a: a.id, reverse=True):
            if analysis.status == "complete" and analysis.structured_result:
                s1 = (analysis.structured_result or {}).get("stage_1") or {}
                v = s1.get("confidence")
                if isinstance(v, int):
                    return v
        return None


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

    @property
    def hazard_ack(self) -> bool:
        """
        Delegate to the owning session's hazard_ack flag.

        Used by AnalysisResponse (from_attributes=True) so the frontend
        receives the case-level acknowledgement state alongside the analysis
        data — no extra API call required.

        Requires the session relationship to be pre-loaded (selectinload)
        before this property is accessed.  Returns False gracefully if the
        relationship is not loaded or if any error occurs.
        """
        try:
            sess = getattr(self, "session", None)
            return bool(sess.hazard_ack) if sess is not None else False
        except Exception:
            return False


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


# ---------------------------------------------------------------------------
# Feedback  (Phase 1 — feedback / data-collection mechanism)
# ---------------------------------------------------------------------------

class Feedback(Base):
    """User feedback on a completed analysis.

    Additive table — references existing rows, never mutates them. output_mode
    and is_safety are SNAPSHOT at creation so dashboard queries stay simple and
    are not affected if the case is later re-analysed. Safety-flagged rows carry
    a small review lifecycle (status + resolution + reviewer_note); ordinary
    feedback simply feeds aggregates.
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Links to the case, the specific analysis, and the user who gave feedback.
    # ondelete=SET NULL on analysis so a re-analysis cleanup never orphans the row;
    # session/user use SET NULL so feedback survives even if a case is removed
    # (the case-deletion guard separately blocks deleting cases with safety rows).
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    analysis_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Core feedback. rating: 'helpful' | 'partial' | 'not'
    rating: Mapped[str] = mapped_column(String(20), nullable=False)
    issue_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Snapshot of the analysis context at feedback time (for trivial analytics).
    output_mode: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    # Safety lane. is_safety is auto-flagged (by mode or by safety issue type).
    is_safety: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    safety_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Review lifecycle (used for safety incidents): 'new' | 'reviewing' | 'resolved'.
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="new", server_default="new"
    )
    # Resolution reason when status == 'resolved': fixed | accepted_limitation | dismissed
    resolution: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )