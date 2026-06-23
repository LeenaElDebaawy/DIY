"""add video fields to uploads

Revision ID: e5f6a7b8c9d0
Revises: f1a2b3c4d5e6
Create Date: 2026-06-21

Adds three columns to the uploads table to support video uploads alongside the
existing image columns.  All existing image rows receive safe defaults and are
never altered otherwise.

New columns
───────────
  source_type       VARCHAR(10)  NOT NULL  DEFAULT 'image'
                    Discriminator: 'image' (all pre-migration rows) or 'video'.
                    server_default means PostgreSQL assigns it immediately on
                    ALTER TABLE — no UPDATE backfill is needed.

  duration_seconds  FLOAT        NULLABLE
                    Populated from the video container header by the upload
                    service using PyAV.  NULL for all image rows.

  thumbnail_paths   JSONB        NULLABLE
                    List of filename strings (not full paths) for the JPEG
                    frames extracted from the video at 10 %, 50 %, 90 % of
                    its duration.  NULL for all image rows.

Design notes
────────────
• Purely additive: three new columns on an existing table.  No column is
  removed, renamed, or type-changed.  All existing query paths and ORM
  callsites that do not name these columns are unaffected.

• thumbnail_paths stores filenames only (e.g. ["abc123_thumb_1.jpg"]).
  The serving path /static/uploads/ is prepended by the frontend, keeping
  the DB value URL-scheme-agnostic.

• No index is added: video upload queries use the existing session_id /
  uploaded_at indices.  If video volume grows, an index on source_type can
  be added in a later migration.

Rollback
────────
Downgrade drops all three columns.  No FK, no index on these columns,
no dependent DB objects.  The companion ORM-model and service changes
must be reverted at the same time.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "uploads",
        sa.Column(
            "source_type",
            sa.String(length=10),
            nullable=False,
            server_default="image",
        ),
    )
    op.add_column(
        "uploads",
        sa.Column(
            "duration_seconds",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "uploads",
        sa.Column(
            "thumbnail_paths",
            JSONB(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("uploads", "thumbnail_paths")
    op.drop_column("uploads", "duration_seconds")
    op.drop_column("uploads", "source_type")
