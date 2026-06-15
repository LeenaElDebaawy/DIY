"""add hazard_ack to sessions

Revision ID: b4e7f2a9c0d1
Revises: c17eeef6f32c
Create Date: 2026-06-10

Adds two columns to the sessions table to permanently record whether the user
has acknowledged the hazardous-case safety notice for each repair case:

  hazard_ack     BOOLEAN  NOT NULL  DEFAULT false
  hazard_ack_at  TIMESTAMPTZ        NULL

Design notes
────────────
• server_default='false' is used (not just SQLAlchemy's Python-side default)
  so PostgreSQL assigns the value to every existing row immediately when
  ALTER TABLE executes — no separate UPDATE backfill is needed.

• hazard_ack_at is nullable; NULL means "not yet acknowledged".
  PostgreSQL assigns NULL to all existing rows automatically.

• All existing rows receive hazard_ack=false / hazard_ack_at=NULL, which is
  correct: assessment-first cases that existed before this migration will show
  the acknowledgement card on next visit and require one click. DIY,
  clarification, care, and verify-first cases will never have these fields
  read or written — the values are harmless.

Rollback
────────
Downgrade removes both columns. No FK, no index, no dependent objects.
The three components that depend on these columns (this migration, the ORM
model update, and the API/frontend changes) must be reverted together.

P5 integration point
────────────────────
When hazard_events is implemented, the HAZARD_ACK event will be seeded from
hazard_ack_at. No data is lost: the timestamp already recorded here becomes
the event's occurred_at value.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'b4e7f2a9c0d1'
down_revision: Union[str, None] = 'c17eeef6f32c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'sessions',
        sa.Column(
            'hazard_ack',
            sa.Boolean(),
            nullable=False,
            server_default='false',
        ),
    )
    op.add_column(
        'sessions',
        sa.Column(
            'hazard_ack_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('sessions', 'hazard_ack_at')
    op.drop_column('sessions', 'hazard_ack')
