"""add reason_chips to feedback

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-06-28

Adds an optional JSONB column `reason_chips` to the feedback table.
Stores the multi-select reason chips a user picks when rating "Partially"
or "No". Old rows default to NULL (treated as an empty list in the API).

Rollback
────────
Downgrade drops the column with no data loss to other columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'feedback',
        sa.Column('reason_chips', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('feedback', 'reason_chips')
