"""add server_default free to users.plan

Revision ID: c17eeef6f32c
Revises: dd9275c4f79f
Create Date: 2026-05-12

The original migration created users.plan as NOT NULL but without a
PostgreSQL-level DEFAULT clause.  That meant the column existed in the
schema only through SQLAlchemy's Python-side default="free".  This
migration adds the matching server_default so the database itself
enforces the fallback, and back-fills any existing rows that somehow
ended up with NULL (shouldn't happen in practice, but safe to fix).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'c17eeef6f32c'
down_revision: Union[str, None] = 'dd9275c4f79f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Back-fill any rows that somehow have NULL (defensive; should be 0 rows).
    op.execute("UPDATE users SET plan = 'free' WHERE plan IS NULL")

    # 2. Add the PostgreSQL-level DEFAULT so the database enforces it going forward.
    op.alter_column(
        'users',
        'plan',
        existing_type=sa.String(length=50),
        server_default='free',
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'users',
        'plan',
        existing_type=sa.String(length=50),
        server_default=None,
        existing_nullable=False,
    )
