"""add feedback table

Revision ID: f1a2b3c4d5e6
Revises: b4e7f2a9c0d1
Create Date: 2026-06-16

Adds the `feedback` table for the Phase 1 feedback / data-collection mechanism.

Design notes
────────────
• Purely additive: a new table referencing existing rows. No existing table is
  altered, so existing flows are unaffected.

• FKs use ondelete=SET NULL so feedback rows are never silently destroyed by a
  re-analysis cleanup or a case deletion. (Deleting a case that carries SAFETY
  feedback is separately blocked at the application layer to preserve evidence.)

• output_mode and is_safety are snapshot at creation time by the application so
  analytics queries do not need to read the analyses JSONB.

Rollback
────────
Downgrade drops the table. No dependent objects.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'b4e7f2a9c0d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'feedback',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('session_id', sa.Integer(),
                  sa.ForeignKey('sessions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('analysis_id', sa.Integer(),
                  sa.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True),
        sa.Column('user_id', sa.Integer(),
                  sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('rating', sa.String(length=20), nullable=False),
        sa.Column('issue_category', sa.String(length=50), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('output_mode', sa.String(length=30), nullable=True),
        sa.Column('is_safety', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('safety_category', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='new'),
        sa.Column('resolution', sa.String(length=30), nullable=True),
        sa.Column('reviewer_note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    # Index the columns the dashboard filters/aggregates on most.
    op.create_index('ix_feedback_is_safety', 'feedback', ['is_safety'])
    op.create_index('ix_feedback_output_mode', 'feedback', ['output_mode'])
    op.create_index('ix_feedback_created_at', 'feedback', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_feedback_created_at', table_name='feedback')
    op.drop_index('ix_feedback_output_mode', table_name='feedback')
    op.drop_index('ix_feedback_is_safety', table_name='feedback')
    op.drop_table('feedback')
