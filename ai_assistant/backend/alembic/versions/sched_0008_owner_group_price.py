"""Add group_price_60_cents to auth.User

Revision ID: sched_0008_owner_group_price
Revises: sched_0007_group_lessons
Create Date: 2025-01-20
"""

from alembic import op
import sqlalchemy as sa


revision = "sched_0008_owner_group_price"
down_revision = "sched_0007_group_lessons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    try:
        op.add_column(
            "User",
            sa.Column("group_price_60_cents", sa.Integer(), nullable=True),
            schema="auth",
        )
    except Exception:
        # Fallback for non-PG dev backends without schemas
        op.add_column(
            "User", sa.Column("group_price_60_cents", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    try:
        op.drop_column("User", "group_price_60_cents", schema="auth")
    except Exception:
        op.drop_column("User", "group_price_60_cents")
