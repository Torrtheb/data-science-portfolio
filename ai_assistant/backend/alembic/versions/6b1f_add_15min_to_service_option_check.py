"""
Add 15 minutes to service_options.duration_minutes check constraint.

Revision ID: 6b1f_add_15min
Revises: bd3f7bde6d9c
Create Date: 2025-09-29
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "6b1f_add_15min"
down_revision = "bd3f7bde6d9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old constraint and add new one with 15 included
    op.execute(
        "ALTER TABLE service_options DROP CONSTRAINT IF EXISTS ck_duration_allowed;"
    )
    op.execute(
        """
        ALTER TABLE service_options
        ADD CONSTRAINT ck_duration_allowed
        CHECK (duration_minutes IN (15,30,45,60,120));
        """
    )


def downgrade() -> None:
    # Revert back to without 15
    op.execute(
        "ALTER TABLE service_options DROP CONSTRAINT IF EXISTS ck_duration_allowed;"
    )
    op.execute(
        """
        ALTER TABLE service_options
        ADD CONSTRAINT ck_duration_allowed
        CHECK (duration_minutes IN (30,45,60,120));
        """
    )
