"""Add composite indexes for timeoffs and special openings

Revision ID: 81b2_time_indexes
Revises: 44f0a2380526
Create Date: 2025-10-01
"""

from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "81b2_time_indexes"
down_revision = "sched_0010_merge_all_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # timeoffs(owner_id, start_utc, end_utc)
    op.create_index(
        "ix_timeoff_owner_start_end",
        "timeoffs",
        ["owner_id", "start_utc", "end_utc"],
        unique=False,
    )
    # special_openings(owner_id, start_utc, end_utc)
    op.create_index(
        "ix_special_open_owner_start_end",
        "special_openings",
        ["owner_id", "start_utc", "end_utc"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_special_open_owner_start_end", table_name="special_openings")
    op.drop_index("ix_timeoff_owner_start_end", table_name="timeoffs")
