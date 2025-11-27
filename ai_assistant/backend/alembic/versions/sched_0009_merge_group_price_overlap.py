"""Merge heads: group price + drop overlap index

Revision ID: sched_0009_merge_group_price_overlap
Revises: sched_0008_owner_group_price, sched_0008_drop_overlap_excl
Create Date: 2025-10-01
"""

revision = "sched_0009_merge_group_price_overlap"
down_revision = ("sched_0008_owner_group_price", "sched_0008_drop_overlap_excl")
branch_labels = None
depends_on = None


def upgrade() -> None:
    # No-op merge
    pass


def downgrade() -> None:
    # No-op; splitting heads is not supported automatically
    pass
