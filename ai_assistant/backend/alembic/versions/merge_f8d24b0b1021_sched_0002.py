"""Merge scheduling and payments branches

Revision ID: 7a1b59b2a3cc
Revises: f8d24b0b1021, sched_0002
Create Date: 2025-02-05
"""

# revision identifiers, used by Alembic.
revision = "7a1b59b2a3cc"
down_revision = ("f8d24b0b1021", "sched_0002")
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Pure merge revision; no runtime operations required.
    pass


def downgrade() -> None:
    # Pure merge revision; nothing to roll back.
    pass
