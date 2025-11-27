"""Merge admin fee branch with attendance updates

Revision ID: 9f4d25c8e6ab
Revises: 7a1b59b2a3cc, 2de73ae08490
Create Date: 2025-02-05
"""

# This merge resolves parallel heads created by admin fee work and attendance updates.

revision = "9f4d25c8e6ab"
down_revision = ("7a1b59b2a3cc", "2de73ae08490")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
