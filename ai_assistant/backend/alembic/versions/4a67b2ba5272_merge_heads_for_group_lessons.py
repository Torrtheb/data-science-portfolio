"""merge heads for group lessons

Revision ID: 4a67b2ba5272
Revises: 7c21d2d3a4b1, sched_0007_group_lessons
Create Date: 2025-09-30 11:03:06.119838

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "4a67b2ba5272"
down_revision: Union[str, Sequence[str], None] = (
    "7c21d2d3a4b1",
    "sched_0007_group_lessons",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
