"""merge all heads

Revision ID: 44f0a2380526
Revises: 4a67b2ba5272, sched_0008_owner_group_price
Create Date: 2025-09-30 11:24:11.285345

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "44f0a2380526"
down_revision: Union[str, Sequence[str], None] = (
    "4a67b2ba5272",
    "sched_0008_owner_group_price",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
