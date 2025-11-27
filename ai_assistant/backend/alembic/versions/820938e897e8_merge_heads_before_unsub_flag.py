"""merge heads before unsub flag

Revision ID: 820938e897e8
Revises: sched_0011_client_email_unsubscribed
Create Date: 2025-11-24 16:22:39.636834

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "820938e897e8"
down_revision: Union[str, Sequence[str], None] = (
    "81b2_time_indexes",
    "sched_0011_client_email_unsubscribed",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
