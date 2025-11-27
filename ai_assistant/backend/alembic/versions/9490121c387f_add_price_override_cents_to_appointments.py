"""add price_override_cents to appointments

Revision ID: 9490121c387f
Revises: 96e4f9179eee
Create Date: 2025-09-19 09:35:48.779311

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9490121c387f"
down_revision: Union[str, Sequence[str], None] = "96e4f9179eee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        "appointments",
        sa.Column("price_override_cents", sa.Integer(), nullable=True),
    )


def downgrade():
    op.drop_column("appointments", "price_override_cents")
