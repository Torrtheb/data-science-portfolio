"""add appointment notes columns

Revision ID: 691c1131a629
Revises: 190bdf4f2cbe
Create Date: 2025-09-17 17:34:05.499651

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "691c1131a629"
down_revision: Union[str, Sequence[str], None] = "190bdf4f2cbe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Table is "appointments" with default schema (no schema param)
    op.add_column(
        "appointments", sa.Column("client_change_note", sa.Text(), nullable=True)
    )
    op.add_column("appointments", sa.Column("cancel_reason", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("appointments", "cancel_reason")
    op.drop_column("appointments", "client_change_note")
