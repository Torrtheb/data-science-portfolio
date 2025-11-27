"""add service-options table

Revision ID: 96e4f9179eee
Revises: db2e71427007
Create Date: 2025-09-18 12:28:56.255339

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "96e4f9179eee"
down_revision: Union[str, Sequence[str], None] = "db2e71427007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "service_options",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.String(), nullable=False, index=True),
        sa.Column("duration_minutes", sa.SmallInteger(), nullable=False),
        sa.Column("price_cents", sa.Integer(), nullable=False),
        sa.Column(
            "currency", sa.String(length=10), nullable=False, server_default="USD"
        ),
        sa.Column("is_active", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.UniqueConstraint("owner_id", "duration_minutes", name="uq_owner_duration"),
        sa.CheckConstraint(
            "duration_minutes IN (30,45,60,120)", name="ck_duration_allowed"
        ),
        sa.CheckConstraint("price_cents >= 0", name="ck_price_nonneg"),
    )
    # If you're using schemas, be sure to set schema names accordingly.


def downgrade():
    op.drop_table("service_options")
