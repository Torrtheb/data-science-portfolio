"""Relax owner start unique constraint to allow rebooking cancelled slots

Revision ID: bd3f7bde6d9c
Revises: 9f4d25c8e6ab
Create Date: 2024-04-22
"""

from alembic import op
import sqlalchemy as sa


revision = "bd3f7bde6d9c"
down_revision = "9f4d25c8e6ab"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_owner_start", "appointments", type_="unique")
    op.create_index(
        "ix_owner_start_active_unique",
        "appointments",
        ["owner_id", "start_utc"],
        unique=True,
        postgresql_where=sa.text("status <> 'canceled'"),
    )


def downgrade() -> None:
    op.drop_index("ix_owner_start_active_unique", table_name="appointments")
    op.create_unique_constraint(
        "uq_owner_start", "appointments", ["owner_id", "start_utc"]
    )
