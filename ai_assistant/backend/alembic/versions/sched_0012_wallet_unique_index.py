"""Add partial unique index for active wallets per owner/client

Revision ID: sched_0012_wallet_unique_index
Revises: 820938e897e8
Create Date: 2025-02-15
"""

from alembic import op
import sqlalchemy as sa


revision = "sched_0012_wallet_unique_index"
down_revision = "820938e897e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_wallet_active_owner_client",
        "prepaid_bundles",
        ["owner_id", "client_id"],
        unique=True,
        postgresql_where=sa.text("total_credits = 0 AND status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_wallet_active_owner_client", table_name="prepaid_bundles")
