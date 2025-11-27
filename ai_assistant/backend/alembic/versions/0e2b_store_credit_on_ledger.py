"""add amount_cents to prepaid_ledger for store credit

Revision ID: 0e2b_store_credit_on_ledger
Revises: 08e6328f2c09
Create Date: 2025-09-27
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0e2b_store_credit_on_ledger"
down_revision = "52df755b0390"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("prepaid_ledger") as batch:
        batch.add_column(
            sa.Column("amount_cents", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("prepaid_ledger") as batch:
        batch.drop_column("amount_cents")
