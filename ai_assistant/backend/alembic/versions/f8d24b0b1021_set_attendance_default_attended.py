"""set attendance default to attended

Revision ID: f8d24b0b1021
Revises: 0e2b_store_credit_on_ledger
Create Date: 2025-02-05
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "f8d24b0b1021"
down_revision = "0e2b_store_credit_on_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE appointments ALTER COLUMN attendance_status SET DEFAULT 'attended';"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE appointments ALTER COLUMN attendance_status SET DEFAULT 'unknown';"
    )
