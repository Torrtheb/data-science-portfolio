"""add amount_paid_cents to appointments

Revision ID: 9511eb2b5896
Revises: 08e6328f2c09
Create Date: 2025-09-18 11:24:01.761164

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9511eb2b5896"
down_revision: Union[str, Sequence[str], None] = "08e6328f2c09"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        "appointments",
        sa.Column("amount_paid_cents", sa.Integer(), nullable=True),
    )

    # (Optional) if you want to initialize it for old rows, add one of these:
    # 1) set to 0 wherever payment_status='paid' and amount is null
    # op.execute("UPDATE appointments SET amount_paid_cents = 0 WHERE payment_status = 'paid' AND amount_paid_cents IS NULL")
    #
    # 2) or leave NULLs as-is (recommended unless you have a known default)


def downgrade():
    op.drop_column("appointments", "amount_paid_cents")
