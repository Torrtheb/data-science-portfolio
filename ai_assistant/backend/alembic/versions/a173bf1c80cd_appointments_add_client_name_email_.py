"""appointments: add client_name/email; client_accounts.client_user_id nullable

Revision ID: a173bf1c80cd
Revises: 0ac637417f9a
Create Date: 2025-09-25 23:25:45.516200

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a173bf1c80cd"
down_revision: Union[str, Sequence[str], None] = "0ac637417f9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1) appointments: add denormalized identity
    op.add_column("appointments", sa.Column("client_name", sa.String(), nullable=True))
    op.add_column("appointments", sa.Column("client_email", sa.String(), nullable=True))
    op.create_index(
        "ix_appointments_client_email", "appointments", ["client_email"], unique=False
    )

    # 2) client_accounts: make client_user_id nullable
    with op.batch_alter_table("client_accounts") as batch_op:
        batch_op.alter_column(
            "client_user_id", existing_type=sa.String(), nullable=True
        )


def downgrade():
    # 2) client_accounts: revert nullability
    with op.batch_alter_table("client_accounts") as batch_op:
        batch_op.alter_column(
            "client_user_id", existing_type=sa.String(), nullable=False
        )

    # 1) appointments: drop fields
    op.drop_index("ix_appointments_client_email", table_name="appointments")
    op.drop_column("appointments", "client_email")
    op.drop_column("appointments", "client_name")
