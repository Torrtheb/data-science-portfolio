"""client profile: emails table + contact fields

Revision ID: 190bdf4f2cbe
Revises: sched_0002
Create Date: 2025-09-16 16:24:42.570858

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "190bdf4f2cbe"
down_revision: Union[str, Sequence[str], None] = "sched_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Add new columns to client_accounts
    op.add_column("client_accounts", sa.Column("name", sa.String(), nullable=True))
    op.add_column("client_accounts", sa.Column("phone", sa.String(), nullable=True))
    op.add_column(
        "client_accounts", sa.Column("emergency_contact", sa.String(), nullable=True)
    )

    # 2) Create client_emails table
    op.create_table(
        "client_emails",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("client_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(), nullable=False),
        # stored as 1/0 to match your model; you can convert to Boolean later if you prefer
        sa.Column("is_primary", sa.Integer(), nullable=False, server_default="1"),
        sa.UniqueConstraint("account_id", "email", name="uq_account_email"),
    )
    op.create_index("ix_client_emails_account_id", "client_emails", ["account_id"])

    # 3) Backfill a primary email from auth."User".email when available
    # Use a single SQL statement so offline SQL generation works too.
    # This runs at apply-time on the database (safe no-op if no rows match).
    op.execute(
        sa.text(
            """
            INSERT INTO client_emails (account_id, email, is_primary)
            SELECT ca.id AS account_id, u.email AS email, 1 AS is_primary
            FROM client_accounts ca
            JOIN auth."User" u ON u.id = ca.client_user_id
            WHERE ca.deleted_at IS NULL AND u.email IS NOT NULL
            ON CONFLICT ON CONSTRAINT uq_account_email DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_client_emails_account_id", table_name="client_emails")
    op.drop_table("client_emails")
    op.drop_column("client_accounts", "emergency_contact")
    op.drop_column("client_accounts", "phone")
    op.drop_column("client_accounts", "name")
