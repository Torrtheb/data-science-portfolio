"""add admin fee tables

Revision ID: 2de73ae08490
Revises: 0e2b_store_credit_on_ledger
Create Date: 2025-09-28 23:26:20.292052

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "2de73ae08490"
down_revision: Union[str, Sequence[str], None] = "0e2b_store_credit_on_ledger"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enum for status (if you want database-level validation)
    admin_fee_status = postgresql.ENUM(
        "unpaid",
        "bundle",
        "refunded",
        "waived",
        "paid",
        name="admin_fee_status",
        create_type=False,  # change to True if the enum doesn't exist yet
    )

    op.create_table(
        "owner_fee_settings",
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column(
            "admin_fee_cents", sa.Integer(), nullable=False, server_default="1500"
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("owner_id"),
        sa.ForeignKeyConstraint(["owner_id"], ["auth.User.id"], ondelete="CASCADE"),
    )

    admin_fee_status.create(op.get_bind())

    op.create_table(
        "admin_fee_charges",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column("client_account_id", sa.Integer(), nullable=False),
        sa.Column("client_user_id", sa.String(), nullable=True),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("status", admin_fee_status, nullable=False, server_default="unpaid"),
        sa.Column("paid_cash_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "bundle_applied_cents", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["auth.User.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["client_account_id"], ["client_accounts.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["client_user_id"], ["auth.User.id"], ondelete="SET NULL"
        ),
        sa.CheckConstraint("amount_cents >= 0", name="ck_admin_fee_amount_nonneg"),
        sa.CheckConstraint("paid_cash_cents >= 0", name="ck_admin_fee_cash_nonneg"),
        sa.CheckConstraint(
            "bundle_applied_cents >= 0", name="ck_admin_fee_bundle_nonneg"
        ),
    )
    op.create_index("ix_admin_fee_charges_owner_id", "admin_fee_charges", ["owner_id"])
    op.create_index(
        "ix_admin_fee_charges_client_account_id",
        "admin_fee_charges",
        ["client_account_id"],
    )
    op.create_index(
        "ix_admin_fee_charges_client_user_id", "admin_fee_charges", ["client_user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_admin_fee_charges_client_user_id", table_name="admin_fee_charges")
    op.drop_index(
        "ix_admin_fee_charges_client_account_id", table_name="admin_fee_charges"
    )
    op.drop_index("ix_admin_fee_charges_owner_id", table_name="admin_fee_charges")
    op.drop_table("admin_fee_charges")

    op.drop_table("owner_fee_settings")

    postgresql.ENUM(
        "unpaid", "bundle", "refunded", "waived", "paid", name="admin_fee_status"
    ).drop(op.get_bind(), checkfirst=True)
