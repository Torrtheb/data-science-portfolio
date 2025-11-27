"""appt notes + attendance/payment + bundles

Revision ID: 08e6328f2c09
Revises: 691c1131a629
Create Date: 2025-09-17 21:31:00.291845
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "08e6328f2c09"
down_revision: Union[str, Sequence[str], None] = "691c1131a629"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Create ENUM types up front (required for ALTER TABLE on existing table) ---
    attendance_enum = postgresql.ENUM(
        "unknown",
        "on_time",
        "late",
        "no_show",
        name="attendance_enum",
        create_type=True,
    )
    payment_enum = postgresql.ENUM(
        "unpaid", "paid", "refunded", "waived", name="payment_enum", create_type=True
    )
    bind = op.get_bind()
    attendance_enum.create(bind=bind, checkfirst=True)
    payment_enum.create(bind=bind, checkfirst=True)

    # --- appointments columns ---
    op.add_column(
        "appointments",
        sa.Column("client_previsit_note", sa.Text(), nullable=True),
    )
    op.add_column(
        "appointments",
        sa.Column("owner_private_note", sa.Text(), nullable=True),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "attendance_status",
            attendance_enum,
            nullable=False,
            server_default=sa.text("'unknown'::attendance_enum"),
        ),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "late_minutes",
            sa.SmallInteger(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "payment_status",
            payment_enum,
            nullable=False,
            server_default=sa.text("'unpaid'::payment_enum"),
        ),
    )
    op.add_column(
        "appointments",
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "appointments",
        sa.Column("bundle_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_appointments_bundle_id", "appointments", ["bundle_id"])

    # --- prepaid_bundles ---
    op.create_table(
        "prepaid_bundles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.String(), nullable=False, index=True),
        sa.Column("client_id", sa.String(), nullable=False, index=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("total_credits", sa.SmallInteger(), nullable=False),
        sa.Column("remaining_credits", sa.SmallInteger(), nullable=False),
        sa.Column("price_cents", sa.Integer(), nullable=False),
        sa.Column(
            "currency", sa.String(length=10), nullable=False, server_default="USD"
        ),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default="active"
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_check_constraint(
        "ck_bundle_total_nonneg", "prepaid_bundles", "total_credits >= 0"
    )
    op.create_check_constraint(
        "ck_bundle_remaining_nonneg", "prepaid_bundles", "remaining_credits >= 0"
    )

    # --- prepaid_ledger ---
    op.create_table(
        "prepaid_ledger",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("bundle_id", sa.Integer(), nullable=False, index=True),
        sa.Column("event", sa.String(length=20), nullable=False),
        sa.Column("delta_credits", sa.SmallInteger(), nullable=False),
        sa.Column("appointment_id", sa.String(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # --- client_notes ---
    op.create_table(
        "client_notes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.String(), nullable=False, index=True),
        sa.Column("client_id", sa.String(), nullable=False, index=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("pinned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # fk appointments.bundle_id -> prepaid_bundles.id
    op.create_foreign_key(
        None,
        "appointments",
        "prepaid_bundles",
        ["bundle_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # drop FK + columns/indexes first
    op.drop_constraint(None, "appointments", type_="foreignkey")
    op.drop_index("ix_appointments_bundle_id", table_name="appointments")
    op.drop_column("appointments", "bundle_id")
    op.drop_column("appointments", "paid_at")
    op.drop_column("appointments", "payment_status")
    op.drop_column("appointments", "late_minutes")
    op.drop_column("appointments", "attendance_status")
    op.drop_column("appointments", "owner_private_note")
    op.drop_column("appointments", "client_previsit_note")

    op.drop_table("client_notes")
    op.drop_table("prepaid_ledger")
    op.drop_table("prepaid_bundles")

    # finally drop ENUM types (only after no columns reference them)
    bind = op.get_bind()
    payment_enum = postgresql.ENUM(
        "unpaid", "paid", "refunded", "waived", name="payment_enum"
    )
    attendance_enum = postgresql.ENUM(
        "unknown", "on_time", "late", "no_show", name="attendance_enum"
    )
    payment_enum.drop(bind=bind, checkfirst=True)
    attendance_enum.drop(bind=bind, checkfirst=True)
