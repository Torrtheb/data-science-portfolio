"""outbox emails

Revision ID: 3e6ddb0fc6fe
Revises: 1d9f827be1e4
Create Date: 2025-09-24 18:57:28.443831

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "3e6ddb0fc6fe"
down_revision: Union[str, Sequence[str], None] = "1d9f827be1e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1) Create enum once
    outbox_enum = postgresql.ENUM(
        "pending", "approved", "rejected", "sent", name="outboxemailstatus"
    )
    outbox_enum.create(op.get_bind(), checkfirst=True)

    # 2) Create table WITHOUT inline FKs
    op.create_table(
        "outbox_emails",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_user_id", sa.String(), nullable=False),
        sa.Column("to_email", sa.String(), nullable=False),
        sa.Column("to_name", sa.String(), nullable=True),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("text_body", sa.Text(), nullable=False),
        sa.Column("preview_html", sa.Text(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(name="outboxemailstatus", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by", sa.String(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_outbox_owner_status", "outbox_emails", ["owner_user_id", "status"]
    )

    # 3) Add FKs after the fact (no quotes, just names; set schema explicitly)
    op.create_foreign_key(
        "fk_outbox_owner_user",
        source_table="outbox_emails",
        referent_table="User",
        local_cols=["owner_user_id"],
        remote_cols=["id"],
        referent_schema="auth",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_outbox_approved_by",
        source_table="outbox_emails",
        referent_table="User",
        local_cols=["approved_by"],
        remote_cols=["id"],
        referent_schema="auth",
        ondelete="SET NULL",
    )


def downgrade():
    # Drop FKs (they’ll be auto-dropped with table, but explicit is fine too)
    try:
        op.drop_constraint("fk_outbox_owner_user", "outbox_emails", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_outbox_approved_by", "outbox_emails", type_="foreignkey")
    except Exception:
        pass

    op.drop_index("idx_outbox_owner_status", table_name="outbox_emails")
    op.drop_table("outbox_emails")
    postgresql.ENUM(name="outboxemailstatus").drop(op.get_bind(), checkfirst=True)
