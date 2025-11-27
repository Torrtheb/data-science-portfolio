"""email

Revision ID: 52df755b0390
Revises: 3eccecbb2462
Create Date: 2025-09-26 12:03:45.878245

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "52df755b0390"
down_revision: Union[str, Sequence[str], None] = "3eccecbb2462"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Create new table for recipients
    op.create_table(
        "outbox_email_recipients",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("outbox_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["outbox_id"],
            ["outbox_emails.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_outbox_email_recipients_outbox",
        "outbox_email_recipients",
        ["outbox_id"],
    )
    op.create_index(
        "ix_outbox_email_recipients_email",
        "outbox_email_recipients",
        ["email"],
    )
    op.create_unique_constraint(
        "uq_outbox_recipient_per_email",
        "outbox_email_recipients",
        ["outbox_id", "email"],
    )

    # 2) Relax outbox_emails.to_email to be nullable
    op.alter_column(
        "outbox_emails",
        "to_email",
        existing_type=sa.String(),
        nullable=True,
    )


def downgrade() -> None:
    # reverse alter
    op.alter_column(
        "outbox_emails",
        "to_email",
        existing_type=sa.String(),
        nullable=False,
    )
    # drop recipients table
    op.drop_constraint(
        "uq_outbox_recipient_per_email", "outbox_email_recipients", type_="unique"
    )
    op.drop_index(
        "ix_outbox_email_recipients_email", table_name="outbox_email_recipients"
    )
    op.drop_index(
        "ix_outbox_email_recipients_outbox", table_name="outbox_email_recipients"
    )
    op.drop_table("outbox_email_recipients")
