"""Add unsubscribed flag to client_emails

Revision ID: sched_0011_client_email_unsubscribed
Revises: sched_0010_merge_all_heads
Create Date: 2025-02-15
"""

from alembic import op
import sqlalchemy as sa


revision = "sched_0011_client_email_unsubscribed"
down_revision = "sched_0010_merge_all_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "client_emails",
        sa.Column(
            "unsubscribed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("client_emails", "unsubscribed")
