"""agent_threads table

Revision ID: 1b34cdebe3a3
Revises: 72b520b2e226
Create Date: 2025-09-23 10:44:58.649186

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1b34cdebe3a3"
down_revision: Union[str, Sequence[str], None] = "72b520b2e226"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    """Create agent_threads in a safe, idempotent way.

    If the table or index already exists (e.g., created manually earlier),
    skip creation to avoid aborting the migration chain.
    """
    # Ensure gen_random_uuid() is available
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    bind = op.get_bind()
    insp = sa.inspect(bind)

    # Create table only if missing
    if not insp.has_table("agent_threads", schema="public"):
        op.create_table(
            "agent_threads",
            sa.Column(
                "id",
                sa.UUID(),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("title", sa.String(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            schema="public",
        )

    # Ensure index exists
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_threads_user ON public.agent_threads(user_id)"
    )


def downgrade():
    op.drop_index("ix_agent_threads_user", table_name="agent_threads")
    op.drop_table("agent_threads")
