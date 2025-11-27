"""create agent_messages table

Revision ID: 7c21d2d3a4b1
Revises: 6b1f_add_15min
Create Date: 2025-09-30 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7c21d2d3a4b1"
down_revision: Union[str, Sequence[str], None] = "6b1f_add_15min"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Ensure pgcrypto for gen_random_uuid() is present in this DB (already used elsewhere)
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    bind = op.get_bind()
    insp = sa.inspect(bind)
    if not insp.has_table("agent_messages"):
        op.create_table(
            "agent_messages",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("thread_id", sa.UUID(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["thread_id"],
                ["agent_threads.id"],
                name="fk_agent_messages_thread",
                ondelete="CASCADE",
            ),
        )
        op.create_check_constraint(
            "ck_agent_messages_role",
            "agent_messages",
            "role in ('user','ai','tool')",
        )
        op.create_index("ix_agent_messages_thread", "agent_messages", ["thread_id"])
        op.create_index("ix_agent_messages_created", "agent_messages", ["created_at"])
    else:
        # If table exists (e.g., created outside Alembic), ensure indexes exist.
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_messages_thread ON agent_messages(thread_id)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_agent_messages_created ON agent_messages(created_at)"
        )
        # Conditionally add the check constraint if missing.
        op.execute(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'ck_agent_messages_role'
              ) THEN
                ALTER TABLE agent_messages
                ADD CONSTRAINT ck_agent_messages_role CHECK (role IN ('user','ai','tool'));
              END IF;
            END$$;
            """
        )


def downgrade():
    op.drop_index("ix_agent_messages_created", table_name="agent_messages")
    op.drop_index("ix_agent_messages_thread", table_name="agent_messages")
    op.drop_table("agent_messages")
