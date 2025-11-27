"""add appt_edge_buffer_min to auth.User

Revision ID: 0ac637417f9a
Revises: 73797f48674c
Create Date: 2025-09-25 12:34:43.270850

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0ac637417f9a"
down_revision: Union[str, Sequence[str], None] = "73797f48674c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Make idempotent to avoid failing if Prisma or a prior run already added the column
    op.execute(
        """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'auth'
              AND table_name = 'User'
              AND column_name = 'appt_edge_buffer_min'
          ) THEN
            ALTER TABLE auth."User" ADD COLUMN appt_edge_buffer_min SMALLINT DEFAULT '5' NOT NULL;
          END IF;
        END$$;
        """
    )


def downgrade():
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'auth'
              AND table_name = 'User'
              AND column_name = 'appt_edge_buffer_min'
          ) THEN
            ALTER TABLE auth."User" DROP COLUMN appt_edge_buffer_min;
          END IF;
        END$$;
        """
    )
