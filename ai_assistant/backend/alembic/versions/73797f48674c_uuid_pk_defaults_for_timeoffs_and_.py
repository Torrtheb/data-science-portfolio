"""uuid pk defaults for timeoffs and special_openings

Revision ID: 73797f48674c
Revises: 3e6ddb0fc6fe
Create Date: 2025-09-25 10:02:30.984094

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "73797f48674c"
down_revision: Union[str, Sequence[str], None] = "3e6ddb0fc6fe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Ensure gen_random_uuid() is available
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # TIMEOFFS: set default to gen_random_uuid() and not-null (type stays text)
    op.execute("ALTER TABLE timeoffs ALTER COLUMN id SET DEFAULT gen_random_uuid();")
    # backfill any null/empty ids just in case (shouldn't normally exist)
    op.execute(
        "UPDATE timeoffs SET id = gen_random_uuid() WHERE id IS NULL OR id = '';"
    )
    op.execute("ALTER TABLE timeoffs ALTER COLUMN id SET NOT NULL;")

    # SPECIAL_OPENINGS: same
    op.execute(
        "ALTER TABLE special_openings ALTER COLUMN id SET DEFAULT gen_random_uuid();"
    )
    op.execute(
        "UPDATE special_openings SET id = gen_random_uuid() WHERE id IS NULL OR id = '';"
    )
    op.execute("ALTER TABLE special_openings ALTER COLUMN id SET NOT NULL;")


def downgrade():
    # Remove defaults (optional)
    op.execute("ALTER TABLE timeoffs ALTER COLUMN id DROP DEFAULT;")
    op.execute("ALTER TABLE special_openings ALTER COLUMN id DROP DEFAULT;")
