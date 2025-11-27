"""enable pgvector

Revision ID: 72b520b2e226
Revises: 9490121c387f
Create Date: 2025-09-22 11:15:00.747953

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "72b520b2e226"
down_revision: Union[str, Sequence[str], None] = "9490121c387f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')


def downgrade():
    op.execute('DROP EXTENSION IF EXISTS "vector";')
