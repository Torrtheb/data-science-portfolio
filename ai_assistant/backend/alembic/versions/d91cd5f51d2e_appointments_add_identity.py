"""appointments: add identity

Revision ID: d91cd5f51d2e
Revises: a173bf1c80cd
Create Date: 2025-09-25 23:54:11.693067

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d91cd5f51d2e"
down_revision: Union[str, Sequence[str], None] = "a173bf1c80cd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add as NOT VALID so existing bad rows don't block deploy; we’ll fix then validate.
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK (person_id IS NOT NULL OR client_email IS NOT NULL) NOT VALID;
    """
    )


def downgrade():
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS ck_appointments_identity;"
    )
