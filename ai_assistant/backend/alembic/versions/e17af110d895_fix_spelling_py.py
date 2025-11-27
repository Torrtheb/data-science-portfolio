"""fix_spelling.py

Revision ID: e17af110d895
Revises: 2919a558d246
Create Date: 2025-09-26 08:43:25.257558

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "e17af110d895"
down_revision: Union[str, Sequence[str], None] = "2919a558d246"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # --- 1) Drop existing checks if present
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS appointments_status_check"
    )
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS ck_appointments_identity"
    )

    # --- 2) TEMP identity: allow missing identity IF status is canceled/cancelled (so we can update)
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK (
            person_id IS NOT NULL
            OR client_id IS NOT NULL
            OR client_email IS NOT NULL
            OR client_name IS NOT NULL
            OR status IN ('canceled','cancelled')
        )
    """
    )

    # --- 3) TEMP status check: allow both spellings so UPDATE won't fail
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT appointments_status_check
        CHECK (status IN ('booked','completed','canceled','cancelled'))
    """
    )

    # --- 4) Normalize data
    op.execute("UPDATE appointments SET status='canceled' WHERE status='cancelled'")

    # --- 5) Finalize status check to the canonical set
    op.execute("ALTER TABLE appointments DROP CONSTRAINT appointments_status_check")
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT appointments_status_check
        CHECK (status IN ('booked','completed','canceled'))
    """
    )

    # --- 6) Finalize identity check: allow missing identity only when canceled
    op.execute("ALTER TABLE appointments DROP CONSTRAINT ck_appointments_identity")
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK (
            person_id IS NOT NULL
            OR client_id IS NOT NULL
            OR client_email IS NOT NULL
            OR client_name IS NOT NULL
            OR status = 'canceled'
        )
    """
    )


def downgrade():
    # Revert to two-L spelling and (stricter) identity rule
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS appointments_status_check"
    )
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS ck_appointments_identity"
    )

    # Status back to 3 states using 'cancelled'
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT appointments_status_check
        CHECK (status IN ('booked','completed','cancelled'))
    """
    )

    # Identity rule: require some identity for all non-cancelled statuses
    # (and allow missing identity only when status='cancelled' to mirror upgrade symmetry)
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK (
            person_id IS NOT NULL
            OR client_id IS NOT NULL
            OR client_email IS NOT NULL
            OR client_name IS NOT NULL
            OR status = 'cancelled'
        )
    """
    )
