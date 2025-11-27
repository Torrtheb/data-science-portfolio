"""normalize_appt.py

Revision ID: 2919a558d246
Revises: d91cd5f51d2e
Create Date: 2025-09-26 00:36:10.114532

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2919a558d246"
down_revision: Union[str, Sequence[str], None] = "d91cd5f51d2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ACTIVE = ("booked", "confirmed", "pending")


def upgrade():
    # 0) Be resilient to prior runs / different names
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'ck_appointments_identity'
              AND conrelid = 'appointments'::regclass
          ) THEN
            ALTER TABLE appointments DROP CONSTRAINT ck_appointments_identity;
          END IF;
        END$$;
    """
    )

    # 1) Normalize status spellings BEFORE adding any new checks/defaults
    op.execute(
        "UPDATE appointments SET status = 'cancelled' WHERE status = 'canceled';"
    )
    op.execute("UPDATE appointments SET status = 'booked'    WHERE status IS NULL;")

    # 2) Backfill identity ONLY for active rows missing identity
    #    (cancelled rows may remain without identity)
    op.execute(
        f"""
        UPDATE appointments a
        SET client_name  = COALESCE(client_name, 'Unknown'),
            client_email = COALESCE(client_email, 'unknown-'||a.id::text||'@invalid.local')
        WHERE a.status IN {ACTIVE}
          AND a.person_id IS NULL
          AND (a.client_name IS NULL OR a.client_email IS NULL);
    """
    )

    # 3) Add finite status check + NOT NULL + default
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT appointments_status_check
        CHECK (status IN ('booked','confirmed','pending','cancelled'));
    """
    )
    op.alter_column(
        "appointments",
        "status",
        existing_type=sa.String(),
        server_default="booked",
        nullable=False,
    )

    # 4) Re-add identity check, but ONLY for active statuses
    op.execute(
        """
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK (
          status = 'cancelled'
          OR person_id IS NOT NULL
          OR (client_name IS NOT NULL AND client_email IS NOT NULL)
        );
    """
    )


def downgrade():
    # Drop new checks/defaults
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS ck_appointments_identity;"
    )
    op.alter_column("appointments", "status", server_default=None, nullable=True)
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS appointments_status_check;"
    )
