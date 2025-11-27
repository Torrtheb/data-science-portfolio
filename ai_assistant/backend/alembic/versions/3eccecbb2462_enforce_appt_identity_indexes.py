"""enforce appt identity + indexes

Revision ID: 3eccecbb2462
Revises: e17af110d895
Create Date: 2025-09-26 11:40:28.266895

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3eccecbb2462"
down_revision: Union[str, Sequence[str], None] = "e17af110d895"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # UNIQUE (owner_id, start_utc)
    op.execute(
        """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE c.conname = 'uq_owner_start' AND t.relname = 'appointments'
      ) THEN
        ALTER TABLE appointments
        ADD CONSTRAINT uq_owner_start UNIQUE (owner_id, start_utc);
      END IF;
    END$$;
    """
    )

    # INDEX (owner_id, start_utc, end_utc)
    op.execute(
        """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = 'ix_appt_owner_start_end'
      ) THEN
        CREATE INDEX ix_appt_owner_start_end
          ON appointments (owner_id, start_utc, end_utc);
      END IF;
    END$$;
    """
    )

    # CHECK: person_id OR (client_name AND client_email)
    op.execute(
        """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE c.conname = 'ck_appointments_identity' AND t.relname = 'appointments'
      ) THEN
        ALTER TABLE appointments
        ADD CONSTRAINT ck_appointments_identity
        CHECK ((person_id IS NOT NULL) OR (client_name IS NOT NULL AND client_email IS NOT NULL));
      END IF;
    END$$;
    """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_appt_owner_start_end;")
    op.execute("ALTER TABLE appointments DROP CONSTRAINT IF EXISTS uq_owner_start;")
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT IF EXISTS ck_appointments_identity;"
    )
