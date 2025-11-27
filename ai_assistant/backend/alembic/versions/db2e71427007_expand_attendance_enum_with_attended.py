"""expand attendance_enum with 'attended'

Revision ID: db2e71427007
Revises: 9511eb2b5896
Create Date: 2025-09-18 11:39:54.417522

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "db2e71427007"
down_revision: Union[str, Sequence[str], None] = "9511eb2b5896"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Drop default so Postgres doesn't try to cast the old default to the new type
    op.execute("ALTER TABLE appointments ALTER COLUMN attendance_status DROP DEFAULT;")

    # Rename old type and create new type with the extra label
    op.execute("ALTER TYPE attendance_enum RENAME TO attendance_enum_old;")
    op.execute(
        "CREATE TYPE attendance_enum AS ENUM ('unknown','attended','late','no_show');"
    )

    # Convert the column to the new type
    op.execute(
        """
        ALTER TABLE appointments
        ALTER COLUMN attendance_status TYPE attendance_enum
        USING attendance_status::text::attendance_enum
    """
    )

    # Restore default, then drop the old type
    op.execute(
        "ALTER TABLE appointments ALTER COLUMN attendance_status SET DEFAULT 'unknown';"
    )
    op.execute("DROP TYPE attendance_enum_old;")


def downgrade():
    # Coerce 'attended' back to 'unknown' before shrinking the enum
    op.execute(
        """
        UPDATE appointments
        SET attendance_status = 'unknown'
        WHERE attendance_status = 'attended'
    """
    )

    # Drop default, recreate the old (smaller) type, convert back, restore default, drop temp
    op.execute("ALTER TABLE appointments ALTER COLUMN attendance_status DROP DEFAULT;")
    op.execute("ALTER TYPE attendance_enum RENAME TO attendance_enum_new;")
    op.execute("CREATE TYPE attendance_enum AS ENUM ('unknown','late','no_show');")
    op.execute(
        """
        ALTER TABLE appointments
        ALTER COLUMN attendance_status TYPE attendance_enum
        USING attendance_status::text::attendance_enum
    """
    )
    op.execute(
        "ALTER TABLE appointments ALTER COLUMN attendance_status SET DEFAULT 'unknown';"
    )
    op.execute("DROP TYPE attendance_enum_new;")
