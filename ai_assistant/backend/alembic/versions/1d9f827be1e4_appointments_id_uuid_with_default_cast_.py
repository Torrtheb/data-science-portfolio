"""appointments.id -> UUID with default; cast prepaid_ledger.appointment_id

Revision ID: 1d9f827be1e4
Revises: 1b34cdebe3a3
Create Date: 2025-09-24 10:56:02.836374

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "1d9f827be1e4"
down_revision: Union[str, Sequence[str], None] = "1b34cdebe3a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _drop_fk_by_introspection(table: str, constrained_col: str, referred_table: str):
    """
    Find FK name that references `referred_table(id)` and constrains `constrained_col` on `table`,
    then drop it if present. Works regardless of auto-naming.
    """
    bind = op.get_bind()
    # Try the SQLAlchemy inspector first (works on many setups)
    insp = sa.inspect(bind)
    fks = insp.get_foreign_keys(table_name=table)
    for fk in fks:
        cols = fk.get("constrained_columns") or []
        ref_table = fk.get("referred_table")
        if constrained_col in cols and ref_table == referred_table:
            name = fk.get("name")
            if name:
                op.drop_constraint(name, table, type_="foreignkey")
                return

    # Fallback: query pg_constraint to find the FK name
    row = bind.execute(
        sa.text(
            """
        SELECT c.conname
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE t.relname = :tbl
          AND c.contype = 'f'
          AND (SELECT relname FROM pg_class WHERE oid = c.confrelid) = :reftbl
          AND EXISTS (
              SELECT 1
              FROM unnest(c.conkey) WITH ORDINALITY AS ck(attnum, ord)
              JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ck.attnum
              WHERE a.attname = :col
          )
        LIMIT 1
    """
        ),
        {"tbl": table, "reftbl": referred_table, "col": constrained_col},
    ).scalar()
    if row:
        op.drop_constraint(row, table, type_="foreignkey")


def upgrade():
    # 1) Ensure extension for gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # 2) Drop FK on prepaid_ledger.appointment_id (so we can alter type)
    _drop_fk_by_introspection("prepaid_ledger", "appointment_id", "appointments")

    # 3) Alter appointments.id from text/varchar -> uuid (cast existing)
    op.alter_column(
        "appointments",
        "id",
        type_=postgresql.UUID(as_uuid=True),
        postgresql_using="id::uuid",
        existing_type=sa.String(),
        existing_nullable=False,
    )
    # 3b) Set server default for new rows
    op.execute("ALTER TABLE appointments ALTER COLUMN id SET DEFAULT gen_random_uuid()")

    # 4) Alter prepaid_ledger.appointment_id to uuid (nullable ok)
    op.alter_column(
        "prepaid_ledger",
        "appointment_id",
        type_=postgresql.UUID(as_uuid=True),
        postgresql_using="appointment_id::uuid",
        existing_type=sa.String(),
        existing_nullable=True,
    )

    # 5) Recreate FK with a stable name
    op.create_foreign_key(
        "prepaid_ledger_appointment_id_fkey",
        "prepaid_ledger",
        "appointments",
        ["appointment_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    # Drop FK (whatever its name is right now)
    _drop_fk_by_introspection("prepaid_ledger", "appointment_id", "appointments")

    # Remove default on appointments.id
    op.execute("ALTER TABLE appointments ALTER COLUMN id DROP DEFAULT")

    # Cast columns back to text
    op.alter_column(
        "prepaid_ledger",
        "appointment_id",
        type_=sa.String(),
        postgresql_using="appointment_id::text",
        existing_type=postgresql.UUID(as_uuid=True),
        existing_nullable=True,
    )
    op.alter_column(
        "appointments",
        "id",
        type_=sa.String(),
        postgresql_using="id::text",
        existing_type=postgresql.UUID(as_uuid=True),
        existing_nullable=False,
    )
