"""Drop exclusion index that blocks group lessons

Revision ID: sched_0008_drop_overlap_excl
Revises: sched_0007_group_lessons
Create Date: 2025-10-01
"""

from alembic import op


revision = "sched_0008_drop_overlap_excl"
down_revision = "sched_0007_group_lessons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The GiST exclusion index `(owner_id, tstzrange(start_utc, end_utc))` prevents
    # multiple simultaneous attendees for group lessons. We drop it to rely on
    # (owner_id, start_utc, person_id) uniqueness and app-level conflict checks.
    # Use IF EXISTS to avoid errors that would abort the transaction.
    # First drop the exclusion CONSTRAINT (canonical in Postgres)
    op.execute(
        "ALTER TABLE public.appointments DROP CONSTRAINT IF EXISTS appointments_no_overlap;"
    )
    # Then drop any leftover index with the same name (defensive cleanup)
    op.execute("DROP INDEX IF EXISTS public.appointments_no_overlap;")


def downgrade() -> None:
    # Best-effort: recreate exclusion constraint (rarely needed; kept for symmetry)
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist;")
    except Exception:
        pass
    try:
        op.execute(
            "ALTER TABLE public.appointments ADD CONSTRAINT appointments_no_overlap "
            "EXCLUDE USING gist (owner_id WITH =, tstzrange(start_utc, end_utc) WITH &&);"
        )
    except Exception:
        # If recreation fails, leave downgraded state without the constraint
        pass
