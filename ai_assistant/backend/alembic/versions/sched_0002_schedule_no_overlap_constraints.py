from alembic import op

# revision identifiers, used by Alembic.
revision = "sched_0002"
down_revision = "base_0001"
branch_labels = None
depends_on = None


def upgrade():
    # btree_gist is needed for "=" operator with GiST indexes on text/int
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    # availability_rules: use minutes-of-day range so two windows on same weekday can’t overlap
    op.execute(
        """
        ALTER TABLE public.availability_rules
        ADD CONSTRAINT availability_rules_no_overlap
        EXCLUDE USING gist (
          owner_id WITH =,
          weekday WITH =,
          int4range(
            EXTRACT(EPOCH FROM start_local)::int / 60,
            EXTRACT(EPOCH FROM end_local)::int / 60
          ) WITH &&
        )
    """
    )

    # timeoffs: no overlapping time off for same owner
    op.execute(
        """
        ALTER TABLE public.timeoffs
        ADD CONSTRAINT timeoffs_no_overlap
        EXCLUDE USING gist (
          owner_id WITH =,
          tstzrange(start_utc, end_utc) WITH &&
        )
    """
    )

    # appointments: no appointment overlaps for same owner (prevents double-book)
    op.execute(
        """
        ALTER TABLE public.appointments
        ADD CONSTRAINT appointments_no_overlap
        EXCLUDE USING gist (
          owner_id WITH =,
          tstzrange(start_utc, end_utc) WITH &&
        )
    """
    )

    # special_openings: no overlapping one-off openings per owner
    op.execute(
        """
        ALTER TABLE public.special_openings
        ADD CONSTRAINT special_openings_no_overlap
        EXCLUDE USING gist (
          owner_id WITH =,
          tstzrange(start_utc, end_utc) WITH &&
        )
    """
    )


def downgrade():
    # drop in reverse order (defensive)
    for tbl, cns in [
        ("public.special_openings", "special_openings_no_overlap"),
        ("public.appointments", "appointments_no_overlap"),
        ("public.timeoffs", "timeoffs_no_overlap"),
        ("public.availability_rules", "availability_rules_no_overlap"),
    ]:
        op.execute(
            f"""
        DO $$
        BEGIN
          IF to_regclass('{tbl}') IS NOT NULL THEN
            IF EXISTS (
              SELECT 1 FROM pg_constraint c
              JOIN pg_class t ON c.conrelid = t.oid
              WHERE t.relname = split_part('{tbl}', '.', 2)
                AND c.conname = '{cns}'
            ) THEN
              EXECUTE 'ALTER TABLE {tbl} DROP CONSTRAINT {cns}';
            END IF;
          END IF;
        END$$;
        """
        )
