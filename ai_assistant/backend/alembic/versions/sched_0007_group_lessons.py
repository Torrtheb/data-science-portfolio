"""Group lessons: add group_id and relax unique to (owner,start,person)

Revision ID: sched_0007_group_lessons
Revises: bd3f7bde6d9c
Create Date: 2025-01-20
"""

from alembic import op
import sqlalchemy as sa


revision = "sched_0007_group_lessons"
down_revision = "bd3f7bde6d9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add group_id (nullable)
    try:
        op.add_column(
            "appointments",
            sa.Column(
                "group_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True
            ),
        )
    except Exception:
        # Fallback for SQLite or generic drivers
        op.add_column(
            "appointments", sa.Column("group_id", sa.String(length=36), nullable=True)
        )

    # Drop prior unique index if present, then create the new scoped unique
    try:
        op.drop_index("ix_owner_start_active_unique", table_name="appointments")
    except Exception:
        pass
    op.create_index(
        "ix_owner_start_person_active_unique",
        "appointments",
        ["owner_id", "start_utc", "person_id"],
        unique=True,
        postgresql_where=sa.text("status <> 'canceled'"),
    )


def downgrade() -> None:
    try:
        op.drop_index("ix_owner_start_person_active_unique", table_name="appointments")
    except Exception:
        pass
    # Recreate previous (stricter) unique index
    op.create_index(
        "ix_owner_start_active_unique",
        "appointments",
        ["owner_id", "start_utc"],
        unique=True,
        postgresql_where=sa.text("status <> 'canceled'"),
    )
    try:
        op.drop_column("appointments", "group_id")
    except Exception:
        pass
