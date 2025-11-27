from __future__ import annotations

import os
import re
import sys
from typing import Iterable

try:
    import psycopg  # psycopg v3
except Exception:  # pragma: no cover
    print("ERROR: psycopg package not installed in this environment.")
    print("Install with: pip install psycopg[binary]")
    raise


def parse_ids_from_dump(path: str) -> list[str]:
    want = {
        "client_accounts": ["owner_user_id", "client_user_id"],
        "prepaid_bundles": ["owner_id", "client_id"],
        "appointments": ["owner_id", "client_id"],
        "outbox_emails": ["owner_user_id", "approved_by"],
        "admin_fee_charges": ["owner_id", "client_user_id"],
        "client_notes": ["owner_id", "client_id"],
        "owner_fee_settings": ["owner_id"],
        "availability_rules": ["owner_id"],
        "special_openings": ["owner_id"],
        "timeoffs": ["owner_id"],
        "service_options": ["owner_id"],
    }
    ids: set[str] = set()

    copy_re = re.compile(r"^COPY\s+([^\s]+)\s+\(([^)]+)\)\s+FROM\s+stdin;")

    with open(path, "r", encoding="utf-8") as f:
        table: str | None = None
        idx: dict[str, int] = {}
        in_copy = False
        for raw in f:
            line = raw.rstrip("\n")
            m = copy_re.match(line)
            if m:
                raw_table = m.group(1)
                cols = [c.strip() for c in m.group(2).split(",")]
                table = raw_table.split(".")[-1]
                in_copy = table in want
                idx = {c: i for i, c in enumerate(cols)}
                continue
            if line == r"\.":
                in_copy = False
                table = None
                idx = {}
                continue
            if in_copy and table:
                fields = line.split("\t")
                for col in want[table]:
                    i = idx.get(col)
                    if i is None or i >= len(fields):
                        continue
                    v = fields[i]
                    if v and v != r"\N":
                        ids.add(v)
    return sorted(ids)


def get_dsn() -> str:
    dsn = (
        os.getenv("NEON_PSQL_DIRECT")
        or os.getenv("BACKEND_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not dsn:
        raise RuntimeError("Set NEON_PSQL_DIRECT or BACKEND_DATABASE_URL/DATABASE_URL")
    # Normalize SQLAlchemy URL (postgresql+psycopg://...) to psycopg DSN
    dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
    return dsn


def seed_users(conn: psycopg.Connection, ids: Iterable[str]) -> int:
    total = 0
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, auth")
        for uid in ids:
            cur.execute(
                (
                    'INSERT INTO auth."User"(id, role, timezone, "createdAt", "updatedAt", appt_edge_buffer_min) '
                    "VALUES (%s, 'CLIENT', 'America/Toronto', now(), now(), 5) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                (uid,),
            )
            total += 1
    conn.commit()
    return total


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/seed_auth_users.py path/to/app_data.sql")
        sys.exit(1)
    dump_path = sys.argv[1]
    ids = parse_ids_from_dump(dump_path)
    print(f"[seed] Found {len(ids)} referenced user ids")
    if not ids:
        print("[seed] No ids found; nothing to do")
        return
    dsn = get_dsn()
    with psycopg.connect(dsn) as conn:
        n = seed_users(conn, ids)
    print(f'[seed] Inserted (or ensured) {n} users in auth."User"')


if __name__ == "__main__":
    main()
