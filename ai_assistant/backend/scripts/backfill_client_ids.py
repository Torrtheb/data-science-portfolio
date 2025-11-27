#!/usr/bin/env python
from __future__ import annotations

from sqlalchemy import text
from app.db import SessionLocal, init_db


def main() -> None:
    # Ensure models are registered (for migrations/projects that rely on import side-effects)
    init_db()
    sql = text(
        """
        UPDATE appointments AS a
        SET client_id = ca.client_user_id
        FROM people AS p
        JOIN client_accounts AS ca ON ca.id = p.account_id
        WHERE a.person_id = p.id
          AND a.client_id IS NULL
        """
    )
    with SessionLocal() as db:
        res = db.execute(sql)
        db.commit()
        try:
            rc = res.rowcount if res is not None else 0
        except Exception:
            rc = 0
        print(f"Backfill complete. Updated rows: {rc}")


if __name__ == "__main__":
    main()
