"""
Seed a minimal demo dataset so the app isn't empty after deploy.

Creates:
- Owner user (auth."User") with id 'owner-demo-1'
- One client user 'client-demo-1' and matching ClientAccount/Person/ClientEmail
- A special opening on the next Friday 09:00–12:00 owner-local
- A time off block on the next Wednesday 13:00–15:00 owner-local

Safe to run multiple times; it will skip rows that already exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.db import SessionLocal
from app.models import (
    User,
    RoleEnum,
    ClientAccount,
    ClientEmail,
    Person,
    SpecialOpening,
    TimeOff,
)

OWNER_ID = (
    os.getenv("SEED_OWNER_ID")
    or os.getenv("DEV_FAKE_OWNER_ID")
    or os.getenv("DEV_OWNER_ID")
    or "owner-demo-1"
)
CLIENT_ID = os.getenv("SEED_CLIENT_ID") or "client-demo-1"
OWNER_EMAIL = (
    os.getenv("SEED_OWNER_EMAIL") or os.getenv("OWNER_EMAIL") or "owner@example.com"
)
CLIENT_EMAIL = os.getenv("SEED_CLIENT_EMAIL") or "dev1@example.com"
TZ = os.getenv("SEED_TIMEZONE") or "America/New_York"


def _next_weekday(start: datetime, target_wd: int) -> datetime:
    delta = (target_wd - start.weekday()) % 7
    delta = delta or 7
    return start + timedelta(days=delta)


def seed():
    owner_id = OWNER_ID
    client_id = CLIENT_ID

    with SessionLocal() as db:
        # Owner
        owner = (
            db.query(User)
            .filter((User.id == OWNER_ID) | (User.email == OWNER_EMAIL))
            .first()
        )
        if not owner:
            owner = User(
                id=OWNER_ID,
                name="Demo Owner",
                email=OWNER_EMAIL,
                role=RoleEnum.OWNER,
                timezone=TZ,
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc),
            )
            db.add(owner)
        else:
            owner_id = owner.id

        # Client user + account
        client_user = (
            db.query(User)
            .filter((User.id == CLIENT_ID) | (User.email == CLIENT_EMAIL))
            .first()
        )
        if not client_user:
            client_user = User(
                id=CLIENT_ID,
                name="Dev One",
                email=CLIENT_EMAIL,
                role=RoleEnum.CLIENT,
                timezone=TZ,
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc),
            )
            db.add(client_user)
        else:
            client_id = client_user.id
        db.flush()

        acct = (
            db.query(ClientAccount)
            .filter(
                ClientAccount.owner_user_id == owner_id,
                ClientAccount.client_user_id == client_id,
            )
            .first()
        )
        if not acct:
            acct = ClientAccount(
                owner_user_id=owner_id,
                client_user_id=client_id,
                name="Dev1 Household",
                phone="555-123-4567",
            )
            db.add(acct)
            db.flush()

        if not db.query(ClientEmail).filter(ClientEmail.account_id == acct.id).first():
            db.add(
                ClientEmail(
                    account_id=acct.id,
                    email=CLIENT_EMAIL,
                    is_primary=1,
                    unsubscribed=0,
                )
            )
        if not db.query(Person).filter(Person.account_id == acct.id).first():
            db.add(Person(account_id=acct.id, full_name="Dev One", email=CLIENT_EMAIL))

        # Opening: next Friday 09:00–12:00 local
        now = datetime.now(ZoneInfo(TZ))
        next_fri = _next_weekday(now, 4)
        start_local = next_fri.replace(hour=9, minute=0, second=0, microsecond=0)
        end_local = next_fri.replace(hour=12, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)
        existing_open = (
            db.query(SpecialOpening)
            .filter(
                SpecialOpening.owner_id == owner_id,
                SpecialOpening.start_utc == start_utc,
                SpecialOpening.end_utc == end_utc,
            )
            .first()
        )
        if not existing_open:
            db.add(
                SpecialOpening(
                    owner_id=owner_id,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    slot_minutes=60,
                    buffer_minutes=0,
                    note="Demo opening",
                )
            )

        # Time off: next Wednesday 13:00–15:00 local
        next_wed = _next_weekday(now, 2)
        toff_start = next_wed.replace(hour=13, minute=0, second=0, microsecond=0)
        toff_end = next_wed.replace(hour=15, minute=0, second=0, microsecond=0)
        toff_start_utc = toff_start.astimezone(timezone.utc)
        toff_end_utc = toff_end.astimezone(timezone.utc)
        existing_to = (
            db.query(TimeOff)
            .filter(
                TimeOff.owner_id == owner_id,
                TimeOff.start_utc == toff_start_utc,
                TimeOff.end_utc == toff_end_utc,
            )
            .first()
        )
        if not existing_to:
            db.add(
                TimeOff(
                    owner_id=owner_id,
                    start_utc=toff_start_utc,
                    end_utc=toff_end_utc,
                    note="Demo time off",
                )
            )

        db.commit()
        print("Demo seed complete.")
        print(f"Owner login: {owner.email} (id={owner.id}) timezone={owner.timezone}")
        print(f"Client login: {client_user.email} (id={client_user.id})")
        print("Password: demo-pass-123")


if __name__ == "__main__":
    seed()
