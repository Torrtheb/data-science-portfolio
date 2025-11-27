from __future__ import annotations
from datetime import datetime, timedelta, time
import os
from zoneinfo import ZoneInfo

import pytest

# Use lightweight in-memory DB URL to satisfy app.db import-time check.
os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.services_scheduling as sched  # module for monkeypatching
from services.services_scheduling import (
    generate_daily_slots,
    book_appointment,
    ServiceBookingError,
)
from app.models import Appointment, AvailabilityRule


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)

    # Accept and ignore common query chaining; return self for fluency
    def filter(self, *args, **kwargs):
        return self

    def filter_by(self, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def with_for_update(self):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def limit(self, *_):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def scalar(self):
        # For code paths that call .scalar() expecting a simple value
        return self._items[0] if self._items else None


class FakeSession:
    def __init__(self, store: dict[type, list]):
        self.store = {k: list(v) for k, v in store.items()}
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        return FakeQuery(self.store.get(model, []))

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _obj):
        # No-op for detached instances
        pass


class Owner:
    def __init__(self, oid: str, tz: str = "America/Toronto"):
        self.id = oid
        self.timezone = tz
        self.appt_edge_buffer_min = 0


def dt(y, m, d, hh=0, mm=0, tz="America/Toronto"):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


def make_rule(owner_id: str, weekday: int, sh: int, eh: int, slot=60, buf=0):
    r = AvailabilityRule(
        id=f"r-{owner_id}-{weekday}",
        owner_id=str(owner_id),
        weekday=int(weekday),
        start_local=time(sh, 0),
        end_local=time(eh, 0),
        slot_minutes=int(slot),
        buffer_minutes=int(buf),
    )
    # Ensure "active" attribute if code reads it
    setattr(r, "active", True)
    return r


def test_generate_daily_slots_simple_future_day(monkeypatch):
    owner = Owner("owner-1", tz="America/Toronto")
    future_day = (datetime.now(ZoneInfo(owner.timezone)) + timedelta(days=30)).date()

    # Provide a single 9-12 window with 60-min slots, no specials, no appts, no time off
    rule = make_rule(owner.id, future_day.weekday(), 9, 12, slot=60, buf=0)
    db = FakeSession({AvailabilityRule: [rule]})

    slots = generate_daily_slots(db, owner, future_day)

    # Expect three 1-hour slots: 09-10, 10-11, 11-12 in owner's local TZ
    assert len(slots) == 3
    assert [(s.hour, e.hour) for s, e in slots] == [(9, 10), (10, 11), (11, 12)]
    # All slots should be timezone-aware and in the owner's TZ
    assert all(
        s.tzinfo is not None and s.tzinfo.key == owner.timezone for s, _ in slots
    )


def test_book_appointment_happy_path_with_denorm_identity(monkeypatch):
    owner_id = "owner-2"
    owner = Owner(owner_id, tz="America/Toronto")

    # Fake session returns the owner for User query; no other data paths are used.
    from app.models import User

    db = FakeSession({User: [owner]})

    # Make duration validation and availability checks pass without DB
    monkeypatch.setattr(
        sched, "_validate_duration_against_service_options", lambda db, oid, dur: None
    )
    monkeypatch.setattr(sched, "is_owner_time_bookable", lambda db, o, s, e: (True, ""))
    # Resolve person as denormalized identity only
    monkeypatch.setattr(
        sched,
        "_resolve_person_for_owner",
        lambda db, oid, client_email, client_name, client_query: (
            None,
            client_name,
            client_email,
        ),
    )

    start_local = dt(2024, 12, 1, 10, 0, tz=owner.timezone)
    appt = book_appointment(
        db,
        owner_id,
        start_local=start_local,
        duration_min=60,
        client_name="Alice",
        client_email="alice@example.com",
        client_query=None,
        price_cents=None,
        private_note=None,
        create_person_if_missing=False,
    )

    assert isinstance(appt, Appointment)
    assert appt.owner_id == owner_id
    assert appt.client_name == "Alice"
    assert appt.client_email == "alice@example.com"
    assert appt.status == "booked"
    # Starts at 15:00 UTC in Dec (ET standard time)
    assert appt.start_utc.tzinfo is not None and appt.start_utc.hour in (15, 14, 16)
    assert (appt.end_utc - appt.start_utc) == timedelta(minutes=60)


def test_book_appointment_missing_identity(monkeypatch):
    owner_id = "owner-3"
    from app.models import User

    owner = Owner(owner_id)
    db = FakeSession({User: [owner]})

    monkeypatch.setattr(
        sched, "_validate_duration_against_service_options", lambda db, oid, dur: None
    )
    monkeypatch.setattr(sched, "is_owner_time_bookable", lambda db, o, s, e: (True, ""))
    # No person and no denorm identity
    monkeypatch.setattr(
        sched,
        "_resolve_person_for_owner",
        lambda db, oid, client_email, client_name, client_query: (None, None, None),
    )

    with pytest.raises(ServiceBookingError) as ei:
        book_appointment(
            db,
            owner_id,
            start_local=dt(2024, 12, 1, 10),
            duration_min=30,
            client_name=None,
            client_email=None,
        )
    assert ei.value.code == "MISSING_IDENTITY"


def test_book_appointment_no_availability(monkeypatch):
    owner_id = "owner-4"
    from app.models import User

    owner = Owner(owner_id)
    db = FakeSession({User: [owner]})

    monkeypatch.setattr(
        sched, "_validate_duration_against_service_options", lambda db, oid, dur: None
    )
    monkeypatch.setattr(
        sched, "is_owner_time_bookable", lambda db, o, s, e: (False, "time off")
    )
    monkeypatch.setattr(
        sched,
        "_resolve_person_for_owner",
        lambda db, oid, client_email, client_name, client_query: (
            None,
            "Bob",
            "b@example.com",
        ),
    )

    with pytest.raises(ServiceBookingError) as ei:
        book_appointment(
            db,
            owner_id,
            start_local=dt(2024, 12, 1, 10),
            duration_min=30,
            client_name="Bob",
            client_email="b@example.com",
        )
    assert ei.value.code in ("NO_AVAILABILITY", "OVERLAP")
