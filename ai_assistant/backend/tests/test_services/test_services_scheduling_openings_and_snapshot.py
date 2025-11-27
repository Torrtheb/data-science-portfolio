from __future__ import annotations
from datetime import datetime, date, time
import uuid
import os
from zoneinfo import ZoneInfo


os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.services_scheduling as sched
from services.services_scheduling import (
    carve_opening_through_timeoff,
    merge_or_get_special_opening,
    is_owner_time_bookable,
    owner_calendar_snapshot,
)
from app.models import (
    TimeOff,
    SpecialOpening,
    AvailabilityRule,
    Appointment,
    Person,
    User,
)


def dt(y, m, d, hh=0, mm=0, tz="UTC"):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)

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
        return self._items[0] if self._items else None


class FakeResult:
    def __init__(self, data, tuple_mode=False):
        self._data = data
        self._tuple_mode = tuple_mode

    def scalars(self):
        return self

    def all(self):
        return list(self._data)


class StubSelect:
    def __init__(self, *models):
        self.models = models

    def where(self, *args, **kwargs):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self


class FakeSession:
    def __init__(self, store: dict[type, list]):
        self.store = {k: list(v) for k, v in store.items()}
        self.added = []
        self.deleted = []
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        return FakeQuery(self.store.get(model, []))

    def add(self, obj):
        self.added.append(obj)
        # Also reflect immediately into store for its class for subsequent queries
        self.store.setdefault(obj.__class__, []).append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        try:
            self.store.get(obj.__class__, []).remove(obj)
        except ValueError:
            pass

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _obj):
        pass

    def execute(self, stmt):
        # Works with our StubSelect monkeypatched into the module
        if isinstance(stmt, StubSelect):
            models = stmt.models
            if models == (AvailabilityRule,):
                return FakeResult(self.store.get(AvailabilityRule, []))
            if models == (TimeOff,):
                return FakeResult(self.store.get(TimeOff, []))
            if models == (SpecialOpening,):
                return FakeResult(self.store.get(SpecialOpening, []))
            if models == (Appointment, Person):
                appts = list(self.store.get(Appointment, []))
                # sort by start for determinism
                appts.sort(key=lambda a: a.start_utc)
                persons = {
                    getattr(p, "id", None): p for p in self.store.get(Person, [])
                }
                rows = [(a, persons.get(getattr(a, "person_id", None))) for a in appts]
                return FakeResult(rows, tuple_mode=True)
        return FakeResult([])


class Owner:
    def __init__(self, oid: str, tz: str = "UTC", edge_buf=0):
        self.id = oid
        self.timezone = tz
        self.appt_edge_buffer_min = edge_buf


# -------- Special opening helpers --------


def test_carve_opening_through_timeoff_full_delete():
    owner_id = "own1"
    off = TimeOff(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 9),
        end_utc=dt(2024, 1, 1, 12),
        note="off",
    )
    db = FakeSession({TimeOff: [off]})

    carve_opening_through_timeoff(db, owner_id, dt(2024, 1, 1, 8), dt(2024, 1, 1, 13))

    assert off in db.deleted
    assert off not in db.store.get(TimeOff, [])


def test_carve_opening_through_timeoff_head_overlap_truncates_end():
    owner_id = "own2"
    off = TimeOff(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 9, 30),
        end_utc=dt(2024, 1, 1, 10, 30),
        note="off",
    )
    db = FakeSession({TimeOff: [off]})

    carve_opening_through_timeoff(db, owner_id, dt(2024, 1, 1, 10), dt(2024, 1, 1, 11))
    assert off.end_utc == dt(2024, 1, 1, 10)
    assert "carved" in (off.note or "")


def test_carve_opening_through_timeoff_tail_overlap_truncates_start():
    owner_id = "own3"
    off = TimeOff(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 10, 30),
        end_utc=dt(2024, 1, 1, 12),
        note="off",
    )
    db = FakeSession({TimeOff: [off]})

    carve_opening_through_timeoff(db, owner_id, dt(2024, 1, 1, 10), dt(2024, 1, 1, 11))
    assert off.start_utc == dt(2024, 1, 1, 11)
    assert "carved" in (off.note or "")


def test_carve_opening_through_timeoff_split_creates_right_piece():
    owner_id = "own4"
    off = TimeOff(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 9),
        end_utc=dt(2024, 1, 1, 12),
        note="off",
    )
    db = FakeSession({TimeOff: [off]})

    carve_opening_through_timeoff(db, owner_id, dt(2024, 1, 1, 10), dt(2024, 1, 1, 11))
    assert off.end_utc == dt(2024, 1, 1, 10)
    # A new right piece was added 11-12
    right = [x for x in db.added if isinstance(x, TimeOff)]
    assert (
        right
        and right[0].start_utc == dt(2024, 1, 1, 11)
        and right[0].end_utc == dt(2024, 1, 1, 12)
    )


def test_merge_or_get_special_opening_create_new():
    owner_id = "own5"
    db = FakeSession({SpecialOpening: []})
    sp = merge_or_get_special_opening(
        db,
        owner_id,
        dt(2024, 1, 2, 9),
        dt(2024, 1, 2, 10),
        slot_minutes=30,
        buffer_minutes=5,
        note="Avail",
    )
    assert isinstance(sp, SpecialOpening)
    assert sp in db.store.get(SpecialOpening, [])
    assert sp.slot_minutes == 30 and sp.buffer_minutes == 5 and sp.note


def test_merge_or_get_special_opening_reuse_existing_and_fill_fields():
    owner_id = "own6"
    existing = SpecialOpening(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 2, 9),
        end_utc=dt(2024, 1, 2, 12),
        slot_minutes=0,
        buffer_minutes=0,
        note=None,
    )
    db = FakeSession({SpecialOpening: [existing]})
    sp = merge_or_get_special_opening(
        db,
        owner_id,
        dt(2024, 1, 2, 10),
        dt(2024, 1, 2, 11),
        slot_minutes=45,
        buffer_minutes=10,
        note="Avail",
    )
    assert sp is existing
    assert sp.slot_minutes == 45 and sp.buffer_minutes == 10 and sp.note == "Avail"


def test_merge_or_get_special_opening_merge_union_and_delete_others():
    owner_id = "own7"
    a = SpecialOpening(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 2, 9),
        end_utc=dt(2024, 1, 2, 10),
        slot_minutes=0,
        buffer_minutes=0,
    )
    b = SpecialOpening(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 2, 11),
        end_utc=dt(2024, 1, 2, 12),
        slot_minutes=0,
        buffer_minutes=0,
    )
    db = FakeSession({SpecialOpening: [a, b]})
    sp = merge_or_get_special_opening(
        db,
        owner_id,
        dt(2024, 1, 2, 9, 30),
        dt(2024, 1, 2, 11, 30),
        slot_minutes=60,
        buffer_minutes=15,
        note="Avail",
    )
    assert sp.start_utc == dt(2024, 1, 2, 9)
    assert sp.end_utc == dt(2024, 1, 2, 12)
    # One survives, one deleted
    assert len([x for x in db.store.get(SpecialOpening, []) if x is sp]) == 1
    assert any(x is a or x is b for x in db.deleted)


# -------- is_owner_time_bookable --------


def test_is_owner_time_bookable_timeoff_blocks():
    owner = Owner("own8", tz="UTC")
    offs = [
        TimeOff(
            owner_id=owner.id, start_utc=dt(2024, 1, 3, 9), end_utc=dt(2024, 1, 3, 12)
        )
    ]
    db = FakeSession({TimeOff: offs})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 3, 10), dt(2024, 1, 3, 11)
    )
    assert not ok and reason == "time off"


def test_is_owner_time_bookable_appt_blocks(monkeypatch):
    owner = Owner("own9", tz="UTC")
    # Active appointment
    appt = Appointment(
        id=uuid.uuid4(),
        owner_id=owner.id,
        start_utc=dt(2024, 1, 3, 10),
        end_utc=dt(2024, 1, 3, 11),
        status="booked",
    )
    monkeypatch.setattr(
        sched, "ACTIVE_APPT_STATUSES", ["booked"]
    )  # ensure status is considered active
    db = FakeSession({Appointment: [appt]})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 3, 10, 30), dt(2024, 1, 3, 10, 45)
    )
    assert not ok and "conflicts" in reason


def test_is_owner_time_bookable_special_opening_allows():
    owner = Owner("own10", tz="UTC")
    sp = SpecialOpening(
        owner_id=owner.id,
        start_utc=dt(2024, 1, 3, 9),
        end_utc=dt(2024, 1, 3, 12),
        slot_minutes=60,
        buffer_minutes=0,
    )
    db = FakeSession({SpecialOpening: [sp]})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 3, 10), dt(2024, 1, 3, 11)
    )
    assert ok and reason == ""


def test_is_owner_time_bookable_weekly_rule_allows():
    owner = Owner("own11", tz="UTC")
    weekday = dt(2024, 1, 1).weekday()
    r = AvailabilityRule(
        id="r1",
        owner_id=owner.id,
        weekday=weekday,
        start_local=time(9, 0),
        end_local=time(12, 0),
        slot_minutes=60,
        buffer_minutes=0,
    )
    db = FakeSession({AvailabilityRule: [r]})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 1, 10), dt(2024, 1, 1, 11)
    )
    assert ok and reason == ""


def test_is_owner_time_bookable_crosses_day_boundary():
    owner = Owner("own12", tz="UTC")
    db = FakeSession({})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 1, 23, 30), dt(2024, 1, 2, 0, 30)
    )
    assert not ok and reason == "crosses day boundary"


def test_is_owner_time_bookable_no_opening():
    owner = Owner("own13", tz="UTC")
    db = FakeSession({})
    ok, reason = is_owner_time_bookable(
        db, owner, dt(2024, 1, 1, 10), dt(2024, 1, 1, 11)
    )
    assert not ok and "no opening" in reason


# -------- owner_calendar_snapshot --------


def test_owner_calendar_snapshot_weekly_rules_and_timeoff(monkeypatch):
    # Use stub select chain compatible with our FakeSession
    monkeypatch.setattr(sched, "select", lambda *models: StubSelect(*models))
    owner_id = "own14"
    owner = User(
        id=owner_id,
        name="Owner",
        email=None,
        emailVerified=None,
        image=None,
        password=None,
        role="OWNER",
        timezone="UTC",
    )
    # One rule 9-12, one time off 10-11, one active appointment 9:30-10:00
    anchor = date(2024, 1, 1)
    r = AvailabilityRule(
        id="r2",
        owner_id=owner_id,
        weekday=anchor.weekday(),
        start_local=time(9, 0),
        end_local=time(12, 0),
        slot_minutes=60,
        buffer_minutes=0,
    )
    off = TimeOff(
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 10),
        end_utc=dt(2024, 1, 1, 11),
        note="off",
    )
    appt = Appointment(
        id=uuid.uuid4(),
        owner_id=owner_id,
        start_utc=dt(2024, 1, 1, 9, 30),
        end_utc=dt(2024, 1, 1, 10),
        status="booked",
    )

    monkeypatch.setattr(sched, "ACTIVE_APPT_STATUSES", ["booked"])  # active

    db = FakeSession(
        {
            AvailabilityRule: [r],
            TimeOff: [off],
            SpecialOpening: [],
            Appointment: [appt],
            Person: [],
            User: [owner],
        }
    )

    snap = owner_calendar_snapshot(
        db, owner_id, scope="today", anchor=anchor, tz_str="UTC"
    )

    assert snap["tz"] == "UTC"
    events = snap["events"]
    # Expect: 2 openings (9:00-9:30 and 11:00-12:00), 1 time_off, 1 appointment
    openings = [(e["start"], e["end"]) for e in events if e["type"] == "opening"]
    offs = [(e["start"], e["end"]) for e in events if e["type"] == "time_off"]
    appts = [(e["start"], e["end"]) for e in events if e["type"] == "appointment"]

    assert (dt(2024, 1, 1, 9), dt(2024, 1, 1, 9, 30)) in openings
    assert (dt(2024, 1, 1, 11), dt(2024, 1, 1, 12)) in openings
    assert offs == [(dt(2024, 1, 1, 10), dt(2024, 1, 1, 11))]
    assert appts == [(dt(2024, 1, 1, 9, 30), dt(2024, 1, 1, 10))]
