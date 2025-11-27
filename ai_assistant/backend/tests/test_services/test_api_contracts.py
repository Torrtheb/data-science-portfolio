from __future__ import annotations

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks

from app.models import Appointment, User
from routers.owner_appointments import (
    list_appointments,
    UpdateAppointmentPayload,
    admin_create_appointment,
    AdminCreateAppt,
    admin_create_recurring_appointments,
    AdminCreateRecurringAppts,
)


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)
        self._entity = None

    def filter(self, *_, **__):
        return self

    def filter_by(self, **__):
        return self

    def outerjoin(self, *_, **__):
        return self

    def order_by(self, *_, **__):
        return self

    def with_entities(self, *args, **kwargs):
        return self

    def scalar(self):
        # Used for count distinct
        return len(self._items)

    def limit(self, *_):
        return self

    def offset(self, *_):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class FakeSession:
    def __init__(self, store: dict[type, list]):
        self.store = {k: list(v) for k, v in store.items()}
        self.added = []

    def query(self, model):
        return FakeQuery(self.store.get(model, []))

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def refresh(self, _obj):
        pass

    def get(self, model, pk):
        rows = self.store.get(model, [])
        for r in rows:
            if getattr(r, "id", None) == pk:
                return r
        return None


def dt(y, m, d, hh=0, mm=0, tz="America/Toronto"):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


def test_list_appointments_sets_pagination_headers():
    owner = User(
        id="owner-1", email="o@example.com", name="Owner", role="OWNER", timezone="UTC"
    )
    a = Appointment(
        id="a1",
        owner_id=owner.id,
        client_id=None,
        start_utc=dt(2025, 1, 1, 9, 0, "UTC"),
        end_utc=dt(2025, 1, 1, 10, 0, "UTC"),
        status="booked",
    )
    b = Appointment(
        id="a2",
        owner_id=owner.id,
        client_id=None,
        start_utc=dt(2025, 1, 1, 11, 0, "UTC"),
        end_utc=dt(2025, 1, 1, 12, 0, "UTC"),
        status="booked",
    )

    db = FakeSession({User: [owner], Appointment: [a, b]})

    class DummyResp:
        def __init__(self):
            self.headers = {}

    resp = DummyResp()
    out = list_appointments(
        response=resp, db=db, user=type("U", (), {"sub": owner.id})(), limit=1, offset=0
    )
    assert isinstance(out, list)
    assert resp.headers.get("X-Total-Count") == "2"
    assert resp.headers.get("X-Next-Offset") == "1"


def test_update_payload_extra_field_422():
    with pytest.raises(Exception):
        UpdateAppointmentPayload(
            start_local=dt(2025, 1, 1, 9, 0), duration_minutes=60, not_a_field=True
        )  # type: ignore


def test_owner_settings_in_extra_field_forbidden():
    # Import here to avoid circulars
    from routers.owner_appointments import OwnerSettingsIn

    with pytest.raises(Exception):
        OwnerSettingsIn(
            appt_edge_buffer_min=5,
            auto_apply_wallet_on_book=True,
            wallet_deposits_as_paid=False,
            group_price_60_cents=None,
            extra_field=True,
        )  # type: ignore


def test_booking_conflict_payload_single_and_recurring():
    tz = "America/Toronto"
    owner = User(
        id="owner-1", email="o@example.com", name="Owner", role="OWNER", timezone=tz
    )
    # existing appt at 2-3pm
    existing = Appointment(
        id="a2",
        owner_id=owner.id,
        client_id=None,
        start_utc=dt(2025, 1, 5, 14, 0, tz).astimezone(ZoneInfo("UTC")),
        end_utc=dt(2025, 1, 5, 15, 0, tz).astimezone(ZoneInfo("UTC")),
        status="booked",
    )
    db = FakeSession({User: [owner], Appointment: [existing]})

    payload = AdminCreateAppt(
        client_name="Test",
        client_email="test@example.com",
        start_local=dt(2025, 1, 5, 14, 0, tz),
        duration_minutes=60,
        status="booked",
        allow_override=False,
        confirm_if_conflicts=False,
    )
    bt = BackgroundTasks()
    with pytest.raises(Exception) as ei:
        admin_create_appointment(
            payload=payload,
            background_tasks=bt,
            request=None,
            db=db,
            user=type("U", (), {"sub": owner.id})(),
        )
    assert "CONFIRM_REQUIRED:" in str(ei.value)

    rec = AdminCreateRecurringAppts(
        client_name="Test",
        client_email="test@example.com",
        start_local=dt(2025, 1, 5, 14, 0, tz),
        duration_minutes=60,
        status="booked",
        repeat_every_weeks=1,
        occurrences=2,
        allow_override=False,
        confirm_if_conflicts=False,
    )
    with pytest.raises(Exception) as ei2:
        admin_create_recurring_appointments(
            payload=rec,
            background_tasks=bt,
            request=None,
            db=db,
            user=type("U", (), {"sub": owner.id})(),
        )
    assert "CONFIRM_REQUIRED:" in str(ei2.value)
