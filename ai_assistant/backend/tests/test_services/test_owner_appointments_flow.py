from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import os
import pytest

# satisfy app.db import-time DSN checks
os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

from fastapi import BackgroundTasks
from fastapi import HTTPException

from app.models import Appointment, User, TimeOff, PrepaidBundle, PrepaidLedger
from routers.owner_appointments import (
    update_appointment,
    UpdateAppointmentPayload,
    cancel_appointment,
)
from app.core.auth import TokenUser


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)

    def filter(self, *_, **__):
        return self

    def filter_by(self, **__):
        return self

    def order_by(self, *_, **__):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def scalar(self):
        # For sum/count scalar paths, we return the first primitive value if present
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
        pass

    def get(self, model, pk):
        rows = self.store.get(model, [])
        for r in rows:
            if getattr(r, "id", None) == pk:
                return r
        return None


def dt(y, m, d, hh=0, mm=0, tz="America/Toronto"):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


def test_reschedule_conflict_yields_confirm_required():
    tz = "America/Toronto"
    owner = User(
        id="owner-1", email="o@example.com", name="Owner", role="OWNER", timezone=tz
    )
    # existing appointment that will conflict
    a_existing = Appointment(
        id="a2",
        owner_id=owner.id,
        client_id=None,
        start_utc=dt(2025, 1, 5, 14, 0, tz).astimezone(ZoneInfo("UTC")),
        end_utc=dt(2025, 1, 5, 15, 0, tz).astimezone(ZoneInfo("UTC")),
        status="booked",
    )
    # target appointment we try to move into conflict window
    a_target = Appointment(
        id="a1",
        owner_id=owner.id,
        client_id=None,
        start_utc=dt(2025, 1, 5, 10, 0, tz).astimezone(ZoneInfo("UTC")),
        end_utc=dt(2025, 1, 5, 11, 0, tz).astimezone(ZoneInfo("UTC")),
        status="booked",
    )

    store = {
        User: [owner],
        Appointment: [a_target, a_existing],
        TimeOff: [],
    }
    db = FakeSession(store)

    payload = UpdateAppointmentPayload(
        start_local=dt(2025, 1, 5, 14, 0, tz),
        duration_minutes=60,
        allow_override=False,
    )
    user = TokenUser(sub=owner.id, email=owner.email, role="OWNER", timezone=tz)
    bt = BackgroundTasks()

    with pytest.raises(HTTPException) as ei:
        update_appointment("a1", payload, bt, db=db, user=user)
    assert ei.value.status_code == 409
    assert isinstance(ei.value.detail, str) and ei.value.detail.startswith(
        "CONFIRM_REQUIRED:"
    )


def test_cancel_appointment_full_refund_to_wallet():
    tz = "America/Toronto"
    owner = User(
        id="owner-1", email="o@example.com", name="Owner", role="OWNER", timezone=tz
    )
    client = User(
        id="client-1", email="c@example.com", name="Client", role="CLIENT", timezone=tz
    )
    start = datetime.now(ZoneInfo(tz)) + timedelta(days=2)
    end = start + timedelta(minutes=60)
    appt = Appointment(
        id="appt-1",
        owner_id=owner.id,
        client_id=client.id,
        start_utc=start.astimezone(ZoneInfo("UTC")),
        end_utc=end.astimezone(ZoneInfo("UTC")),
        status="booked",
    )
    setattr(appt, "amount_paid_cents", 5000)

    db = FakeSession(
        {
            User: [owner, client],
            Appointment: [appt],
            PrepaidBundle: [],
            PrepaidLedger: [],
        }
    )
    user = TokenUser(sub=owner.id, email=owner.email, role="OWNER", timezone=tz)
    bt = BackgroundTasks()

    out = cancel_appointment(
        appt_id="appt-1", background_tasks=bt, message=None, db=db, user=user
    )
    assert out.get("ok") is True
    # appointment was marked canceled and refunded
    assert appt.status == "canceled"
    assert getattr(appt, "payment_status", None) == "refunded"
    # cash amount cleared to 0
    assert getattr(appt, "amount_paid_cents", 0) == 0
    # refund ledger was created
    refund_rows = [
        x
        for x in db.added
        if isinstance(x, PrepaidLedger)
        and x.event == "refund"
        and int(x.amount_cents) == 5000
    ]
    assert len(refund_rows) == 1


def test_cancel_appointment_under_24h_no_financial_change():
    tz = "America/Toronto"
    owner = User(
        id="owner-1", email="o@example.com", name="Owner", role="OWNER", timezone=tz
    )
    client = User(
        id="client-1", email="c@example.com", name="Client", role="CLIENT", timezone=tz
    )
    start = datetime.now(ZoneInfo(tz)) + timedelta(hours=12)
    end = start + timedelta(minutes=60)
    appt = Appointment(
        id="appt-2",
        owner_id=owner.id,
        client_id=client.id,
        start_utc=start.astimezone(ZoneInfo("UTC")),
        end_utc=end.astimezone(ZoneInfo("UTC")),
        status="booked",
    )
    setattr(appt, "amount_paid_cents", 7000)
    setattr(appt, "payment_status", "paid")

    db = FakeSession(
        {
            User: [owner, client],
            Appointment: [appt],
            PrepaidBundle: [],
            PrepaidLedger: [],
        }
    )
    user = TokenUser(sub=owner.id, email=owner.email, role="OWNER", timezone=tz)
    bt = BackgroundTasks()

    out = cancel_appointment(
        appt_id="appt-2", background_tasks=bt, message=None, db=db, user=user
    )
    assert out.get("ok") is True
    assert appt.status == "canceled"
    # No change to payment_status or amounts
    assert getattr(appt, "payment_status", None) == "paid"
    assert getattr(appt, "amount_paid_cents", 0) == 7000
    # No refund ledger created
    refund_rows = [
        x for x in db.added if isinstance(x, PrepaidLedger) and x.event == "refund"
    ]
    assert len(refund_rows) == 0
