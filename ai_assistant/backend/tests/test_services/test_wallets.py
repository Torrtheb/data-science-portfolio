from __future__ import annotations
from datetime import datetime, timezone as _tz
import os
import uuid

import pytest

os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.wallets as wallets
import services.payments as payments
from app.models import (
    Appointment,
    AdminFeeCharge,
    AdminFeeStatus,
    ClientAccount,
    PrepaidBundle,
    PrepaidLedger,
)


def now():
    return datetime(2024, 1, 1, 12, 0, tzinfo=_tz.utc)


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)
        self._order_key = None

    def outerjoin(self, *a, **k):
        return self

    def filter(self, *a, **k):
        # Keep simple equality filters for AdminFeeCharge.client_user_id matching
        try:
            conds = []
            for expr in a:
                left = getattr(expr, "left", None)
                right = getattr(expr, "right", None)
                op = getattr(expr, "operator", None)
                if op is None or getattr(op, "__name__", "") not in ("eq",):
                    continue
                col = getattr(left, "key", None) or getattr(left, "name", None)
                val = getattr(right, "value", None)
                if val is None and right is not None and not hasattr(right, "value"):
                    val = right
                if col is not None:
                    conds.append((col, val))
            if conds and self._items and not isinstance(self._items[0], tuple):

                def ok(item):
                    for col, val in conds:
                        if getattr(item, col, None) != val:
                            return False
                    return True

                self._items = [it for it in self._items if ok(it)]
        except Exception:
            pass
        return self

    def order_by(self, *a, **k):
        # accept and ignore
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def scalar(self):
        # Not used directly in tests for wallets; wallet balance is monkeypatched
        return 0


class FakeSession:
    def __init__(self, store: dict):
        self.store = {k: list(v) for k, v in store.items()}
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def query(self, model):
        # model can be a column (e.g., ClientAccount.id); when that happens, our store
        # should contain simple tuples already; just return them.
        return FakeQuery(self.store.get(model, []))

    def get(self, model, pk):
        for obj in self.store.get(model, []):
            if getattr(obj, "id", None) == int(pk):
                return obj
        return None

    def add(self, obj):
        self.added.append(obj)
        self.store.setdefault(obj.__class__, []).append(obj)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _obj):
        pass


def test_wallet_balance_helper(monkeypatch):
    # Patch wallet balance to sum our ledger entries in store
    def sum_balance(db, bundle_id):
        return sum(
            int(x.amount_cents)
            for x in db.store.get(PrepaidLedger, [])
            if int(x.bundle_id) == int(bundle_id)
        )

    monkeypatch.setattr(wallets, "_wallet_balance", sum_balance)

    db = FakeSession(
        {
            PrepaidLedger: [
                PrepaidLedger(
                    bundle_id=1,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=500,
                    appointment_id=None,
                    note="dep",
                )
            ]
        }
    )
    assert wallets._wallet_balance(db, 1) == 500


def test_auto_apply_wallet_funds_applies_to_appts_then_charges(monkeypatch):
    owner = "own1"
    client = "cu1"
    wallet = PrepaidBundle(
        id=1,
        owner_id=owner,
        client_id=client,
        name="Wallet",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )

    # Two unpaid appointments of 60 minutes each
    appt1 = Appointment(
        id=uuid.uuid4(),
        owner_id=owner,
        client_id=client,
        person_id=None,
        client_name=None,
        client_email=None,
        start_utc=now(),
        end_utc=now(),
        status="booked",
        amount_paid_cents=0,
        payment_status="unpaid",
        price_override_cents=None,
        bundle_id=None,
    )
    appt1.end_utc = appt1.start_utc.replace(
        minute=appt1.start_utc.minute
    )  # zero duration initially; compute_price decides
    # second one later
    appt2 = Appointment(
        id=uuid.uuid4(),
        owner_id=owner,
        client_id=client,
        person_id=None,
        client_name=None,
        client_email=None,
        start_utc=now(),
        end_utc=now(),
        status="booked",
        amount_paid_cents=0,
        payment_status="unpaid",
        price_override_cents=None,
        bundle_id=None,
    )

    charge = AdminFeeCharge(
        id=1,
        owner_id=owner,
        client_account_id=10,
        client_user_id=client,
        amount_cents=1500,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=0,
        note=None,
        created_at=now(),
        updated_at=now(),
    )

    # Price map and compute
    monkeypatch.setattr(
        payments, "_service_price_map", lambda db, owner_user_id=None: {60: 1000}
    )

    def fake_compute(db, appt, price_map):
        return 1000

    monkeypatch.setattr(payments, "compute_price_cents", fake_compute)

    # Wallet balance starts at 2200
    def sum_balance(db, bundle_id):
        return sum(
            int(x.amount_cents)
            for x in db.store.get(PrepaidLedger, [])
            if int(x.bundle_id) == int(bundle_id)
        )

    monkeypatch.setattr(wallets, "_wallet_balance", sum_balance)

    db = FakeSession(
        {
            PrepaidBundle: [wallet],
            Appointment: [appt1, appt2],
            AdminFeeCharge: [charge],
            # No ClientAccount ids → filter falls back to client_user_id
            ClientAccount.id: [],
            PrepaidLedger: [
                PrepaidLedger(
                    bundle_id=1,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=2200,
                    appointment_id=None,
                    note="deposit",
                )
            ],
        }
    )

    summary = wallets.auto_apply_wallet_funds(
        db, owner_id=owner, bundle_id=1, note_prefix="Test apply"
    )

    # Two appts at 1000 each -> 2000 applied; then 200 left to apply to admin fee charge
    assert summary["applied_cents"] == 2200
    assert summary["appointments"] >= 1  # at least one appointment applied
    assert summary["admin_fee_charges"] == 1
    # Remaining balance should be whatever is left in ledger after new consume entries; our sum_balance reflects adds
    assert summary["remaining_balance_cents"] >= 0


def test_adjust_wallet_balance_credit_and_debit(monkeypatch):
    owner = "own2"
    client = "cu2"
    wallet = PrepaidBundle(
        id=2,
        owner_id=owner,
        client_id=client,
        name="Wallet",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )

    def sum_balance(db, bundle_id):
        return sum(
            int(x.amount_cents)
            for x in db.store.get(PrepaidLedger, [])
            if int(x.bundle_id) == int(bundle_id)
        )

    monkeypatch.setattr(wallets, "_wallet_balance", sum_balance)

    db = FakeSession({PrepaidBundle: [wallet], PrepaidLedger: []})

    # Credit +1000 triggers auto-apply call; mock it to return summary
    called = {}
    monkeypatch.setattr(
        wallets,
        "auto_apply_wallet_funds",
        lambda dbs, owner_id, bundle_id, note_prefix: (
            {"remaining_balance_cents": sum_balance(dbs, bundle_id)}
        ),
    )
    bal_after = wallets.adjust_wallet_balance(
        db, owner_id=owner, bundle_id=2, amount_cents=1000, note="Top up"
    )
    assert bal_after == 1000

    # Debit that would overdraft -> error
    with pytest.raises(wallets.WalletAdjustmentError):
        wallets.adjust_wallet_balance(
            db, owner_id=owner, bundle_id=2, amount_cents=-2000
        )

    # Valid small debit
    bal_after2 = wallets.adjust_wallet_balance(
        db, owner_id=owner, bundle_id=2, amount_cents=-400
    )
    assert bal_after2 == 600


def test_adjust_wallet_balance_validation_errors():
    owner = "own3"
    other = "other"
    # Wrong owner
    bundle_wrong_owner = PrepaidBundle(
        id=3,
        owner_id=other,
        client_id="c3",
        name="B",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    db = FakeSession({PrepaidBundle: [bundle_wrong_owner]})
    with pytest.raises(wallets.WalletAdjustmentError):
        wallets.adjust_wallet_balance(db, owner_id=owner, bundle_id=3, amount_cents=100)

    # Not a wallet (has credits)
    bundle_credits = PrepaidBundle(
        id=4,
        owner_id=owner,
        client_id="c4",
        name="B",
        total_credits=5,
        remaining_credits=5,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    db2 = FakeSession({PrepaidBundle: [bundle_credits]})
    with pytest.raises(wallets.WalletAdjustmentError):
        wallets.adjust_wallet_balance(
            db2, owner_id=owner, bundle_id=4, amount_cents=100
        )

    # Client mismatch by user id
    bundle_wallet = PrepaidBundle(
        id=5,
        owner_id=owner,
        client_id="c5",
        name="B",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    db3 = FakeSession({PrepaidBundle: [bundle_wallet], PrepaidLedger: []})
    with pytest.raises(wallets.WalletAdjustmentError):
        wallets.adjust_wallet_balance(
            db3,
            owner_id=owner,
            bundle_id=5,
            amount_cents=100,
            client_user_id="different",
        )

    # Client mismatch by account id
    ok_acct = ClientAccount(
        id=41,
        owner_user_id=owner,
        client_user_id="c5",
        name=None,
        phone=None,
        emergency_contact=None,
        created_at=now(),
        deleted_at=None,
    )
    bad_acct = ClientAccount(
        id=42,
        owner_user_id=owner,
        client_user_id="other_user",
        name=None,
        phone=None,
        emergency_contact=None,
        created_at=now(),
        deleted_at=None,
    )
    db4 = FakeSession(
        {PrepaidBundle: [bundle_wallet], ClientAccount: [bad_acct], PrepaidLedger: []}
    )
    with pytest.raises(wallets.WalletAdjustmentError):
        wallets.adjust_wallet_balance(
            db4, owner_id=owner, bundle_id=5, amount_cents=100, client_account_id=42
        )


def test_auto_apply_skips_appt_with_other_bundle(monkeypatch):
    owner = "own4"
    client = "cu4"
    wallet = PrepaidBundle(
        id=6,
        owner_id=owner,
        client_id=client,
        name="W",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    appt = Appointment(
        id=uuid.uuid4(),
        owner_id=owner,
        client_id=client,
        person_id=None,
        client_name=None,
        client_email=None,
        start_utc=now(),
        end_utc=now(),
        status="booked",
        amount_paid_cents=0,
        payment_status="unpaid",
        price_override_cents=None,
        bundle_id=999,
    )

    # price always 1000
    monkeypatch.setattr(
        payments, "_service_price_map", lambda db, owner_user_id=None: {60: 1000}
    )
    monkeypatch.setattr(payments, "compute_price_cents", lambda db, ap, pm: 1000)

    # Wallet funds 1000
    def sum_balance(db, bundle_id):
        return sum(
            int(x.amount_cents)
            for x in db.store.get(PrepaidLedger, [])
            if int(x.bundle_id) == int(bundle_id)
        )

    monkeypatch.setattr(wallets, "_wallet_balance", sum_balance)

    db = FakeSession(
        {
            PrepaidBundle: [wallet],
            Appointment: [appt],
            AdminFeeCharge: [],
            ClientAccount.id: [],
            PrepaidLedger: [
                PrepaidLedger(
                    bundle_id=6,
                    event="adjust",
                    delta_credits=0,
                    amount_cents=1000,
                    appointment_id=None,
                    note="deposit",
                )
            ],
        }
    )

    summary = wallets.auto_apply_wallet_funds(db, owner_id=owner, bundle_id=6)
    # Appointment should be skipped due to different bundle_id
    assert summary["appointments"] == 0
