from __future__ import annotations
from datetime import datetime, timezone as _tz
import os

import pytest

os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.admin_fee as admin_fee
from app.models import (
    AdminFeeCharge,
    AdminFeeStatus,
    ClientAccount,
    OwnerFeeSetting,
    PrepaidBundle,
    PrepaidLedger,
    User,
)


def now():
    return datetime(2024, 1, 1, 12, 0, tzinfo=_tz.utc)


class FakeQuery:
    def __init__(self, items):
        self._items = list(items)
        self._limit = None

    def filter(self, *args, **kwargs):
        # Support simple equality comparisons for single-model lists
        if not self._items or isinstance(self._items[0], tuple):
            return self
        try:
            conds = []
            for expr in args:
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
            if conds:

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
        return self

    def limit(self, n: int):
        self._limit = int(n)
        return self

    def all(self):
        items = self._items
        if self._limit is not None:
            items = items[: self._limit]
        return list(items)

    def first(self):
        return self._items[0] if self._items else None

    def scalar(self):
        # If numbers, sum them
        if self._items and isinstance(self._items[0], (int, float)):
            return sum(self._items)
        return self.first()


class FakeSession:
    def __init__(self, store: dict):
        self.store = {k: list(v) for k, v in store.items()}
        self.added = []
        self.deleted = []

    def query(self, model):
        return FakeQuery(self.store.get(model, []))

    def add(self, obj):
        # Provide default ids and timestamps for new rows
        if getattr(obj, "id", None) in (None, 0):
            # Allocate incremental id within this model
            cur = [getattr(x, "id", 0) or 0 for x in self.store.get(obj.__class__, [])]
            next_id = (max(cur) + 1) if cur else 1
            try:
                obj.id = next_id
            except Exception:
                pass
        for fld in ("created_at", "updated_at"):
            if not hasattr(obj, fld) or getattr(obj, fld) is None:
                try:
                    setattr(obj, fld, now())
                except Exception:
                    pass
        self.added.append(obj)
        self.store.setdefault(obj.__class__, []).append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        try:
            self.store.get(obj.__class__, []).remove(obj)
        except ValueError:
            pass

    def commit(self):
        pass

    def refresh(self, _obj):
        pass


def test_get_and_set_admin_fee_setting():
    db = FakeSession({OwnerFeeSetting: []})
    # default
    s = admin_fee.get_admin_fee_setting(db, "o1")
    assert s.admin_fee_cents == admin_fee.DEFAULT_ADMIN_FEE_CENTS
    # set
    out = admin_fee.set_admin_fee_setting(db, "o1", 2500)
    assert out.admin_fee_cents == 2500
    # update
    out2 = admin_fee.set_admin_fee_setting(db, "o1", -10)
    assert out2.admin_fee_cents == 0


def test_create_admin_fee_charge_with_wallet_and_email(monkeypatch):
    owner = "own1"
    acct = ClientAccount(
        id=10,
        owner_user_id=owner,
        client_user_id="c1",
        name="Alice",
        phone=None,
        emergency_contact=None,
        created_at=now(),
        deleted_at=None,
    )
    # Wallet bundle present for client
    wallet = PrepaidBundle(
        id=99,
        owner_id=owner,
        client_id="c1",
        name="Wallet",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    db = FakeSession(
        {
            ClientAccount: [acct],
            PrepaidBundle: [wallet],
            User: [
                User(
                    id=owner,
                    name="Owner Name",
                    email=None,
                    emailVerified=None,
                    image=None,
                    password=None,
                    role="OWNER",
                    timezone="UTC",
                    createdAt=now(),
                    updatedAt=now(),
                    appt_edge_buffer_min=5,
                    group_price_60_cents=None,
                )
            ],
        }
    )

    # Patch auto_apply to simulate applying 500 cents to the just-created charge
    def fake_auto_apply(db_sess, owner_id: str, bundle_id: int, note_prefix: str):
        for c in db_sess.store.get(AdminFeeCharge, []):
            if c.owner_id == owner and c.client_account_id == acct.id:
                c.bundle_applied_cents = 500

    monkeypatch.setattr(admin_fee, "auto_apply_wallet_funds", fake_auto_apply)

    sent = {}
    monkeypatch.setattr(
        admin_fee, "_account_primary_email", lambda dbs, account_id: "acct@example.com"
    )
    monkeypatch.setattr(admin_fee, "_send_email", lambda **k: sent.update(k))

    out = admin_fee.create_admin_fee_charge(
        db, owner, client_account_id=acct.id, amount_cents=1000, note="Late cancel"
    )
    assert out.amount_cents == 1000
    assert out.bundle_applied_cents == 500
    assert out.client_label in {"Alice", None}
    # Email was sent to account email with subject containing amount
    assert sent.get("to") == "acct@example.com"
    assert "$10.00" in sent.get("subject", "")


def test_update_admin_fee_charge_apply_and_status_transitions(monkeypatch):
    owner = "own2"
    acct = ClientAccount(
        id=11,
        owner_user_id=owner,
        client_user_id="c2",
        name=None,
        phone=None,
        emergency_contact=None,
        created_at=now(),
        deleted_at=None,
    )
    charge = AdminFeeCharge(
        id=1,
        owner_id=owner,
        client_account_id=acct.id,
        client_user_id="c2",
        amount_cents=1500,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=0,
        note=None,
        created_at=now(),
        updated_at=now(),
    )
    db = FakeSession({ClientAccount: [acct], AdminFeeCharge: [charge]})

    # apply_wallet should apply 1500 and mark bundle paid
    monkeypatch.setattr(
        admin_fee,
        "_apply_wallet_to_charge",
        lambda dbs, ch: (setattr(ch, "bundle_applied_cents", 1500) or 1500),
    )

    out = admin_fee.update_admin_fee_charge(db, owner, charge_id=1, apply_wallet=True)
    assert out.bundle_applied_cents == 1500
    assert out.status in ("bundle", "paid")

    # Now refund -> zero cash and bundle, status refunded
    charge.bundle_applied_cents = 500
    monkeypatch.setattr(
        admin_fee,
        "_refund_wallet_for_charge",
        lambda dbs, ch: (setattr(ch, "bundle_applied_cents", 0) or 500),
    )
    out2 = admin_fee.update_admin_fee_charge(db, owner, charge_id=1, status="refunded")
    assert out2.status == "refunded"
    assert out2.bundle_applied_cents == 0 and out2.paid_cash_cents == 0


def test_delete_admin_fee_charge_rules():
    owner = "own3"
    # Deletable
    c1 = AdminFeeCharge(
        id=1,
        owner_id=owner,
        client_account_id=1,
        client_user_id=None,
        amount_cents=1000,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=0,
        note=None,
        created_at=now(),
        updated_at=now(),
    )
    # Has payments -> not deletable
    c2 = AdminFeeCharge(
        id=2,
        owner_id=owner,
        client_account_id=1,
        client_user_id=None,
        amount_cents=1000,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=100,
        note=None,
        created_at=now(),
        updated_at=now(),
    )
    db = FakeSession({AdminFeeCharge: [c1, c2]})

    admin_fee.delete_admin_fee_charge(db, owner, charge_id=1)
    assert c1 in db.deleted

    with pytest.raises(ValueError):
        admin_fee.delete_admin_fee_charge(db, owner, charge_id=2)


def test_apply_and_refund_wallet_helpers(monkeypatch):
    owner = "own4"
    charge = AdminFeeCharge(
        id=1,
        owner_id=owner,
        client_account_id=1,
        client_user_id="cu4",
        amount_cents=1000,
        status=AdminFeeStatus.UNPAID,
        paid_cash_cents=0,
        bundle_applied_cents=0,
        note=None,
        created_at=now(),
        updated_at=now(),
    )
    wallet = PrepaidBundle(
        id=77,
        owner_id=owner,
        client_id="cu4",
        name="Wallet",
        total_credits=0,
        remaining_credits=0,
        price_cents=0,
        currency="USD",
        status="active",
        expires_at=None,
        created_at=now(),
    )
    db = FakeSession(
        {PrepaidBundle: [wallet], AdminFeeCharge: [charge], PrepaidLedger: []}
    )

    # Simplify balance and lookup
    monkeypatch.setattr(admin_fee, "_wallet_for_charge", lambda dbs, ch: wallet)
    monkeypatch.setattr(admin_fee, "_wallet_balance", lambda dbs, bid: 600)

    applied = admin_fee._apply_wallet_to_charge(db, charge)
    assert applied == 600
    assert charge.bundle_applied_cents == 600
    # refund the wallet entries
    refunded = admin_fee._refund_wallet_for_charge(db, charge)
    assert refunded == 600
    assert charge.bundle_applied_cents == 0


def test_list_admin_fee_charges_limit_and_owner_filter():
    owner = "own5"
    others = "other"
    rows = [
        AdminFeeCharge(
            id=i,
            owner_id=owner if i <= 3 else others,
            client_account_id=1,
            client_user_id=None,
            amount_cents=1000,
            status=AdminFeeStatus.UNPAID,
            paid_cash_cents=0,
            bundle_applied_cents=0,
            note=None,
            created_at=now(),
            updated_at=now(),
        )
        for i in range(1, 6)
    ]
    db = FakeSession({AdminFeeCharge: rows})
    out = admin_fee.list_admin_fee_charges(db, owner, limit=2)
    assert len(out) == 2
    # Ensure all returned belong to owner
    assert all(r.owner_id == owner for r in out)
