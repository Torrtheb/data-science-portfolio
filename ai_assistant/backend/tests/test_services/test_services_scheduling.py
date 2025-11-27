from datetime import datetime, timedelta, date
import os
from zoneinfo import ZoneInfo

import pytest

# Ensure a lightweight driver is used during import
os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

from services.services_scheduling import (  # noqa: E402
    _as_aware_utc,  # noqa: E402
    _to_utc,  # noqa: E402
    _overlaps,  # noqa: E402
    _ranges_overlap,  # noqa: E402
    _normalize_window,  # noqa: E402
    _segmentize,  # noqa: E402
    _collect_boundaries,  # noqa: E402
    _apply_precedence,  # noqa: E402
    _start_end_from_local,  # noqa: E402
)


def dt(y, m, d, hh=0, mm=0, tz="UTC"):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


def test_as_aware_utc_naive_attaches_utc():
    naive = datetime(2024, 1, 1, 12, 0)
    out = _as_aware_utc(naive)
    assert out.tzinfo is not None
    assert out.tzinfo.key == "UTC"


def test_as_aware_utc_converts_to_utc():
    est = dt(2024, 1, 1, 10, 0, tz="America/Toronto")  # UTC-5
    out = _as_aware_utc(est)
    assert out.tzinfo.key == "UTC"
    # 10:00 ET -> 15:00 UTC in Jan
    assert out.hour == 15 and out.minute == 0


def test_to_utc_idempotent_for_utc():
    x = dt(2024, 6, 1, 12, 0, tz="UTC")
    out = _to_utc(x)
    assert out == x


def test_overlaps_half_open_logic():
    a0, a1 = dt(2024, 1, 1, 10), dt(2024, 1, 1, 11)
    b0, b1 = dt(2024, 1, 1, 10, 30), dt(2024, 1, 1, 11, 30)
    c0, c1 = dt(2024, 1, 1, 11), dt(2024, 1, 1, 12)
    # overlap
    assert _overlaps(a0, a1, b0, b1) is True
    # just touching end/start should NOT overlap (half-open)
    assert _overlaps(a0, a1, c0, c1) is False
    # identical spans overlap
    assert _overlaps(a0, a1, a0, a1) is True


def test_ranges_overlap():
    a0, a1 = dt(2024, 1, 1, 9), dt(2024, 1, 1, 10)
    b0, b1 = dt(2024, 1, 1, 9, 30), dt(2024, 1, 1, 9, 45)
    assert _ranges_overlap(a0, a1, b0, b1) is True
    # no overlap
    c0, c1 = dt(2024, 1, 1, 10), dt(2024, 1, 1, 11)
    assert _ranges_overlap(a0, a1, c0, c1) is False


def test_normalize_window_today_week_month():
    tz = "America/Toronto"
    anchor = date(2024, 1, 17)  # Wednesday
    s, e, tzinfo = _normalize_window(anchor, "today", tz)
    assert tzinfo.key == tz
    assert s.hour == 0 and s.minute == 0 and (e - s) == timedelta(days=1)

    s, e, _ = _normalize_window(anchor, "week", tz)
    # Monday start
    assert s.weekday() == 0 and (e - s) == timedelta(days=7)

    s, e, _ = _normalize_window(anchor, "month", tz)
    assert s.day == 1
    # January -> February boundary
    assert e.month == 2 and e.day == 1


def test_segmentize_and_collect_boundaries():
    spans1 = [(dt(2024, 1, 1, 9), dt(2024, 1, 1, 12))]
    spans2 = [(dt(2024, 1, 1, 10), dt(2024, 1, 1, 11))]
    boundaries = _collect_boundaries(spans1, spans2)
    segments = _segmentize(boundaries)
    # Expected cut points: 9,10,11,12
    assert segments == [
        (dt(2024, 1, 1, 9), dt(2024, 1, 1, 10)),
        (dt(2024, 1, 1, 10), dt(2024, 1, 1, 11)),
        (dt(2024, 1, 1, 11), dt(2024, 1, 1, 12)),
    ]


def test_apply_precedence_time_off_beats_and_appt_subtracts():
    # Base openings 09:00-17:00 UTC
    w_open = [(dt(2024, 1, 2, 9), dt(2024, 1, 2, 17), object())]
    s_open = []
    # Time off 12:00-13:30
    offs = [(dt(2024, 1, 2, 12), dt(2024, 1, 2, 13, 30), object())]
    # Appointment 10:00-10:30 (no edge pad)
    appts = [(dt(2024, 1, 2, 10), dt(2024, 1, 2, 10, 30), object())]

    openings, time_off, appt_spans = _apply_precedence(
        w_open, s_open, offs, appts, appt_edge_buffer_min=0
    )

    # Time off preserved as-is
    assert time_off == [(dt(2024, 1, 2, 12), dt(2024, 1, 2, 13, 30))]
    assert appt_spans == [(dt(2024, 1, 2, 10), dt(2024, 1, 2, 10, 30))]

    # Openings should be: 09:00-10:00, 10:30-12:00, 13:30-17:00
    assert openings == [
        (dt(2024, 1, 2, 9), dt(2024, 1, 2, 10)),
        (dt(2024, 1, 2, 10, 30), dt(2024, 1, 2, 12)),
        (dt(2024, 1, 2, 13, 30), dt(2024, 1, 2, 17)),
    ]


def test_apply_precedence_with_edge_buffer():
    w_open = [(dt(2024, 1, 3, 9), dt(2024, 1, 3, 10), object())]
    offs = []
    appts = [(dt(2024, 1, 3, 9, 30), dt(2024, 1, 3, 9, 45), object())]
    # 5-minute pad around appt removes 9:25-9:50 from openings
    openings, _, _ = _apply_precedence(w_open, [], offs, appts, appt_edge_buffer_min=5)
    # Remaining fragments should be 09:00-09:25 and 09:50-10:00
    assert openings == [
        (dt(2024, 1, 3, 9, 0), dt(2024, 1, 3, 9, 25)),
        (dt(2024, 1, 3, 9, 50), dt(2024, 1, 3, 10, 0)),
    ]


def test_ensure_local_and_start_end_from_local():
    class Owner:
        def __init__(self, tz):
            self.timezone = tz

    owner = Owner("America/Toronto")

    # Naive local gets tz attached
    naive_local = datetime(2024, 1, 1, 10, 0)
    s_utc, e_utc = _start_end_from_local(owner, naive_local, 45)
    assert s_utc.tzinfo.key == "UTC" and e_utc.tzinfo.key == "UTC"
    # 10:00 ET -> 15:00 UTC in Jan
    assert s_utc.hour == 15 and e_utc.hour == 15 and e_utc.minute == 45

    # Aware in another TZ should convert to owner's local first
    aware_pacific = dt(2024, 7, 1, 10, 0, tz="America/Los_Angeles")  # PDT
    s_utc, e_utc = _start_end_from_local(owner, aware_pacific, 60)
    # 10:00 PT == 13:00 ET (DST), which is 17:00 UTC
    assert s_utc.hour == 17 and e_utc.hour == 18


def test_start_end_from_local_rejects_nonpositive_duration():
    class Owner:
        def __init__(self, tz):
            self.timezone = tz

    owner = Owner("America/Toronto")
    with pytest.raises(Exception):
        _start_end_from_local(owner, datetime(2024, 1, 1, 10), 0)
