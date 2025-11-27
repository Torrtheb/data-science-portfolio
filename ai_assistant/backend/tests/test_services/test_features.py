from __future__ import annotations
import json
import os
from pathlib import Path

import pytest

# Satisfy backend import-time DB check globally
os.environ.setdefault("BACKEND_DATABASE_URL", "sqlite:///:memory:")

import services.features as features


def setup_tmp_settings(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    settings_file = data_dir / "owner_settings.json"
    monkeypatch.setattr(features, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(features, "SETTINGS_FILE", settings_file, raising=False)
    return data_dir, settings_file


def test_get_owner_flags_returns_empty_when_missing(monkeypatch, tmp_path):
    _, settings = setup_tmp_settings(monkeypatch, tmp_path)
    assert not settings.exists()
    out = features.get_owner_flags("owner-1")
    assert out == {}


def test_set_owner_flags_creates_file_and_merges(monkeypatch, tmp_path):
    data_dir, settings = setup_tmp_settings(monkeypatch, tmp_path)
    # First write
    cur = features.set_owner_flags("o1", feature_a=True, feature_b=1)
    assert cur == {"feature_a": True, "feature_b": True}  # coerced to bools
    assert settings.exists()

    # Merge on second call
    cur2 = features.set_owner_flags("o1", feature_b=False, feature_c="yes")
    # feature_b becomes False, feature_c coerced to True
    assert cur2 == {"feature_a": True, "feature_b": False, "feature_c": True}

    # Read back using get_owner_flags
    out = features.get_owner_flags("o1")
    assert out == cur2


def test_get_owner_flag_owner_overrides_env(monkeypatch, tmp_path):
    _, settings = setup_tmp_settings(monkeypatch, tmp_path)
    # Set env to true, but owner flag overrides to False
    monkeypatch.setenv("FEATURE_X", "true")
    features.set_owner_flags("o2", FEATURE_X=False)
    assert (
        features.get_owner_flag("o2", "FEATURE_X", "FEATURE_X", default=True) is False
    )


@pytest.mark.parametrize(
    "val,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        (None, True),  # default=True when env missing
    ],
)
def test_get_owner_flag_env_fallback(monkeypatch, tmp_path, val, expected):
    setup_tmp_settings(monkeypatch, tmp_path)
    env = "FEATURE_Y"
    if val is None:
        monkeypatch.delenv(env, raising=False)
    else:
        monkeypatch.setenv(env, val)
    # No owner flag set -> falls back to env/default
    got = features.get_owner_flag("o3", "FEATURE_Y", env, default=True)
    assert got is expected


def test_get_owner_flags_with_invalid_json_returns_empty(monkeypatch, tmp_path):
    data_dir, settings = setup_tmp_settings(monkeypatch, tmp_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    settings.write_text("{ this is not: json }", encoding="utf-8")
    out = features.get_owner_flags("o4")
    assert out == {}


def test_save_all_creates_directory_and_writes(monkeypatch, tmp_path):
    data_dir, settings = setup_tmp_settings(monkeypatch, tmp_path)
    # Directory does not exist initially
    assert not data_dir.exists()
    features._save_all({"o5": {"A": True}})
    # Directory and file are created
    assert data_dir.exists() and settings.exists()
    loaded = json.loads(settings.read_text(encoding="utf-8"))
    assert loaded == {"o5": {"A": True}}


def test_save_all_swallows_open_exception(monkeypatch, tmp_path):
    data_dir, settings = setup_tmp_settings(monkeypatch, tmp_path)

    class Boom(Exception):
        pass

    class DummySettings:
        def open(self, *a, **k):
            raise Boom("no write")

    # Replace SETTINGS_FILE with an object whose open() raises
    monkeypatch.setattr(features, "SETTINGS_FILE", DummySettings(), raising=True)

    # Should not raise even if writing fails
    features._save_all({"o6": {"B": True}})
    # Original path should still not exist
    assert not settings.exists()


def test_ensure_dir_handles_mkdir_exception(monkeypatch, tmp_path):
    setup_tmp_settings(monkeypatch, tmp_path)

    class Boom(Exception):
        pass

    called = {"mkdir": 0}

    class DummyDir:
        def mkdir(self, *a, **k):
            called["mkdir"] += 1
            raise Boom("no perms")

    # Replace DATA_DIR with dummy that raises from mkdir
    monkeypatch.setattr(features, "DATA_DIR", DummyDir(), raising=True)
    features._ensure_dir()  # should not raise
    assert called["mkdir"] == 1
