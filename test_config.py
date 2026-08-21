"""config.py: the set_config(key, None) crash and its fix.

Real production bug: sandy_config.value is a NOT NULL jsonb column;
set_config(key, None) upserts a literal SQL NULL and Postgres rejects it
with error 23502. Confirmed live, then fixed with a hard guard plus a
real delete_config(). These tests exercise config.py's REAL functions
(not a fixture stand-in for them) against a fake low-level client, so
they prove the actual guard clause works, not just that a test double
agrees with itself.
"""
import pytest

import config
from conftest import FakeSupabaseClient


@pytest.fixture
def real_config_over_fake_db(monkeypatch):
    """Patches config._db() specifically -- config.py's own set_config/
    get_config/delete_config bodies run for real against the fake."""
    client = FakeSupabaseClient()
    monkeypatch.setattr(config, "_db", lambda: client)
    return client


def test_set_config_none_is_rejected(real_config_over_fake_db):
    """The exact crash: set_config(key, None) must fail LOUD and
    IMMEDIATELY in Python, before ever reaching Postgres."""
    with pytest.raises(ValueError):
        config.set_config("some_key", None)


def test_set_config_accepts_real_values_and_get_config_reads_them_back(real_config_over_fake_db):
    """The guard must only reject None specifically -- confirmed by
    actually round-tripping legitimate falsy-but-real values through
    the real set_config -> get_config path."""
    for key, value in [("empty_dict", {}), ("empty_list", []), ("zero", 0), ("false", False)]:
        config.set_config(key, value)
        assert config.get_config(key) == value


def test_delete_config_actually_removes_the_row(real_config_over_fake_db):
    """delete_config is the correct way to clear a key -- after
    deleting, get_config must return None (key genuinely absent), same
    as if it had never been set."""
    config.set_config("temp_key", {"real": "value"})
    assert config.get_config("temp_key") == {"real": "value"}
    config.delete_config("temp_key")
    assert config.get_config("temp_key") is None
