"""Shared test fixtures.

FakeSupabaseClient matches REAL Supabase/PostgREST semantics closely
enough to catch real concurrency bugs -- specifically: every read and
write returns/stores an independent deep copy, the same way a real HTTP
round-trip always deserializes a fresh dict. This isn't a cosmetic
detail: an earlier version of this fake (built ad-hoc, not saved) shared
live object references between reads, and it produced a FALSE NEGATIVE
on a real race condition in native_mastery.py -- the bug only reproduced
once the fake was fixed to copy on every read/write like the real thing
does. Getting this right is the whole point of the fixture existing.

select(columns) also does real column-shape checking (raises if a
requested column isn't in the stored row), not just select("*") --
that's what makes test_health.py able to prove the exact
announced_in_chat bug from production is now caught.
"""
import copy

import pytest


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, store: dict, table: str):
        self.store = store
        self.table = table
        self.filters: list[tuple[str, str, object]] = []
        self._order = None
        self._limit = None
        self._op = None
        self._payload = None

    def select(self, columns: str = "*"):
        self._select_columns = columns
        return self

    def eq(self, key, value):
        self.filters.append(("eq", key, value))
        return self

    def neq(self, key, value):
        self.filters.append(("neq", key, value))
        return self

    def order(self, key, desc=False):
        self._order = (key, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def upsert(self, row):
        self._op, self._payload = "upsert", row
        return self

    def insert(self, row):
        self._op, self._payload = "insert", row
        return self

    def update(self, updates):
        self._op, self._payload = "update", updates
        return self

    def delete(self):
        self._op = "delete"
        return self

    def _match(self, row: dict) -> bool:
        for kind, key, value in self.filters:
            if kind == "eq" and row.get(key) != value:
                return False
            if kind == "neq" and row.get(key) == value:
                return False
        return True

    def _check_columns(self, row: dict) -> None:
        cols = getattr(self, "_select_columns", "*")
        if cols == "*":
            return
        for c in cols.split(","):
            if c not in row:
                raise Exception(f"column {self.table}.{c} does not exist")

    def execute(self):
        table = self.store.setdefault(self.table, {})
        if self._op == "upsert":
            row = copy.deepcopy(self._payload)
            key = row.get("id") or row.get("job_ref") or row.get("key")
            table[key] = row
            return _FakeResult([copy.deepcopy(row)])
        if self._op == "insert":
            row = copy.deepcopy(self._payload)
            row.setdefault("id", len(table) + 1)
            key = row.get("job_ref", row["id"])
            table[key] = row
            return _FakeResult([copy.deepcopy(row)])
        if self._op == "update":
            changed = []
            for row in table.values():
                if self._match(row):
                    row.update(self._payload)
                    changed.append(copy.deepcopy(row))
            return _FakeResult(changed)
        if self._op == "delete":
            to_remove = [k for k, r in table.items() if self._match(r)]
            for k in to_remove:
                del table[k]
            return _FakeResult([])

        rows = [copy.deepcopy(r) for r in table.values() if self._match(r)]
        for r in rows:
            self._check_columns(r)
        if self._order:
            key, desc = self._order
            rows.sort(key=lambda r: r.get(key) or "", reverse=desc)
        if self._limit:
            rows = rows[: self._limit]
        return _FakeResult(rows)


class FakeSupabaseClient:
    """Drop-in stand-in for config.get_client(). One instance = one
    isolated in-memory "database" for the life of a single test."""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self.store, name)


@pytest.fixture
def fake_db(monkeypatch):
    """Patches config.get_client() for the duration of one test. Also
    patches the plain key-value get_config/set_config/delete_config
    functions against the same in-memory store, since some modules
    (healing.py's seen-failures bookkeeping) still use that path."""
    import config

    client = FakeSupabaseClient()
    kv_store: dict = {}

    monkeypatch.setattr(config, "get_client", lambda: client)
    monkeypatch.setattr(config, "get_config", lambda k: kv_store.get(k))

    def _set_config(k, v):
        if v is None:
            raise ValueError("set_config(key, None) is not allowed -- use delete_config(key)")
        kv_store[k] = v

    monkeypatch.setattr(config, "set_config", _set_config)
    monkeypatch.setattr(config, "delete_config", lambda k: kv_store.pop(k, None))
    return client
