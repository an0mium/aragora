"""Tests for aragora.utils.env.preserve_environ.

Regression coverage for #8277: the official ``rlm`` package calls
``dotenv.load_dotenv()`` at import time, which under pytest-xdist workers
resolves from the current working directory upward and can inject a
repository ``.env`` (e.g. ARAGORA_SECRETS_STRICT=true) into ``os.environ``
process-wide. ``preserve_environ`` wraps such imports so any environment
mutation is rolled back.
"""

from __future__ import annotations

import os

from aragora.utils.env import preserve_environ


def test_added_keys_are_removed() -> None:
    key = "PRESERVE_ENVIRON_TEST_ADDED"
    assert key not in os.environ
    with preserve_environ():
        os.environ[key] = "injected"
    assert key not in os.environ


def test_changed_keys_are_restored(monkeypatch) -> None:
    key = "PRESERVE_ENVIRON_TEST_CHANGED"
    monkeypatch.setenv(key, "original")
    with preserve_environ():
        os.environ[key] = "mutated"
    assert os.environ[key] == "original"


def test_deleted_keys_are_restored(monkeypatch) -> None:
    key = "PRESERVE_ENVIRON_TEST_DELETED"
    monkeypatch.setenv(key, "original")
    with preserve_environ():
        del os.environ[key]
    assert os.environ[key] == "original"


def test_restores_on_exception() -> None:
    key = "PRESERVE_ENVIRON_TEST_EXC"
    assert key not in os.environ
    try:
        with preserve_environ():
            os.environ[key] = "injected"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert key not in os.environ


def test_untouched_environment_is_unchanged() -> None:
    before = dict(os.environ)
    with preserve_environ():
        pass
    assert dict(os.environ) == before
