"""Tests for the shared trusted-author allowlist resolver."""

from __future__ import annotations

from aragora.config.trusted_authors import (
    TRUSTED_AUTHORS_ENV,
    resolve_trusted_authors,
)


def test_defaults_only_when_env_empty() -> None:
    result = resolve_trusted_authors({"alpha[bot]", "beta[bot]"}, env={})
    assert result == frozenset({"alpha[bot]", "beta[bot]"})


def test_no_personal_default_means_empty_without_env() -> None:
    # The whole point: a bare resolver (no defaults, no env) trusts nobody.
    assert resolve_trusted_authors(env={}) == frozenset()


def test_env_adds_and_trims_tokens() -> None:
    result = resolve_trusted_authors(
        {"alpha[bot]"},
        env={TRUSTED_AUTHORS_ENV: " me ,  teammate "},
    )
    assert result == frozenset({"alpha[bot]", "me", "teammate"})


def test_empty_and_whitespace_tokens_dropped() -> None:
    result = resolve_trusted_authors(env={TRUSTED_AUTHORS_ENV: "a,, ,b,"})
    assert result == frozenset({"a", "b"})


def test_multiple_env_vars_are_unioned() -> None:
    result = resolve_trusted_authors(
        {"alpha[bot]"},
        env_vars=("GLOBAL", "SPECIFIC"),
        env={"GLOBAL": "g1", "SPECIFIC": "s1,s2"},
    )
    assert result == frozenset({"alpha[bot]", "g1", "s1", "s2"})


def test_missing_env_var_is_ignored() -> None:
    result = resolve_trusted_authors({"alpha[bot]"}, env_vars=("ABSENT",), env={})
    assert result == frozenset({"alpha[bot]"})


def test_defaults_are_trimmed_and_filtered() -> None:
    result = resolve_trusted_authors({" alpha[bot] ", "", "  "}, env={})
    assert result == frozenset({"alpha[bot]"})


def test_falls_back_to_os_environ(monkeypatch) -> None:
    monkeypatch.setenv(TRUSTED_AUTHORS_ENV, "from-os")
    assert "from-os" in resolve_trusted_authors()


def test_returns_frozenset() -> None:
    assert isinstance(resolve_trusted_authors({"a"}, env={}), frozenset)
