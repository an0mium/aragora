"""Tests for aragora.swarm.env_utils.git_safe_env edge cases."""

from aragora.swarm.env_utils import git_safe_env

_ALL_TOKEN_KEYS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)


def test_default_env_strips_all_token_keys(monkeypatch):
    """git_safe_env() with no args strips all four token keys from os.environ."""
    for key in _ALL_TOKEN_KEYS:
        monkeypatch.setenv(key, "secret-value")

    result = git_safe_env()

    for key in _ALL_TOKEN_KEYS:
        assert key not in result


def test_custom_base_env_with_all_tokens_stripped():
    """Explicit dict containing all four token keys is fully stripped."""
    base = dict.fromkeys(_ALL_TOKEN_KEYS, "tok")
    result = git_safe_env(base)

    for key in _ALL_TOKEN_KEYS:
        assert key not in result


def test_preserves_non_token_vars():
    """Non-token variables PATH, HOME, MY_VAR are preserved in result."""
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "MY_VAR": "hello",
        "GH_TOKEN": "secret",
    }
    result = git_safe_env(base)

    assert result["PATH"] == "/usr/bin:/bin"
    assert result["HOME"] == "/home/user"
    assert result["MY_VAR"] == "hello"


def test_no_mutation_of_input():
    """The original base_env dict is not modified by git_safe_env."""
    base = {"GH_TOKEN": "secret", "KEEP_ME": "value"}
    original_copy = dict(base)

    git_safe_env(base)

    assert base == original_copy


def test_empty_env_falls_back_to_os_environ(monkeypatch):
    """An empty dict is falsy, so git_safe_env falls back to os.environ.

    The implementation uses ``base_env or os.environ``, meaning an empty
    mapping triggers the same default-env path as no argument.  The result
    therefore contains os.environ minus any token keys.
    """
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("MY_SENTINEL", "present")

    result = git_safe_env({})

    # Token keys must be absent even when falling back to os.environ
    assert "GH_TOKEN" not in result
    # Non-token env vars from os.environ must be present
    assert result.get("MY_SENTINEL") == "present"


def test_partial_tokens_only_strips_present_key():
    """Only GH_TOKEN is stripped; other token keys are not added to the result."""
    base = {"GH_TOKEN": "secret", "MY_VAR": "keep"}
    result = git_safe_env(base)

    assert "GH_TOKEN" not in result
    assert result["MY_VAR"] == "keep"
    # Other token keys were never in base, so they must not appear in result either
    for key in ("GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        assert key not in result
