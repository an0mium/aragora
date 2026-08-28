"""Lazy-initialization contract for the ``aragora.server.auth.auth_config`` singleton.

The module-level singleton must NOT parse ``AuthSettings`` (``get_settings().auth``)
at import time: with a cold settings cache and an invalid ``ARAGORA_*`` variable,
an import-time parse kills pytest at collection (``found no collectors``, rc=4)
instead of failing inside a test, misattributing env-pollution failures.

These tests pin two things:

1. Deferral (subprocess-based, cold settings cache): importing the module under a
   poisoned env succeeds, the parse happens on FIRST USE, and pytest collection of
   a module that imports ``aragora.server.auth`` no longer dies.
2. Preservation (in-process): the public name ``auth_config`` keeps working for
   from-imports, module attribute access, ``mock.patch`` /
   ``monkeypatch.setattr`` targets, and the module-level helpers
   (``check_auth``, ``generate_shareable_link``, ``resolve_shareable_session``)
   see a patched module attribute exactly as they did when the singleton was
   eagerly constructed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import aragora.server.auth as auth_mod

REPO_ROOT = Path(__file__).resolve().parents[2]

# Violates AuthSettings.token_ttl (ge=60) so the pydantic parse raises, while
# AuthConfig.configure_from_env's own int() parse of the same variable succeeds:
# the failure isolates to the get_settings().auth parse site.
_POISON_VAR = "ARAGORA_TOKEN_TTL"
_POISON_VALUE = "7"

# The historical collection victim: its module-level
# `from aragora.server.auth import ...` makes pytest import aragora.server.auth
# while collecting it.
_VICTIM_FILE = "tests/server/test_security.py"
_VICTIM_NODE = f"{_VICTIM_FILE}::TestConfigureFromEnv::test_configure_from_env_negative_ttl"


def _poisoned_env() -> dict[str, str]:
    """Inherited env minus all ARAGORA_* vars, plus the poison and safe defaults."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARAGORA_")}
    env.update(
        {
            # Importing aragora.server without AWS neutralization can hit
            # botocore MFA getpass on this machine class.
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "ARAGORA_ENV": "development",
            "ARAGORA_SECRETS_STRICT": "false",
            _POISON_VAR: _POISON_VALUE,
        }
    )
    return env


def _run_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestImportDefersSettingsParse:
    """Cold-cache subprocess proofs that the parse moved from import to first use."""

    def test_import_succeeds_under_poisoned_env_and_parse_defers_to_first_use(self):
        code = (
            "import sys\n"
            "import aragora.server.auth as auth_mod\n"
            "assert 'auth_config' not in vars(auth_mod), 'materialized at import'\n"
            "try:\n"
            "    auth_mod.auth_config\n"
            "except Exception:\n"
            "    sys.exit(0)\n"
            "sys.exit(3)\n"
        )
        proc = _run_python(code, _poisoned_env())
        assert proc.returncode == 0, (
            f"expected import to succeed and first use to raise under poisoned env; "
            f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    def test_first_use_constructs_configured_singleton_under_clean_env(self):
        env = _poisoned_env()
        env[_POISON_VAR] = "3600"
        env["ARAGORA_API_TOKEN"] = "lazy-init-proof-token"
        code = (
            "import aragora.server.auth as auth_mod\n"
            "assert 'auth_config' not in vars(auth_mod), 'materialized at import'\n"
            "from aragora.server.auth import auth_config\n"
            "assert auth_config.enabled, 'configure_from_env not applied on first use'\n"
            "assert auth_config is auth_mod.auth_config\n"
            "assert auth_config is auth_mod.get_auth_config()\n"
        )
        proc = _run_python(code, env)
        assert proc.returncode == 0, (
            f"expected lazy singleton to be configured on first use; "
            f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    def test_collection_of_auth_importing_module_survives_poisoned_env(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                _VICTIM_FILE,
                "--collect-only",
                "-q",
                "-p",
                "no:randomly",
            ],
            cwd=REPO_ROOT,
            env=_poisoned_env(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, (
            f"collection died under poisoned env (rc={proc.returncode}):\n{combined[-3000:]}"
        )
        assert "error during collection" not in combined, (
            f"collection reported errors under poisoned env:\n{combined[-3000:]}"
        )

    def test_node_id_selection_no_longer_reports_found_no_collectors(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                _VICTIM_NODE,
                "--collect-only",
                "-q",
                "-p",
                "no:randomly",
            ],
            cwd=REPO_ROOT,
            env=_poisoned_env(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        combined = proc.stdout + proc.stderr
        assert "found no collectors" not in combined, (
            f"the rc=4 'found no collectors' collection failure is back:\n{combined[-3000:]}"
        )
        assert proc.returncode == 0, (
            f"node-id collection failed under poisoned env (rc={proc.returncode}):\n{combined[-3000:]}"
        )


class TestPublicNamePreserved:
    """In-process guards: importers and patchers keep working against the lazy name."""

    def test_from_import_returns_the_singleton(self):
        from aragora.server.auth import auth_config as first
        from aragora.server.auth import auth_config as second

        assert first is second
        assert isinstance(first, auth_mod.AuthConfig)
        assert first is auth_mod.get_auth_config()

    def test_module_attribute_identity_is_stable(self):
        assert auth_mod.auth_config is auth_mod.auth_config

    def test_unknown_module_attribute_still_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            auth_mod.definitely_not_a_real_attribute

    def test_mock_patch_is_seen_by_check_auth_and_restores_identity(self):
        before = auth_mod.auth_config
        stub = mock.Mock()
        stub.enabled = False
        with mock.patch("aragora.server.auth.auth_config", stub):
            assert auth_mod.auth_config is stub
            authenticated, remaining = auth_mod.check_auth({})
            assert authenticated is True
            assert remaining == -1
        assert auth_mod.auth_config is before

    def test_monkeypatch_setattr_is_seen_and_restored(self, monkeypatch):
        before = auth_mod.auth_config
        sentinel = SimpleNamespace(enabled=False)
        monkeypatch.setattr(auth_mod, "auth_config", sentinel)
        authenticated, remaining = auth_mod.check_auth({})
        assert authenticated is True
        assert remaining == -1
        monkeypatch.undo()
        assert auth_mod.auth_config is before

    def test_shareable_link_helpers_use_patched_config(self):
        class _StubConfig:
            def generate_session(self, loop_id, expires_in):
                assert loop_id == "loop-1"
                return "sess-lazy-123"

            def resolve_session(self, session_id):
                assert session_id == "sess-lazy-123"
                return True, "tok", "loop-1"

        with mock.patch("aragora.server.auth.auth_config", _StubConfig()):
            link = auth_mod.generate_shareable_link("http://example.test/d", "loop-1")
            assert link == "http://example.test/d?session=sess-lazy-123"
            assert auth_mod.resolve_shareable_session("sess-lazy-123") == (
                True,
                "tok",
                "loop-1",
            )
