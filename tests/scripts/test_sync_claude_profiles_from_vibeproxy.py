from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "sync_claude_profiles_from_vibeproxy.py"
    )
    spec = importlib.util.spec_from_file_location("sync_claude_profiles_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = _load_module()

_VP = {
    "access_token": "vp-access",
    "refresh_token": "vp-refresh",
    "expired": "2026-07-22T18:27:18-05:00",
    "type": "claude",
    "disabled": False,
}


def test_blank_refresh_makes_a_pure_consumer_credential() -> None:
    out = sync.translate_credential(_VP, None, blank_refresh=True)
    oauth = out["claudeAiOauth"]
    assert oauth["accessToken"] == "vp-access"
    # No usable refresh token: aragora can never rotate VibeProxy's live token.
    assert oauth["refreshToken"] == ""


def test_keep_refresh_carries_the_vibeproxy_refresh_token() -> None:
    out = sync.translate_credential(_VP, None, blank_refresh=False)
    assert out["claudeAiOauth"]["refreshToken"] == "vp-refresh"


def test_expiry_is_translated_iso_to_epoch_ms() -> None:
    out = sync.translate_credential(_VP, None, blank_refresh=True)
    expected_ms = int(_dt.datetime.fromisoformat(_VP["expired"]).timestamp() * 1000)
    assert out["claudeAiOauth"]["expiresAt"] == expected_ms


def test_plan_and_scope_metadata_preserved_from_existing_credential() -> None:
    existing = {
        "scopes": ["user:inference", "custom:scope"],
        "subscriptionType": "team",
        "rateLimitTier": "default_claude_max_5x",
    }
    out = sync.translate_credential(_VP, existing, blank_refresh=True)
    oauth = out["claudeAiOauth"]
    assert oauth["scopes"] == ["user:inference", "custom:scope"]
    assert oauth["subscriptionType"] == "team"
    assert oauth["rateLimitTier"] == "default_claude_max_5x"


def test_defaults_applied_when_no_existing_credential() -> None:
    out = sync.translate_credential(_VP, None, blank_refresh=True)
    oauth = out["claudeAiOauth"]
    assert oauth["subscriptionType"] == sync._DEFAULT_SUBSCRIPTION
    assert oauth["scopes"] == sync._DEFAULT_SCOPES


def test_sync_profile_skips_when_no_vibeproxy_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sync, "VIBEPROXY_AUTH_DIR", tmp_path)  # empty dir
    result = sync.sync_profile("max-01", "absent@example.com", blank_refresh=True, apply=False)
    assert result.action == "skipped_no_source"


def test_sync_profile_is_idempotent_on_matching_access_token(monkeypatch, tmp_path) -> None:
    vp_dir = tmp_path / "vp"
    vp_dir.mkdir()
    (vp_dir / "claude-x@example.com.json").write_text(json.dumps(_VP), encoding="utf-8")
    prof_root = tmp_path / "profiles"
    cred = prof_root / "max-99" / ".claude" / ".credentials.json"
    cred.parent.mkdir(parents=True)
    cred.write_text(json.dumps({"claudeAiOauth": {"accessToken": "vp-access"}}), encoding="utf-8")
    monkeypatch.setattr(sync, "VIBEPROXY_AUTH_DIR", vp_dir)
    monkeypatch.setattr(sync, "ARAGORA_PROFILE_ROOT", prof_root)
    result = sync.sync_profile("max-99", "x@example.com", blank_refresh=True, apply=True)
    assert result.action == "skipped_fresh"


def test_sync_profile_writes_owner_only_credential(monkeypatch, tmp_path) -> None:
    vp_dir = tmp_path / "vp"
    vp_dir.mkdir()
    (vp_dir / "claude-x@example.com.json").write_text(json.dumps(_VP), encoding="utf-8")
    prof_root = tmp_path / "profiles"
    monkeypatch.setattr(sync, "VIBEPROXY_AUTH_DIR", vp_dir)
    monkeypatch.setattr(sync, "ARAGORA_PROFILE_ROOT", prof_root)
    result = sync.sync_profile("max-99", "x@example.com", blank_refresh=True, apply=True)
    assert result.action == "synced"
    cred = prof_root / "max-99" / ".claude" / ".credentials.json"
    assert (cred.stat().st_mode & 0o777) == 0o600
    written = json.loads(cred.read_text())
    assert written["claudeAiOauth"]["accessToken"] == "vp-access"
    assert written["claudeAiOauth"]["refreshToken"] == ""


def test_disabled_vibeproxy_account_is_not_synced(monkeypatch, tmp_path) -> None:
    vp_dir = tmp_path / "vp"
    vp_dir.mkdir()
    disabled = dict(_VP, disabled=True)
    (vp_dir / "claude-x@example.com.json").write_text(json.dumps(disabled), encoding="utf-8")
    monkeypatch.setattr(sync, "VIBEPROXY_AUTH_DIR", vp_dir)
    result = sync.sync_profile("max-99", "x@example.com", blank_refresh=True, apply=False)
    assert result.action == "skipped_no_source"
