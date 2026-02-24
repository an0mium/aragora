"""Tests for scripts/check_self_host_runtime.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_self_host_runtime.py"
    spec = importlib.util.spec_from_file_location("check_self_host_runtime", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load check_self_host_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proc(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cmd"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_get_service_status_handles_multi_replica_healthy() -> None:
    module = _load_script_module()

    with (
        patch.object(module, "_compose", return_value=_proc("id1\nid2\n")),
        patch.object(module, "_run", side_effect=[_proc("healthy\n"), _proc("healthy\n")]),
    ):
        status, containers = module._get_service_status(["docker", "compose"], "aragora")

    assert status == "healthy"
    assert containers == "id1,id2"


def test_get_service_status_surfaces_unhealthy_replica() -> None:
    module = _load_script_module()

    with (
        patch.object(module, "_compose", return_value=_proc("id1\nid2\n")),
        patch.object(module, "_run", side_effect=[_proc("healthy\n"), _proc("unhealthy\n")]),
    ):
        status, container = module._get_service_status(["docker", "compose"], "aragora")

    assert status == "unhealthy"
    assert container == "id2"


def test_validate_runtime_env_file_reports_missing_required_keys(tmp_path: Path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env.production"
    env_file.write_text("ARAGORA_API_TOKEN=test-token\n", encoding="utf-8")

    errors, warnings = module._validate_runtime_env_file(env_file)

    assert any("POSTGRES_PASSWORD" in error for error in errors)
    assert any("ARAGORA_JWT_SECRET" in error for error in errors)
    assert any("ARAGORA_ENCRYPTION_KEY" in error for error in errors)
    assert warnings == []


def test_validate_runtime_env_file_validates_jwt_and_warns_on_strict_mode(tmp_path: Path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env.production"
    env_file.write_text(
        "\n".join(
            [
                "POSTGRES_PASSWORD=postgres-password",
                "ARAGORA_API_TOKEN=api-token",
                "ARAGORA_JWT_SECRET=short-secret",
                "ARAGORA_ENCRYPTION_KEY=not-hex",
                "ARAGORA_SECRETS_STRICT=true",
            ]
        ),
        encoding="utf-8",
    )

    errors, warnings = module._validate_runtime_env_file(env_file)

    assert any("ARAGORA_JWT_SECRET must be at least 32 characters" in error for error in errors)
    assert any("ARAGORA_ENCRYPTION_KEY should be 64 hex characters" in warning for warning in warnings)
    assert any("ARAGORA_SECRETS_STRICT=true may fail local runtime checks" in warning for warning in warnings)


def test_validate_runtime_env_file_accepts_valid_values(tmp_path: Path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env.production"
    env_file.write_text(
        "\n".join(
            [
                "POSTGRES_PASSWORD=postgres-password",
                "ARAGORA_API_TOKEN=api-token",
                "ARAGORA_JWT_SECRET=abcdefghijklmnopqrstuvwxyz123456",
                "ARAGORA_ENCRYPTION_KEY=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "ARAGORA_SECRETS_STRICT=false",
            ]
        ),
        encoding="utf-8",
    )

    errors, warnings = module._validate_runtime_env_file(env_file)

    assert errors == []
    assert warnings == []
