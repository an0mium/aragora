"""Tests for scripts/check_self_host_runtime.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


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
    assert any(
        "ARAGORA_ENCRYPTION_KEY should be 64 hex characters" in warning for warning in warnings
    )
    assert any(
        "ARAGORA_SECRETS_STRICT=true may fail local runtime checks" in warning
        for warning in warnings
    )


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


def test_readiness_candidates_require_readyz() -> None:
    module = _load_script_module()

    assert module.READINESS_PATH_CANDIDATES == ["/readyz"]
    assert module.LIVENESS_PATH_CANDIDATES == ["/healthz"]


def test_default_core_services_wait_for_otel_before_aragora() -> None:
    module = _load_script_module()

    assert "otel-collector" in module.DEFAULT_CORE_SERVICES
    assert module.DEFAULT_CORE_SERVICES.index(
        "otel-collector"
    ) < module.DEFAULT_CORE_SERVICES.index("aragora")


def test_resolve_runtime_base_url_uses_container_ip_when_no_host_port() -> None:
    module = _load_script_module()

    with (
        patch.object(
            module,
            "_compose",
            side_effect=[
                _proc("", returncode=1, stderr="no host mapping"),
                _proc("aragora-container\n"),
            ],
        ),
        patch.object(module, "_run", return_value=_proc("172.18.0.9\n")),
    ):
        resolved = module._resolve_runtime_base_url(["docker", "compose"], "http://127.0.0.1:8080")

    assert resolved == "http://172.18.0.9:8080"


def test_resolve_runtime_base_url_keeps_requested_when_port_is_published() -> None:
    module = _load_script_module()

    with patch.object(module, "_compose", return_value=_proc("0.0.0.0:8080\n")):
        resolved = module._resolve_runtime_base_url(["docker", "compose"], "http://127.0.0.1:8080")

    assert resolved == "http://127.0.0.1:8080"


def test_resolve_runtime_base_url_uses_published_mapped_port() -> None:
    module = _load_script_module()

    with patch.object(module, "_compose", return_value=_proc("0.0.0.0:32788\n")):
        resolved = module._resolve_runtime_base_url(["docker", "compose"], "http://127.0.0.1:8080")

    assert resolved == "http://127.0.0.1:32788"


def test_resolve_runtime_base_url_ignores_zero_port_mapping_and_falls_back_to_container_ip() -> (
    None
):
    module = _load_script_module()

    with (
        patch.object(
            module,
            "_compose",
            side_effect=[
                _proc("0.0.0.0:0\n"),
                _proc("aragora-container\n"),
            ],
        ),
        patch.object(module, "_run", return_value=_proc("172.18.0.9\n")),
    ):
        resolved = module._resolve_runtime_base_url(["docker", "compose"], "http://127.0.0.1:8080")

    assert resolved == "http://172.18.0.9:8080"


def test_resolve_runtime_base_url_keeps_requested_when_resolution_fails() -> None:
    module = _load_script_module()

    with patch.object(
        module,
        "_compose",
        side_effect=[
            _proc("", returncode=1, stderr="no host mapping"),
            _proc("", returncode=1, stderr="compose unavailable"),
        ],
    ):
        resolved = module._resolve_runtime_base_url(["docker", "compose"], "http://127.0.0.1:8080")

    assert resolved == "http://127.0.0.1:8080"


def test_wait_for_http_200_retries_on_transient_connection_errors() -> None:
    module = _load_script_module()
    transient = module.RuntimeCheckError("HTTP request failed for http://127.0.0.1:8080/healthz")

    with (
        patch.object(module, "_http_request", side_effect=[transient, (200, "ok")]),
        patch.object(module.time, "sleep", return_value=None),
    ):
        module._wait_for_http_200("http://127.0.0.1:8080", "/healthz", timeout_seconds=10)


def test_wait_for_http_200_reports_last_connection_error_on_timeout() -> None:
    module = _load_script_module()
    transient = module.RuntimeCheckError("HTTP request failed for http://127.0.0.1:8080/healthz")

    with (
        patch.object(module, "_http_request", side_effect=transient),
        patch.object(module.time, "monotonic", side_effect=[0.0, 0.1, 1.1]),
        patch.object(module.time, "sleep", return_value=None),
    ):
        with pytest.raises(module.RuntimeCheckError, match="last_error=HTTP request failed"):
            module._wait_for_http_200("http://127.0.0.1:8080", "/healthz", timeout_seconds=1)


def test_wait_for_http_200_reports_last_http_body_on_timeout() -> None:
    module = _load_script_module()

    with (
        patch.object(
            module,
            "_http_request",
            return_value=(
                503,
                (
                    '{"status":"not_ready","reason":"Server in degraded mode",'
                    '"checks":{"degraded_mode":false,"startup_complete":true}}'
                ),
            ),
        ),
        patch.object(module.time, "monotonic", side_effect=[0.0, 0.1, 1.1]),
        patch.object(module.time, "sleep", return_value=None),
    ):
        with pytest.raises(
            module.RuntimeCheckError,
            match=(
                "last_status=503.*last_body=reason=Server in degraded mode; "
                "status=not_ready; failed_checks=degraded_mode"
            ),
        ):
            module._wait_for_http_200("http://127.0.0.1:8080", "/readyz", timeout_seconds=1)


def test_main_builds_services_before_waiting_for_health(tmp_path: Path) -> None:
    module = _load_script_module()
    compose_path = tmp_path / "docker-compose.production.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
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

    with (
        patch.object(
            module.sys,
            "argv",
            [
                "check_self_host_runtime.py",
                "--compose",
                str(compose_path),
                "--env-file",
                str(env_file),
                "--services",
                "aragora",
            ],
        ),
        patch.object(module, "_check_docker_daemon", return_value=None),
        patch.object(module, "_validate_runtime_env_file", return_value=([], [])),
        patch.object(module, "_read_env_value", return_value="api-token"),
        patch.object(module, "_resolve_runtime_base_url", return_value="http://127.0.0.1:8080"),
        patch.object(module, "_wait_for_service", return_value=None),
        patch.object(module, "_wait_for_any_http_200", side_effect=["/healthz", "/readyz"]),
        patch.object(module, "_check_api_flow", return_value=None),
        patch.object(module, "_print_diagnostics", return_value=None),
        patch.object(module, "_compose", return_value=_proc("")) as compose_mock,
    ):
        assert module.main() == 0

    compose_calls = [call.args[1] for call in compose_mock.call_args_list]
    assert ["up", "--build", "-d", "aragora"] in compose_calls


def test_check_api_flow_accepts_auth_required_response() -> None:
    module = _load_script_module()

    with patch.object(
        module,
        "_http_request",
        return_value=(401, '{"error":"Authentication required","code":"auth_required"}'),
    ):
        module._check_api_flow("http://127.0.0.1:8080", "ci-token")


def test_check_api_flow_raises_on_unexpected_get_status() -> None:
    module = _load_script_module()

    with patch.object(module, "_http_request", return_value=(500, "server error")):
        with pytest.raises(module.RuntimeCheckError, match="Expected GET /api/v1/debates"):
            module._check_api_flow("http://127.0.0.1:8080", "ci-token")
