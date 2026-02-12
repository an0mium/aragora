"""Tests for scripts/check_self_host_compose.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_self_host_compose.py"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ),
    )


def test_repo_compose_validation_passes():
    result = _run()
    assert result.returncode == 0
    assert "validation passed" in result.stdout.lower()


def test_fails_with_missing_services(tmp_path: Path):
    compose = tmp_path / "docker-compose.production.yml"
    compose.write_text(
        """
services:
  aragora:
    depends_on:
      postgres: {}
    environment:
      - ARAGORA_REDIS_MODE=sentinel
      - ARAGORA_REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
""".strip()
    )

    env = tmp_path / ".env.production.example"
    env.write_text(
        "\n".join(
            [
                "DOMAIN=example.com",
                "POSTGRES_PASSWORD=secret",
                "ARAGORA_API_TOKEN=token",
                "ARAGORA_JWT_SECRET=jwt",
            ]
        )
    )

    result = _run("--compose", str(compose), "--env-example", str(env))
    assert result.returncode == 1
    assert "missing required services" in result.stdout.lower()
