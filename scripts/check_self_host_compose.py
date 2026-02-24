#!/usr/bin/env python3
"""
Static validation for self-host production compose stack.

This is a lightweight CI guard that validates:
- required services exist in docker-compose.production.yml
- aragora depends_on includes postgres + 3 sentinels
- sentinel Redis env wiring is present
- required vars exist in .env.production.example
- session/rate-limit defaults are wired for distributed deployments
- self-host runbook includes startup/health/recovery sections
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


REQUIRED_SERVICES = {
    "aragora",
    "postgres",
    "redis-master",
    "redis-replica-1",
    "redis-replica-2",
    "sentinel-1",
    "sentinel-2",
    "sentinel-3",
}

REQUIRED_ENV_KEYS = {
    "DOMAIN",
    "POSTGRES_PASSWORD",
    "ARAGORA_API_TOKEN",
    "ARAGORA_JWT_SECRET",
    "ARAGORA_ENCRYPTION_KEY",
    "ARAGORA_RATE_LIMIT_BACKEND",
    "ARAGORA_REDIS_MODE",
    "ARAGORA_REDIS_SENTINEL_HOSTS",
    "ARAGORA_REDIS_SENTINEL_MASTER",
    "ARAGORA_STRICT_DEPLOYMENT",
}

REQUIRED_ARAGORA_ENV_PREFIXES = {
    "DATABASE_URL",
    "ARAGORA_DB_BACKEND",
    "ARAGORA_SECRETS_STRICT",
    "ARAGORA_REDIS_MODE",
    "ARAGORA_REDIS_SENTINEL_HOSTS",
    "ARAGORA_REDIS_SENTINEL_MASTER",
    "ARAGORA_JWT_SECRET",
    "ARAGORA_ENCRYPTION_KEY",
    "ARAGORA_RATE_LIMIT_BACKEND",
}

REQUIRED_HEALTHCHECK_SERVICES = {
    "aragora",
    "postgres",
    "redis-master",
    "redis-replica-1",
    "redis-replica-2",
    "sentinel-1",
    "sentinel-2",
    "sentinel-3",
}

REQUIRED_RUNBOOK_MARKERS = {
    "Startup and Readiness Verification",
    "Health Checks",
    "Failure Recovery Playbook",
}


def _parse_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", line)
        if m:
            keys.add(m.group(1))
    return keys


def _parse_service_env(service: dict[str, object]) -> dict[str, str]:
    raw_env = service.get("environment", [])
    parsed: dict[str, str] = {}
    if isinstance(raw_env, dict):
        for key, value in raw_env.items():
            parsed[str(key)] = "" if value is None else str(value)
        return parsed

    if isinstance(raw_env, list):
        for item in raw_env:
            text = str(item)
            if "=" not in text:
                continue
            key, value = text.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _contains_required_runbook_markers(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [marker for marker in sorted(REQUIRED_RUNBOOK_MARKERS) if marker not in text]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate production self-host compose configuration"
    )
    parser.add_argument("--compose", default="docker-compose.production.yml")
    parser.add_argument("--env-example", default=".env.production.example")
    parser.add_argument("--runbook", default="docs/SELF_HOSTING.md")
    args = parser.parse_args()

    compose_path = Path(args.compose)
    env_path = Path(args.env_example)
    runbook_path = Path(args.runbook)

    if not compose_path.exists():
        print(f"Compose file not found: {compose_path}", file=sys.stderr)
        return 2
    if not env_path.exists():
        print(f"Env example file not found: {env_path}", file=sys.stderr)
        return 2
    if not runbook_path.exists():
        print(f"Runbook file not found: {runbook_path}", file=sys.stderr)
        return 2

    try:
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"Failed to parse compose YAML: {exc}", file=sys.stderr)
        return 2

    services = compose.get("services")
    if not isinstance(services, dict):
        print("Compose file has no valid 'services' section", file=sys.stderr)
        return 2

    errors: list[str] = []

    missing_services = sorted(REQUIRED_SERVICES - set(services.keys()))
    if missing_services:
        errors.append(f"Missing required services: {missing_services}")

    aragora_service = services.get("aragora", {})
    depends_on = aragora_service.get("depends_on", {})
    if isinstance(depends_on, dict):
        dependency_names = set(depends_on.keys())
    elif isinstance(depends_on, list):
        dependency_names = {str(item) for item in depends_on}
    else:
        dependency_names = set()

    required_dependencies = {"postgres", "sentinel-1", "sentinel-2", "sentinel-3"}
    missing_dependencies = sorted(required_dependencies - dependency_names)
    if missing_dependencies:
        errors.append(f"aragora service missing required dependencies: {missing_dependencies}")

    parsed_aragora_env = _parse_service_env(aragora_service)

    missing_aragora_env = sorted(
        key for key in REQUIRED_ARAGORA_ENV_PREFIXES if key not in parsed_aragora_env
    )
    if missing_aragora_env:
        errors.append(f"aragora service missing required env wiring: {missing_aragora_env}")

    db_backend = parsed_aragora_env.get("ARAGORA_DB_BACKEND", "")
    if "postgres" not in db_backend:
        errors.append(
            "aragora service should set ARAGORA_DB_BACKEND=postgres for production compose"
        )

    rate_limit_backend = parsed_aragora_env.get("ARAGORA_RATE_LIMIT_BACKEND", "")
    if "redis" not in rate_limit_backend:
        errors.append(
            "aragora service should set ARAGORA_RATE_LIMIT_BACKEND=redis for distributed limits"
        )

    redis_mode = parsed_aragora_env.get("ARAGORA_REDIS_MODE", "")
    if "sentinel" not in redis_mode:
        errors.append("aragora service should set ARAGORA_REDIS_MODE=sentinel")

    missing_healthcheck = sorted(
        name
        for name in REQUIRED_HEALTHCHECK_SERVICES
        if name in services
        and isinstance(services[name], dict)
        and "healthcheck" not in services[name]
    )
    if missing_healthcheck:
        errors.append(f"services missing healthcheck configuration: {missing_healthcheck}")

    env_keys = _parse_env_keys(env_path)
    missing_env_keys = sorted(REQUIRED_ENV_KEYS - env_keys)
    if missing_env_keys:
        errors.append(f".env production example missing required keys: {missing_env_keys}")

    missing_markers = _contains_required_runbook_markers(runbook_path)
    if missing_markers:
        errors.append(f"self-host runbook missing required sections: {missing_markers}")

    if errors:
        print("Self-host compose validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        "Self-host compose validation passed "
        f"(services={len(services)}, required={len(REQUIRED_SERVICES)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
