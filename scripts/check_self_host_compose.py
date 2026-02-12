#!/usr/bin/env python3
"""
Static validation for self-host production compose stack.

This is a lightweight CI guard that validates:
- required services exist in docker-compose.production.yml
- aragora depends_on includes postgres + 3 sentinels
- sentinel Redis env wiring is present
- required vars exist in .env.production.example
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production self-host compose configuration")
    parser.add_argument("--compose", default="docker-compose.production.yml")
    parser.add_argument("--env-example", default=".env.production.example")
    args = parser.parse_args()

    compose_path = Path(args.compose)
    env_path = Path(args.env_example)

    if not compose_path.exists():
        print(f"Compose file not found: {compose_path}", file=sys.stderr)
        return 2
    if not env_path.exists():
        print(f"Env example file not found: {env_path}", file=sys.stderr)
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
        dependency_names = set(str(item) for item in depends_on)
    else:
        dependency_names = set()

    required_dependencies = {"postgres", "sentinel-1", "sentinel-2", "sentinel-3"}
    missing_dependencies = sorted(required_dependencies - dependency_names)
    if missing_dependencies:
        errors.append(f"aragora service missing required dependencies: {missing_dependencies}")

    env_entries = aragora_service.get("environment", [])
    env_lines = [str(item) for item in env_entries] if isinstance(env_entries, list) else []

    if not any("ARAGORA_REDIS_MODE=" in line for line in env_lines):
        errors.append("aragora service missing ARAGORA_REDIS_MODE environment wiring")
    if not any("ARAGORA_REDIS_SENTINEL_HOSTS=" in line for line in env_lines):
        errors.append("aragora service missing ARAGORA_REDIS_SENTINEL_HOSTS environment wiring")

    env_keys = _parse_env_keys(env_path)
    missing_env_keys = sorted(REQUIRED_ENV_KEYS - env_keys)
    if missing_env_keys:
        errors.append(f".env production example missing required keys: {missing_env_keys}")

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
