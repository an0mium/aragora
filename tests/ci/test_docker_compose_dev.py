import json
import posixpath
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _compose() -> dict[str, object]:
    return yaml.safe_load((ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8"))


def test_frontend_dev_builds_local_sdk_before_starting() -> None:
    compose = _compose()
    service = compose["services"]["frontend-dev"]
    volumes = service["volumes"]
    command = " ".join(service["command"].split())

    assert service["image"] == "node:20.19-alpine"
    assert "typescript-sdk-dev:/sdk/typescript" in volumes
    assert "frontend-dev-npm-cache:/root/.npm" in volumes
    assert {
        "./sdk/typescript/.npmrc:/sdk/typescript/.npmrc:ro",
        "./sdk/typescript/package.json:/sdk/typescript/package.json:ro",
        "./sdk/typescript/package-lock.json:/sdk/typescript/package-lock.json:ro",
        "./sdk/typescript/tsconfig.json:/sdk/typescript/tsconfig.json:ro",
        "./sdk/typescript/src:/sdk/typescript/src:ro",
    }.issubset(volumes)
    assert "typescript-sdk-dev" in compose["volumes"]
    assert "frontend-dev-npm-cache" in compose["volumes"]

    assert command == (
        'sh -c "npm --prefix /sdk/typescript ci --legacy-peer-deps && '
        'npm --prefix /sdk/typescript run build && npm install && npm run dev"'
    )


def test_frontend_sdk_dependency_resolves_to_compose_mount() -> None:
    package = json.loads((ROOT / "aragora/live/package.json").read_text(encoding="utf-8"))
    specifier = package["dependencies"]["@aragora/sdk"]

    assert specifier.startswith("file:")
    assert posixpath.normpath(posixpath.join("/app", specifier.removeprefix("file:"))) == (
        "/sdk/typescript"
    )
