from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _frontend_dev_service() -> dict[str, object]:
    compose = yaml.safe_load((ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8"))
    return compose["services"]["frontend-dev"]


def test_frontend_dev_builds_local_sdk_before_starting() -> None:
    service = _frontend_dev_service()
    volumes = service["volumes"]
    command = service["command"]

    assert "./sdk/typescript:/sdk/typescript:ro" in volumes
    assert "/sdk/typescript/node_modules" in volumes
    assert "/sdk/typescript/dist" in volumes

    steps = (
        "npm --prefix /sdk/typescript ci --legacy-peer-deps",
        "npm --prefix /sdk/typescript run build",
        "npm install --legacy-peer-deps",
        "npm run dev",
    )
    positions = [command.index(step) for step in steps]
    assert positions == sorted(positions)
