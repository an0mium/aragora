from pathlib import Path


def test_docker_entrypoint_invokes_the_migration_cli() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[2] / "deploy/scripts/docker-entrypoint.sh"
    ).read_text(encoding="utf-8")
    assert "python -m aragora.migrations upgrade" in entrypoint
    assert "python -m aragora.migrations.runner upgrade" not in entrypoint
