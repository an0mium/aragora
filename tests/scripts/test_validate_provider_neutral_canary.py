from __future__ import annotations

import os
from pathlib import Path

from scripts.validate_provider_neutral_canary import (
    _PROVIDER_FILES,
    _REQUIRED_FILES,
    build_report,
    validate_image,
)

PINNED_IMAGE = "ghcr.io/synaptent/aragora@sha256:" + "a" * 64


def _write(path: Path, value: str = "test-value") -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def _valid_secret_dir(tmp_path: Path) -> Path:
    tmp_path.chmod(0o700)
    for name in _REQUIRED_FILES:
        _write(tmp_path / name)
    _write(tmp_path / sorted(_PROVIDER_FILES)[0])
    return tmp_path


def test_valid_config_reports_names_without_values(tmp_path: Path) -> None:
    secret_dir = _valid_secret_dir(tmp_path)
    report = build_report(PINNED_IMAGE, str(secret_dir))

    assert report["ok"] is True
    assert set(report["required_secret_names_present"]) == _REQUIRED_FILES
    assert "test-value" not in str(report)


def test_image_must_be_digest_pinned() -> None:
    assert validate_image("ghcr.io/synaptent/aragora:latest")
    assert validate_image("aragora@sha256:" + "a" * 64)
    assert validate_image("ghcr.io/synaptent/aragora@sha256:" + "0" * 64)
    assert validate_image(PINNED_IMAGE) == []


def test_relative_secret_directory_fails_closed() -> None:
    report = build_report(PINNED_IMAGE, "relative/secrets")
    assert report["ok"] is False
    assert "must be absolute" in str(report["errors"])


def test_runtime_identity_must_match_custody_owner(tmp_path: Path) -> None:
    secret_dir = _valid_secret_dir(tmp_path)
    report = build_report(
        PINNED_IMAGE,
        str(secret_dir),
        runtime_uid=os.geteuid() + 1,
        runtime_gid=os.getegid() + 1,
    )
    assert report["ok"] is False
    assert "ownership does not match runtime UID/GID" in str(report["errors"])


def test_missing_required_secret_is_named_not_read(tmp_path: Path) -> None:
    secret_dir = _valid_secret_dir(tmp_path)
    (secret_dir / "DATABASE_URL").unlink()

    report = build_report(PINNED_IMAGE, str(secret_dir))

    assert report["ok"] is False
    assert "missing required custody file: DATABASE_URL" in report["errors"]


def test_requires_at_least_one_provider_key(tmp_path: Path) -> None:
    secret_dir = _valid_secret_dir(tmp_path)
    for name in _PROVIDER_FILES:
        path = secret_dir / name
        if path.exists():
            path.unlink()

    report = build_report(PINNED_IMAGE, str(secret_dir))

    assert report["ok"] is False
    assert "at least one managed AI provider key file is required" in report["errors"]


def test_rejects_open_permissions_and_symlink(tmp_path: Path) -> None:
    secret_dir = _valid_secret_dir(tmp_path)
    database = secret_dir / "DATABASE_URL"
    database.chmod(0o644)
    target = secret_dir / "provider-target"
    _write(target)
    provider = secret_dir / sorted(_PROVIDER_FILES)[0]
    provider.unlink()
    provider.symlink_to(target)

    report = build_report(PINNED_IMAGE, str(secret_dir))

    assert report["ok"] is False
    errors = str(report["errors"])
    assert "owner-readable/owner-only: DATABASE_URL" in errors
    assert "could not be opened safely" in errors


def _load_migration_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "deploy/provider-neutral/run_migrations.py"
    spec = importlib.util.spec_from_file_location("provider_neutral_run_migrations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_migration_manager(monkeypatch, run_migrations, tmp_path, database_url):
    class Manager:
        def __init__(self, config):
            self.config = config

        def _open_secrets_directory(self):
            return os.open(tmp_path, os.O_RDONLY)

        def _read_mounted_secret(self, directory_fd, name):
            assert name == "DATABASE_URL"
            return database_url

    monkeypatch.setattr(run_migrations, "SecretManager", Manager)
    monkeypatch.setattr(run_migrations.SecretsConfig, "from_env", lambda: object())


def test_migration_runner_requires_mounted_database_url(monkeypatch, tmp_path) -> None:
    run_migrations = _load_migration_module()
    _patch_migration_manager(monkeypatch, run_migrations, tmp_path, None)

    try:
        run_migrations.main()
    except RuntimeError as exc:
        assert "managed custody" in str(exc)
    else:
        raise AssertionError("missing managed DATABASE_URL was accepted")


def test_migration_runner_does_not_print_database_url(monkeypatch, capsys, tmp_path) -> None:
    run_migrations = _load_migration_module()
    database_url = "postgresql://user:secret@example.invalid/aragora"
    _patch_migration_manager(monkeypatch, run_migrations, tmp_path, database_url)
    applied = [object(), object()]
    monkeypatch.setattr(run_migrations, "wait_for_database", lambda *args: None)
    monkeypatch.setattr(run_migrations, "apply_migrations", lambda **kwargs: applied)

    assert run_migrations.main() == 0
    output = capsys.readouterr().out
    assert "Applied 2 migration(s)" in output
    assert database_url not in output


def test_database_wait_uses_parsed_host_without_credentials(monkeypatch) -> None:
    run_migrations = _load_migration_module()
    calls = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def connect(address, timeout):
        calls.append((address, timeout))
        return Connection()

    monkeypatch.setattr(run_migrations.socket, "create_connection", connect)
    run_migrations.wait_for_database("postgresql://user:secret@db.example:5433/app", 1)
    assert calls == [(("db.example", 5433), 2.0)]


def test_compose_mounts_migration_script_under_existing_app_path() -> None:
    import yaml

    compose_path = (
        Path(__file__).resolve().parents[2] / "deploy/provider-neutral/docker-compose.canary.yml"
    )
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    migrate = compose["services"]["migrate"]
    assert migrate["command"] == ["python", "/app/run_migrations.py"]
    assert "./run_migrations.py:/app/run_migrations.py:ro" in migrate["volumes"]
    assert compose["x-aragora-common"]["read_only"] is True
    assert compose["x-aragora-common"]["user"].startswith("${ARAGORA_RUNTIME_UID")
