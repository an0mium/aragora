from __future__ import annotations

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
    assert validate_image(PINNED_IMAGE) == []


def test_relative_secret_directory_fails_closed() -> None:
    report = build_report(PINNED_IMAGE, "relative/secrets")
    assert report["ok"] is False
    assert "must be absolute" in str(report["errors"])


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
    assert "not regular" in errors


def _load_migration_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "deploy/provider-neutral/run_migrations.py"
    spec = importlib.util.spec_from_file_location("provider_neutral_run_migrations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_runner_requires_custody_hydration(monkeypatch) -> None:
    run_migrations = _load_migration_module()

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(run_migrations, "hydrate_env_from_secrets", lambda *args, **kwargs: {})

    try:
        run_migrations.main()
    except RuntimeError as exc:
        assert "managed custody" in str(exc)
    else:
        raise AssertionError("missing managed DATABASE_URL was accepted")


def test_migration_runner_does_not_print_database_url(monkeypatch, capsys) -> None:
    run_migrations = _load_migration_module()

    database_url = "postgresql://user:secret@example.invalid/aragora"

    def hydrate(*args, **kwargs):
        monkeypatch.setenv("DATABASE_URL", database_url)
        return {"DATABASE_URL": database_url}

    applied = [object(), object()]
    monkeypatch.setattr(run_migrations, "hydrate_env_from_secrets", hydrate)
    monkeypatch.setattr(run_migrations, "apply_migrations", lambda **kwargs: applied)

    assert run_migrations.main() == 0
    output = capsys.readouterr().out
    assert "Applied 2 migration(s)" in output
    assert database_url not in output
