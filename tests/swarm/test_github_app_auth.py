from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.request import Request

from aragora.swarm import github_app_auth as mod


def test_load_github_app_config_reads_automation_env_file(tmp_path) -> None:
    key_path = tmp_path / "app.pem"
    key_path.write_text("PRIVATE KEY", encoding="utf-8")
    env_path = tmp_path / ".env.automation"
    env_path.write_text(
        "\n".join(
            [
                "GITHUB_APP_ID=123",
                "GITHUB_APP_INSTALLATION_ID=456",
                f"GITHUB_APP_PRIVATE_KEY_PATH={key_path}",
            ]
        ),
        encoding="utf-8",
    )

    config = mod.load_github_app_config({"ARAGORA_AUTOMATION_ENV_FILE": str(env_path)})

    assert config is not None
    assert config.app_id == "123"
    assert config.installation_id == "456"
    assert config.private_key == "PRIVATE KEY"
    assert config.private_key_source == str(key_path)


def test_load_github_app_config_reads_multiline_private_key(tmp_path) -> None:
    env_path = tmp_path / ".env.automation"
    env_path.write_text(
        "\n".join(
            [
                "GITHUB_APP_ID=123",
                "GITHUB_APP_INSTALLATION_ID=456",
                'GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----',
                "key-body",
                '-----END RSA PRIVATE KEY-----"',
            ]
        ),
        encoding="utf-8",
    )

    config = mod.load_github_app_config({"ARAGORA_AUTOMATION_ENV_FILE": str(env_path)})

    assert config is not None
    assert config.private_key == (
        "-----BEGIN RSA PRIVATE KEY-----\nkey-body\n-----END RSA PRIVATE KEY-----"
    )


def test_load_github_app_config_decodes_escaped_newline_private_key() -> None:
    config = mod.load_github_app_config(
        {
            "ARAGORA_AUTOMATION_ENV_FILE": "/missing",
            "GITHUB_APP_ID": "123",
            "GITHUB_APP_INSTALLATION_ID": "456",
            "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\\nkey-body\\n-----END RSA PRIVATE KEY-----",
        }
    )

    assert config is not None
    assert config.private_key == (
        "-----BEGIN RSA PRIVATE KEY-----\nkey-body\n-----END RSA PRIVATE KEY-----"
    )


def test_github_cli_env_prefers_installation_token(monkeypatch, tmp_path) -> None:
    mod.clear_github_app_token_cache()
    key_path = tmp_path / "app.pem"
    key_path.write_text("PRIVATE KEY", encoding="utf-8")
    env_path = tmp_path / ".env.automation"
    env_path.write_text(
        "\n".join(
            [
                "GITHUB_APP_ID=123",
                "GITHUB_APP_INSTALLATION_ID=456",
                f"GITHUB_APP_PRIVATE_KEY_PATH={key_path}",
            ]
        ),
        encoding="utf-8",
    )

    def fake_mint(config: mod.GitHubAppConfig) -> mod.GitHubAppToken:
        assert config.app_id == "123"
        return mod.GitHubAppToken(
            token="installation-token",
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=30),
        )

    monkeypatch.setattr(mod, "_mint_installation_token", fake_mint)

    env = mod.github_cli_env(
        {
            "ARAGORA_AUTOMATION_ENV_FILE": str(env_path),
            "GH_TOKEN": "user-token",
            "PATH": "/usr/bin",
        }
    )

    assert env["GH_TOKEN"] == "installation-token"
    assert env["GITHUB_TOKEN"] == "installation-token"
    assert env["ARAGORA_GITHUB_AUTH_SOURCE"] == "github_app_installation"


def test_github_cli_env_falls_back_when_unconfigured() -> None:
    mod.clear_github_app_token_cache()

    env = mod.github_cli_env({"PATH": "/usr/bin", "ARAGORA_AUTOMATION_ENV_FILE": "/missing"})

    assert env == {"PATH": "/usr/bin", "ARAGORA_AUTOMATION_ENV_FILE": "/missing"}


def test_validate_github_api_request_allows_github_https() -> None:
    request = Request("https://api.github.com/app/installations/456/access_tokens")

    mod._validate_github_api_request(request)


def test_validate_github_api_request_rejects_non_https() -> None:
    request = Request("http://api.github.com/app/installations/456/access_tokens")

    try:
        mod._validate_github_api_request(request)
    except RuntimeError as exc:
        assert "refusing non-GitHub API token request URL" in str(exc)
    else:  # pragma: no cover - explicit assertion branch for readability
        raise AssertionError("expected non-HTTPS GitHub API request to be rejected")


def test_validate_github_api_request_rejects_non_github_host() -> None:
    request = Request("https://example.com/app/installations/456/access_tokens")

    try:
        mod._validate_github_api_request(request)
    except RuntimeError as exc:
        assert "refusing non-GitHub API token request URL" in str(exc)
    else:  # pragma: no cover - explicit assertion branch for readability
        raise AssertionError("expected non-GitHub API request to be rejected")
