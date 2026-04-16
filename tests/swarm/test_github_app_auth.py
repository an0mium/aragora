from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.request import Request

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from aragora.swarm import github_app_auth as mod


def _rsa_private_key_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def _assert_private_key_signs(private_key: str) -> None:
    encoded = jwt.encode({"iss": "123"}, private_key, algorithm="RS256")
    assert encoded


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


def test_load_github_app_config_reads_multiline_private_key_from_env_file(tmp_path) -> None:
    private_key = _rsa_private_key_pem().rstrip("\n")
    env_path = tmp_path / ".env.automation"
    env_path.write_text(
        "\n".join(
            [
                "GITHUB_APP_ID=123",
                "GITHUB_APP_INSTALLATION_ID=456",
                f'GITHUB_APP_PRIVATE_KEY="{private_key}"',
            ]
        ),
        encoding="utf-8",
    )

    config = mod.load_github_app_config({"ARAGORA_AUTOMATION_ENV_FILE": str(env_path)})

    assert config is not None
    assert config.private_key == private_key
    assert config.private_key_source == "env:GITHUB_APP_PRIVATE_KEY"
    _assert_private_key_signs(config.private_key)


def test_load_github_app_config_decodes_escaped_newline_private_key_env() -> None:
    private_key = _rsa_private_key_pem()

    config = mod.load_github_app_config(
        {
            "GITHUB_APP_ID": "123",
            "GITHUB_APP_INSTALLATION_ID": "456",
            "ARAGORA_GITHUB_APP_KEY": private_key.replace("\n", "\\n"),
            "ARAGORA_AUTOMATION_ENV_FILE": "/missing",
        }
    )

    assert config is not None
    assert config.private_key == private_key.strip()
    assert config.private_key_source == "env:ARAGORA_GITHUB_APP_KEY"
    _assert_private_key_signs(config.private_key)


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


def test_github_cli_env_falls_back_when_inline_private_key_is_malformed(tmp_path) -> None:
    mod.clear_github_app_token_cache()
    env_path = tmp_path / ".env.automation"
    env_path.write_text(
        "\n".join(
            [
                "GITHUB_APP_ID=123",
                "GITHUB_APP_INSTALLATION_ID=456",
                'GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----"',
            ]
        ),
        encoding="utf-8",
    )

    env = mod.github_cli_env(
        {"PATH": "/usr/bin", "ARAGORA_AUTOMATION_ENV_FILE": str(env_path), "GH_TOKEN": "user"}
    )

    assert env == {
        "PATH": "/usr/bin",
        "ARAGORA_AUTOMATION_ENV_FILE": str(env_path),
        "GH_TOKEN": "user",
    }


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
