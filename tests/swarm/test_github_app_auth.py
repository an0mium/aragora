from __future__ import annotations

import subprocess
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


# ---------------------------------------------------------------------------
# Rate-limit-aware gh subprocess wrapper
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_recognizes_known_signatures() -> None:
    assert mod.is_rate_limit_error("GraphQL: API rate limit already exceeded for user 1")
    assert mod.is_rate_limit_error("error: API rate limit already exceeded")
    assert mod.is_rate_limit_error("You have exceeded a secondary rate limit")
    assert not mod.is_rate_limit_error("error: not found")
    assert not mod.is_rate_limit_error("")


def test_gh_subprocess_run_returns_success_immediately(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true}', "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", lambda *, prefer_app=True: {"GH_TOKEN": "x"})

    result = mod.gh_subprocess_run(["pr", "list"])
    assert result.returncode == 0
    assert calls == [["gh", "pr", "list"]]


def test_gh_subprocess_run_does_not_retry_non_rate_limit_failures(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "error: not found")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", lambda *, prefer_app=True: {})

    result = mod.gh_subprocess_run(["pr", "view", "999"])
    assert result.returncode == 1
    assert calls == [["gh", "pr", "view", "999"]]  # only one attempt


def test_gh_subprocess_run_retries_on_rate_limit_then_succeeds(monkeypatch) -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def fake_run(cmd, **kwargs):
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            return subprocess.CompletedProcess(
                cmd, 1, "", "GraphQL: API rate limit already exceeded for user 1"
            )
        return subprocess.CompletedProcess(cmd, 0, "[]", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", lambda *, prefer_app=True: {})
    # Force the reset-time probe to fail so we exercise the exponential-backoff fallback
    monkeypatch.setattr(mod, "_seconds_until_reset", lambda env, *, bucket="core": None)

    result = mod.gh_subprocess_run(
        ["api", "graphql", "-f", "query=foo"],
        max_retries=4,
        base_backoff=1.0,
        sleep=lambda delay: sleeps.append(delay),
    )

    assert result.returncode == 0
    assert len(attempts) == 3
    assert len(sleeps) == 2
    assert all(delay > 0 for delay in sleeps)


def test_gh_subprocess_run_respects_reset_delay_when_known(monkeypatch) -> None:
    sleeps: list[float] = []

    def fake_run(cmd, **kwargs):
        if not sleeps:
            return subprocess.CompletedProcess(
                cmd, 1, "", "GraphQL: API rate limit already exceeded"
            )
        return subprocess.CompletedProcess(cmd, 0, "{}", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", lambda *, prefer_app=True: {})
    monkeypatch.setattr(mod, "_seconds_until_reset", lambda env, *, bucket="core": 30.0)

    result = mod.gh_subprocess_run(
        ["api", "graphql", "-f", "query=q"],
        max_retries=2,
        sleep=lambda delay: sleeps.append(delay),
    )

    assert result.returncode == 0
    assert len(sleeps) == 1
    assert 30.0 <= sleeps[0] <= 36.0  # reset_delay + jitter (1..5)


def test_gh_subprocess_run_returns_last_failure_after_max_retries(monkeypatch) -> None:
    sleeps: list[float] = []

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "GraphQL: API rate limit already exceeded")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", lambda *, prefer_app=True: {})
    monkeypatch.setattr(mod, "_seconds_until_reset", lambda env, *, bucket="core": None)

    result = mod.gh_subprocess_run(
        ["api", "graphql"],
        max_retries=2,
        base_backoff=0.5,
        sleep=lambda delay: sleeps.append(delay),
    )

    assert result.returncode == 1
    assert "rate limit" in result.stderr.lower()
    assert len(sleeps) == 2  # max_retries sleeps before giving up


def test_gh_subprocess_run_write_op_drops_app_token(monkeypatch) -> None:
    captured_envs: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        captured_envs.append(dict(kwargs.get("env") or {}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_env(*, prefer_app: bool = True):
        if prefer_app:
            return {
                "GH_TOKEN": "app",
                "GITHUB_TOKEN": "app",
                "ARAGORA_GITHUB_AUTH_SOURCE": "github_app_installation",
            }
        return {"GH_TOKEN": "user", "GITHUB_TOKEN": "user"}

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "github_cli_env", fake_env)

    mod.gh_subprocess_run(["issue", "edit", "1"], write_op=True)

    assert len(captured_envs) == 1
    env = captured_envs[0]
    assert env.get("GH_TOKEN") == "user"
    assert env.get("ARAGORA_GITHUB_AUTH_SOURCE") != "github_app_installation"


def test_bucket_for_args_detects_graphql_via_args() -> None:
    assert mod._bucket_for_args(["api", "graphql"], "") == "graphql"
    assert mod._bucket_for_args(["api", "search/issues"], "") == "core"
    assert mod._bucket_for_args(["pr", "list"], "") == "core"
    assert mod._bucket_for_args(["pr", "list"], "graphql: api rate limit") == "graphql"
