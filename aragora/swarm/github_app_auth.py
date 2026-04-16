"""GitHub App installation-token support for automation-owned GitHub calls."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

DEFAULT_AUTOMATION_ENV_FILE = Path.home() / ".aragora" / ".env.automation"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_API_HOST = "api.github.com"
_TOKEN_CACHE: dict[tuple[str, str, str], "GitHubAppToken"] = {}
_PRIVATE_KEY_ENV_KEYS = ("GITHUB_APP_PRIVATE_KEY", "ARAGORA_GITHUB_APP_KEY")


@dataclass(frozen=True)
class GitHubAppConfig:
    app_id: str
    installation_id: str
    private_key: str
    private_key_source: str


@dataclass(frozen=True)
class GitHubAppToken:
    token: str
    expires_at: datetime


def clear_github_app_token_cache() -> None:
    _TOKEN_CACHE.clear()


def _strip_matching_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _parse_env_value(lines: list[str], start_index: int, raw_value: str) -> tuple[str, int]:
    value = raw_value.strip()
    if not value:
        return "", start_index

    quote = value[0]
    if quote not in {"'", '"'}:
        return _strip_matching_quotes(value), start_index

    current = value[1:]
    parts: list[str] = []
    index = start_index
    while True:
        if current.endswith(quote):
            parts.append(current[:-1])
            return "\n".join(parts), index
        parts.append(current)
        index += 1
        if index >= len(lines):
            return "\n".join(parts), index - 1
        current = lines[index].rstrip()


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            index += 1
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key], index = _parse_env_value(lines, index, value)
        index += 1
    return values


def _first_named_value(values: Mapping[str, str], keys: tuple[str, ...]) -> tuple[str, str]:
    for key in keys:
        value = str(values.get(key) or "").strip()
        if value:
            return key, value
    return "", ""


def _first_value(values: Mapping[str, str], keys: tuple[str, ...]) -> str:
    _, value = _first_named_value(values, keys)
    return value


def _normalize_private_key(value: str) -> str:
    stripped = _strip_matching_quotes(value)
    return stripped.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n").strip()


def _first_private_key(values: Mapping[str, str]) -> tuple[str, str]:
    key, value = _first_named_value(values, _PRIVATE_KEY_ENV_KEYS)
    if not value:
        return "", ""
    return key, _normalize_private_key(value)


def _automation_env_file(env: Mapping[str, str]) -> Path:
    configured = _first_value(
        env,
        (
            "ARAGORA_AUTOMATION_ENV_FILE",
            "GITHUB_APP_ENV_FILE",
            "ARAGORA_GITHUB_APP_ENV_FILE",
        ),
    )
    return Path(configured).expanduser() if configured else DEFAULT_AUTOMATION_ENV_FILE


def load_github_app_config(env: Mapping[str, str] | None = None) -> GitHubAppConfig | None:
    base_env = dict(os.environ if env is None else env)
    file_env = _read_env_file(_automation_env_file(base_env))
    values = {**file_env, **base_env}

    app_id = _first_value(values, ("GITHUB_APP_ID", "GH_APP_ID", "ARAGORA_GITHUB_APP_ID"))
    installation_id = _first_value(
        values,
        (
            "GITHUB_APP_INSTALLATION_ID",
            "GH_APP_INSTALLATION_ID",
            "ARAGORA_GITHUB_INSTALLATION_ID",
        ),
    )
    private_key_name, private_key = _first_private_key(values)
    private_key_source = f"env:{private_key_name}" if private_key else ""
    if not private_key:
        key_path = _first_value(
            values,
            (
                "GITHUB_APP_PRIVATE_KEY_PATH",
                "GH_APP_PRIVATE_KEY_PATH",
                "ARAGORA_GITHUB_APP_PRIVATE_KEY_PATH",
            ),
        )
        if key_path:
            path = Path(key_path).expanduser()
            if path.exists():
                private_key = path.read_text(encoding="utf-8")
                private_key_source = str(path)

    if not app_id or not installation_id or not private_key:
        return None
    return GitHubAppConfig(
        app_id=app_id,
        installation_id=installation_id,
        private_key=private_key,
        private_key_source=private_key_source,
    )


def _validate_github_api_request(request: urllib.request.Request) -> None:
    parsed = urllib.parse.urlparse(request.full_url)
    if parsed.scheme != "https" or parsed.netloc != GITHUB_API_HOST:
        raise RuntimeError(f"refusing non-GitHub API token request URL: {request.full_url}")


def _mint_installation_token(config: GitHubAppConfig) -> GitHubAppToken:
    import jwt

    now = int(time.time())
    encoded = jwt.encode(
        {
            "iat": now - 60,
            "exp": now + 540,
            "iss": config.app_id,
        },
        config.private_key,
        algorithm="RS256",
    )
    jwt_token = encoded.decode("utf-8") if isinstance(encoded, bytes) else str(encoded)
    request = urllib.request.Request(
        f"https://api.github.com/app/installations/{config.installation_id}/access_tokens",
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {jwt_token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    _validate_github_api_request(request)
    # The request URL is constructed internally and validated above.
    with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8") or "{}")
    token = str(payload.get("token") or "").strip()
    expires_at_raw = str(payload.get("expires_at") or "").strip()
    if not token or not expires_at_raw:
        raise RuntimeError("GitHub App installation token response was incomplete")
    expires_at = datetime.fromisoformat(expires_at_raw.replace("Z", "+00:00")).astimezone(UTC)
    return GitHubAppToken(token=token, expires_at=expires_at)


def get_github_app_installation_token(
    env: Mapping[str, str] | None = None,
    *,
    force_refresh: bool = False,
) -> str | None:
    base_env = dict(os.environ if env is None else env)
    if str(base_env.get("ARAGORA_DISABLE_GITHUB_APP_TOKEN") or "").strip() in {"1", "true", "yes"}:
        return None
    config = load_github_app_config(base_env)
    if config is None:
        return None

    cache_key = (config.app_id, config.installation_id, config.private_key_source)
    now = datetime.now(tz=UTC)
    cached = _TOKEN_CACHE.get(cache_key)
    if cached is not None and not force_refresh:
        if (cached.expires_at - now).total_seconds() > 120:
            return cached.token

    try:
        token = _mint_installation_token(config)
    except Exception:
        if str(base_env.get("ARAGORA_GITHUB_APP_AUTH_STRICT") or "").strip() in {
            "1",
            "true",
            "yes",
        }:
            raise
        return None
    _TOKEN_CACHE[cache_key] = token
    return token.token


def github_cli_env(
    base_env: Mapping[str, str] | None = None,
    *,
    prefer_app: bool = True,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if not prefer_app:
        return env
    token = get_github_app_installation_token(env)
    if not token:
        return env
    env["GH_TOKEN"] = token
    env["GITHUB_TOKEN"] = token
    env["ARAGORA_GITHUB_AUTH_SOURCE"] = "github_app_installation"
    return env
