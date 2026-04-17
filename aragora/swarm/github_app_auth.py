"""GitHub App installation-token support for automation-owned GitHub calls."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_AUTOMATION_ENV_FILE = Path.home() / ".aragora" / ".env.automation"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_API_HOST = "api.github.com"
_TOKEN_CACHE: dict[tuple[str, str, str], "GitHubAppToken"] = {}


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
        value = value.strip()
        if not key:
            index += 1
            continue

        if value.startswith(("'", '"')):
            quote = value[0]
            value = value[1:]
            if value.endswith(quote):
                values[key] = value[:-1]
                index += 1
                continue

            parts = [value]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if next_line.endswith(quote):
                    parts.append(next_line[:-1])
                    index += 1
                    break
                parts.append(next_line)
                index += 1
            values[key] = "\n".join(parts)
            continue

        values[key] = value
        index += 1
    return values


def _first_value(values: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(values.get(key) or "").strip()
        if value:
            return value
    return ""


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


def _normalize_private_key(value: str) -> str:
    if "\\n" in value:
        return value.replace("\\n", "\n")
    return value


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
    private_key = _first_value(values, ("GITHUB_APP_PRIVATE_KEY", "ARAGORA_GITHUB_APP_KEY"))
    private_key_source = "env:GITHUB_APP_PRIVATE_KEY" if private_key else ""
    if private_key:
        private_key = _normalize_private_key(private_key)
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


# ---------------------------------------------------------------------------
# Rate-limit-aware gh CLI runner
# ---------------------------------------------------------------------------

_RATE_LIMIT_TOKENS = (
    "api rate limit already exceeded",
    "graphql: api rate limit",
    "secondary rate limit",
    "you have exceeded a secondary rate limit",
)


def is_rate_limit_error(stderr: str) -> bool:
    """True if `stderr` from `gh` indicates a primary or secondary rate-limit hit."""
    lowered = str(stderr or "").lower()
    return any(token in lowered for token in _RATE_LIMIT_TOKENS)


def _drop_app_token(env: dict[str, str]) -> dict[str, str]:
    if env.get("ARAGORA_GITHUB_AUTH_SOURCE") != "github_app_installation":
        return env
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    env.pop("ARAGORA_GITHUB_AUTH_SOURCE", None)
    return env


def _probe_quota(env: Mapping[str, str]) -> dict[str, int] | None:
    """Return a snapshot of `gh api rate_limit` resources, or None on failure.

    Output keys: ``{bucket}_remaining`` and ``{bucket}_reset`` for each bucket
    in ``core``, ``graphql``, ``search``.
    """
    try:
        result = subprocess.run(  # noqa: S603 - controlled gh invocation
            ["gh", "api", "rate_limit"],
            capture_output=True,
            text=True,
            timeout=10,
            env=dict(env),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    resources = dict(payload.get("resources") or {})
    snapshot: dict[str, int] = {}
    for bucket in ("core", "graphql", "search"):
        info = dict(resources.get(bucket) or {})
        remaining = info.get("remaining")
        reset = info.get("reset")
        if remaining is not None:
            snapshot[f"{bucket}_remaining"] = int(remaining)
        if reset is not None:
            snapshot[f"{bucket}_reset"] = int(reset)
    return snapshot or None


def _seconds_until_reset(
    env: Mapping[str, str],
    *,
    bucket: str = "core",
) -> float | None:
    """Best-effort estimate of seconds until the named bucket resets.

    Returns None when the quota cannot be probed; callers should fall back
    to exponential backoff in that case.
    """
    snapshot = _probe_quota(env)
    if snapshot is None:
        return None
    reset = snapshot.get(f"{bucket}_reset")
    if reset is None:
        return None
    delay = float(reset) - time.time()
    return max(delay, 0.0)


def _bucket_for_args(args: Sequence[str], stderr: str) -> str:
    if "graphql" in str(stderr or "").lower():
        return "graphql"
    if any(arg == "graphql" for arg in args):
        return "graphql"
    if any(arg == "search" for arg in args):
        return "search"
    return "core"


def gh_subprocess_run(
    args: Sequence[str],
    *,
    timeout: float = 30.0,
    prefer_app: bool = True,
    write_op: bool = False,
    env: Mapping[str, str] | None = None,
    max_retries: int = 3,
    base_backoff: float = 5.0,
    max_backoff: float = 600.0,
    sleep: Callable[[float], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh <args>`` with App-token preference and rate-limit-aware retry.

    - When ``prefer_app`` is True (the default) and ``write_op`` is False,
      the call is made with the GitHub App installation token via
      :func:`github_cli_env`, isolating it from the user PAT quota.
    - When ``write_op`` is True, the App token is dropped because the
      installation has narrow write scopes; the user PAT is preferred.
    - On rate-limit errors detected via ``is_rate_limit_error(stderr)``,
      sleeps until the relevant bucket resets (capped at ``max_backoff``)
      or by exponential backoff with jitter if the quota cannot be probed.
    - Retries up to ``max_retries`` times; returns the final
      :class:`subprocess.CompletedProcess` regardless of outcome.

    The sleep callable is injected for test isolation.
    """
    use_app = prefer_app and not write_op
    if env is None:
        run_env = github_cli_env(prefer_app=use_app)
    else:
        run_env = dict(env)
        if use_app:
            token = get_github_app_installation_token(run_env)
            if token:
                run_env["GH_TOKEN"] = token
                run_env["GITHUB_TOKEN"] = token
                run_env["ARAGORA_GITHUB_AUTH_SOURCE"] = "github_app_installation"
        else:
            run_env = _drop_app_token(run_env)

    sleep_fn = sleep if sleep is not None else time.sleep
    last_result: subprocess.CompletedProcess[str] | None = None
    attempt = 0
    while attempt <= max_retries:
        result = subprocess.run(  # noqa: S603 - controlled gh invocation
            ["gh", *list(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            check=False,
        )
        last_result = result
        if result.returncode == 0:
            return result
        if not is_rate_limit_error(result.stderr or ""):
            return result
        if attempt >= max_retries:
            break

        bucket = _bucket_for_args(args, result.stderr or "")
        reset_delay = _seconds_until_reset(run_env, bucket=bucket)
        if reset_delay is not None and reset_delay <= max_backoff:
            sleep_seconds = reset_delay + random.uniform(1.0, 5.0)
        else:
            sleep_seconds = min(
                base_backoff * (2**attempt) + random.uniform(0, base_backoff),
                max_backoff,
            )
        logger.warning(
            "gh rate-limit hit (attempt %d/%d, bucket=%s); sleeping %.1fs",
            attempt + 1,
            max_retries + 1,
            bucket,
            sleep_seconds,
        )
        sleep_fn(sleep_seconds)
        attempt += 1

    if last_result is None:  # pragma: no cover - retry loop always assigns
        raise RuntimeError("gh_subprocess_run produced no result")
    return last_result


def gh_subprocess_iter_buckets(
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Convenience wrapper for callers that want to inspect quota state.

    Returns a mapping ``{bucket: {"remaining": n, "reset": epoch}}`` for the
    core, graphql, and search buckets, or ``{}`` if the probe fails.
    """
    base = github_cli_env() if env is None else dict(env)
    snapshot = _probe_quota(base)
    if snapshot is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for bucket in ("core", "graphql", "search"):
        remaining = snapshot.get(f"{bucket}_remaining")
        reset = snapshot.get(f"{bucket}_reset")
        if remaining is None and reset is None:
            continue
        bucket_state: dict[str, int] = {}
        if remaining is not None:
            bucket_state["remaining"] = int(remaining)
        if reset is not None:
            bucket_state["reset"] = int(reset)
        out[bucket] = bucket_state
    return out
