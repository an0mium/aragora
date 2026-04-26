"""
Secure LLM API key storage helpers for the Aragora CLI.

The CLI prefers an OS-backed store on macOS and falls back to an encrypted
local file store elsewhere. Stored keys are hydrated into the CLI process so
existing commands that rely on environment variables keep working unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STORE_VERSION = 1
KEYCHAIN_BACKEND = "macos-keychain"
FILE_BACKEND = "file"
SERVICE_NAME = "aragora-cli-llm-api-keys"
STORE_PATH_ENV = "ARAGORA_API_KEY_STORE_PATH"
BACKEND_ENV = "ARAGORA_API_KEY_BACKEND"


@dataclass(frozen=True)
class ProviderSpec:
    """Canonical metadata for a supported LLM provider."""

    name: str
    display_name: str
    env_vars: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def primary_env_var(self) -> str:
        return self.env_vars[0]


@dataclass(frozen=True)
class StoredKey:
    """Information about a stored provider key."""

    provider: str
    env_var: str
    backend: str
    masked_value: str


@dataclass(frozen=True)
class ProviderStatus:
    """Resolved status for a provider key."""

    provider: str
    display_name: str
    env_var: str
    configured: bool
    source: str
    masked_value: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for a provider key."""

    provider: str
    display_name: str
    env_var: str
    configured: bool
    source: str
    masked_value: str
    format_valid: bool
    remote_status: str
    is_valid: bool
    message: str


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        display_name="Anthropic",
        env_vars=("ANTHROPIC_API_KEY",),
        aliases=("anthropic-api", "claude"),
    ),
    "openai": ProviderSpec(
        name="openai",
        display_name="OpenAI",
        env_vars=("OPENAI_API_KEY",),
        aliases=("openai-api", "codex", "gpt"),
    ),
    "gemini": ProviderSpec(
        name="gemini",
        display_name="Gemini",
        env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        aliases=("google", "gemini-api"),
    ),
    "grok": ProviderSpec(
        name="grok",
        display_name="xAI / Grok",
        env_vars=("XAI_API_KEY", "GROK_API_KEY"),
        aliases=("xai", "grok-api"),
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        display_name="OpenRouter",
        env_vars=("OPENROUTER_API_KEY",),
    ),
    "mistral": ProviderSpec(
        name="mistral",
        display_name="Mistral",
        env_vars=("MISTRAL_API_KEY",),
        aliases=("mistral-api", "codestral"),
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        display_name="DeepSeek",
        env_vars=("DEEPSEEK_API_KEY",),
        aliases=("deepseek-v4-pro", "deepseek-v3", "deepseek-r1", "deepseek-reasoner"),
    ),
    "kimi": ProviderSpec(
        name="kimi",
        display_name="Kimi / Moonshot",
        env_vars=("KIMI_API_KEY",),
        aliases=("kimi-legacy", "moonshot"),
    ),
}

_ALIASES: dict[str, str] = {}
for _provider_name, _spec in PROVIDERS.items():
    for _alias in (_provider_name, *_spec.aliases):
        _ALIASES[_alias.lower()] = _provider_name


def get_supported_provider_names() -> list[str]:
    """Return the supported provider names."""
    return sorted(PROVIDERS.keys())


def resolve_provider(provider: str) -> ProviderSpec:
    """Resolve a provider name or alias to a canonical provider spec."""
    key = provider.strip().lower()
    resolved = _ALIASES.get(key)
    if resolved is None:
        supported = ", ".join(get_supported_provider_names())
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}")
    return PROVIDERS[resolved]


def mask_secret(value: str | None) -> str:
    """Mask a secret for display."""
    if not value:
        return "-"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def hydrate_env_from_secure_store(overwrite: bool = False) -> dict[str, str]:
    """Load stored provider keys into the current process environment."""
    hydrated: dict[str, str] = {}
    document = _load_store_document()
    for provider_name in document.get("providers", {}):
        spec = PROVIDERS.get(provider_name)
        if spec is None:
            continue

        if not overwrite and any(os.environ.get(env_var) for env_var in spec.env_vars):
            continue

        try:
            value = _get_stored_value(spec, document=document)
        except RuntimeError as exc:
            logger.warning("Could not load stored API key for %s: %s", provider_name, exc)
            continue

        if not value:
            continue

        os.environ[spec.primary_env_var] = value
        hydrated[spec.primary_env_var] = value

    return hydrated


def set_provider_key(provider: str, key: str) -> StoredKey:
    """Persist an LLM API key for a provider."""
    spec = resolve_provider(provider)
    normalized_key = key.strip()
    if not normalized_key:
        raise ValueError("API key must not be empty")

    backend = _detect_backend()
    document = _load_store_document()
    entry = {
        "backend": backend,
        "env_var": spec.primary_env_var,
        "updated_at": _utc_now(),
    }

    if backend == KEYCHAIN_BACKEND:
        _set_keychain_secret(spec.name, normalized_key)
    elif backend == FILE_BACKEND:
        entry["encrypted_value"] = _encrypt_for_file_store(normalized_key)
    else:
        raise RuntimeError(f"Unsupported API key backend '{backend}'")

    providers = document.setdefault("providers", {})
    providers[spec.name] = entry
    _save_store_document(document)
    os.environ[spec.primary_env_var] = normalized_key

    return StoredKey(
        provider=spec.name,
        env_var=spec.primary_env_var,
        backend=backend,
        masked_value=mask_secret(normalized_key),
    )


def get_provider_key(provider: str) -> tuple[str | None, str]:
    """Resolve a provider key from environment or the secure store."""
    spec = resolve_provider(provider)
    env_value, env_var = _get_env_value(spec)
    stored_value: str | None = None

    try:
        stored_value = _get_stored_value(spec)
    except RuntimeError as exc:
        logger.warning("Could not read stored API key for %s: %s", spec.name, exc)

    if env_value and stored_value and env_value != stored_value:
        return env_value, f"environment override ({env_var})"
    if stored_value:
        return stored_value, "secure-store"
    if env_value:
        return env_value, f"environment ({env_var})"
    return None, "not configured"


def list_provider_statuses() -> list[ProviderStatus]:
    """Return status for all supported providers."""
    document = _load_store_document()
    statuses: list[ProviderStatus] = []
    for provider_name in get_supported_provider_names():
        spec = PROVIDERS[provider_name]
        env_value, env_var = _get_env_value(spec)
        stored_value: str | None = None
        source = "not configured"

        try:
            stored_value = _get_stored_value(spec, document=document)
        except RuntimeError:
            if spec.name in document.get("providers", {}):
                source = "secure-store (unavailable)"

        if env_value and stored_value and env_value != stored_value:
            source = f"environment override ({env_var})"
        elif stored_value:
            source = "secure-store"
        elif env_value:
            source = f"environment ({env_var})"

        resolved_value = env_value or stored_value
        statuses.append(
            ProviderStatus(
                provider=spec.name,
                display_name=spec.display_name,
                env_var=spec.primary_env_var,
                configured=bool(resolved_value),
                source=source,
                masked_value=mask_secret(resolved_value),
            )
        )

    return statuses


def validate_provider_key(provider: str) -> ValidationReport:
    """Validate a provider key using local checks and a lightweight live probe."""
    spec = resolve_provider(provider)
    value, source = get_provider_key(spec.name)
    if not value:
        return ValidationReport(
            provider=spec.name,
            display_name=spec.display_name,
            env_var=spec.primary_env_var,
            configured=False,
            source=source,
            masked_value="-",
            format_valid=False,
            remote_status="skipped",
            is_valid=False,
            message=f"No API key configured for {spec.display_name}",
        )

    format_valid, format_message = _validate_key_format(spec, value)
    if not format_valid:
        return ValidationReport(
            provider=spec.name,
            display_name=spec.display_name,
            env_var=spec.primary_env_var,
            configured=True,
            source=source,
            masked_value=mask_secret(value),
            format_valid=False,
            remote_status="skipped",
            is_valid=False,
            message=format_message,
        )

    remote_status, remote_message = _probe_provider_key(spec, value)
    is_valid = remote_status in {"valid", "skipped"}

    return ValidationReport(
        provider=spec.name,
        display_name=spec.display_name,
        env_var=spec.primary_env_var,
        configured=True,
        source=source,
        masked_value=mask_secret(value),
        format_valid=True,
        remote_status=remote_status,
        is_valid=is_valid,
        message=remote_message,
    )


def _detect_backend() -> str:
    """Select the secure storage backend."""
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()
    if forced:
        if forced in {FILE_BACKEND, KEYCHAIN_BACKEND}:
            return forced
        raise RuntimeError(
            f"Unsupported {BACKEND_ENV} value '{forced}'. Expected '{FILE_BACKEND}' or '{KEYCHAIN_BACKEND}'."
        )

    if sys.platform == "darwin" and shutil.which("security"):
        return KEYCHAIN_BACKEND
    return FILE_BACKEND


def _get_store_path() -> Path:
    """Return the API key metadata path."""
    configured = os.environ.get(STORE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".aragora" / "api_keys.json"


def _get_key_path() -> Path:
    """Return the encrypted file-store key path."""
    return _get_store_path().with_suffix(".key")


def _load_store_document() -> dict:
    """Load the API key store metadata document."""
    path = _get_store_path()
    if not path.exists():
        return {"version": STORE_VERSION, "providers": {}}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"API key store is unreadable: {exc}") from exc


def _save_store_document(document: dict) -> None:
    """Save the API key store metadata document."""
    document["version"] = STORE_VERSION
    payload = json.dumps(document, indent=2, sort_keys=True)
    _write_private_text(_get_store_path(), payload)


def _get_env_value(spec: ProviderSpec) -> tuple[str | None, str]:
    """Return the first configured environment value for a provider."""
    for env_var in spec.env_vars:
        value = os.environ.get(env_var)
        if value:
            return value, env_var
    return None, spec.primary_env_var


def _get_stored_value(spec: ProviderSpec, document: dict | None = None) -> str | None:
    """Return the stored value for a provider, if any."""
    store_document = document or _load_store_document()
    entry = store_document.get("providers", {}).get(spec.name)
    if not entry:
        return None

    backend = entry.get("backend")
    if backend == KEYCHAIN_BACKEND:
        return _get_keychain_secret(spec.name)
    if backend == FILE_BACKEND:
        encrypted_value = entry.get("encrypted_value")
        if not encrypted_value:
            return None
        return _decrypt_from_file_store(encrypted_value)
    raise RuntimeError(f"Unknown API key backend '{backend}' for provider '{spec.name}'")


def _set_keychain_secret(account: str, value: str) -> None:
    """Store a secret in the macOS keychain."""
    result = subprocess.run(
        [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            SERVICE_NAME,
            "-a",
            account,
            "-w",
            value,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"macOS keychain write failed: {error}")


def _get_keychain_secret(account: str) -> str | None:
    """Read a secret from the macOS keychain."""
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            SERVICE_NAME,
            "-a",
            account,
            "-w",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "could not be found" in stderr or "not found" in stderr:
            return None
        error = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"macOS keychain read failed: {error}")
    return result.stdout.strip() or None


def _encrypt_for_file_store(value: str) -> str:
    """Encrypt a secret for the file-backed store."""
    service = _get_file_encryption_service()
    return service.encrypt(value).to_base64()


def _decrypt_from_file_store(ciphertext: str) -> str:
    """Decrypt a secret from the file-backed store."""
    service = _get_file_encryption_service()
    return service.decrypt_string(ciphertext)


def _get_file_encryption_service():
    """Build the encryption service used by the file-backed store."""
    try:
        from aragora.security.encryption import EncryptionService
    except ImportError as exc:
        raise RuntimeError("cryptography-backed encryption is not available") from exc

    return EncryptionService(master_key=_get_or_create_file_key())


def _get_or_create_file_key() -> bytes:
    """Load or create the local file-store master key."""
    key_path = _get_key_path()
    if key_path.exists():
        try:
            return bytes.fromhex(key_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"API key encryption key is unreadable: {exc}") from exc

    key_hex = secrets.token_hex(32)
    _write_private_text(key_path, key_hex)
    return bytes.fromhex(key_hex)


def _write_private_text(path: Path, content: str) -> None:
    """Write a private text file with user-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        logger.debug("Could not chmod %s to 700", path.parent)

    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    try:
        temp_path.chmod(0o600)
    except OSError:
        logger.debug("Could not chmod %s to 600", temp_path)

    temp_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        logger.debug("Could not chmod %s to 600", path)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _validate_key_format(spec: ProviderSpec, value: str) -> tuple[bool, str]:
    """Perform fast local validation before making a network request."""
    if not value.strip():
        return False, "API key must not be empty"
    if any(char.isspace() for char in value):
        return False, "API key must not contain whitespace"

    if spec.name == "anthropic" and not value.startswith("sk-ant-"):
        return False, "Anthropic API keys should start with 'sk-ant-'"
    if spec.name == "openrouter" and not value.startswith("sk-or-"):
        return False, "OpenRouter API keys should start with 'sk-or-'"
    if spec.name == "openai" and not value.startswith("sk-"):
        return False, "OpenAI API keys should start with 'sk-'"
    if len(value) < 12:
        return False, "API key is too short to be valid"

    return True, "Key format looks valid"


def _probe_provider_key(spec: ProviderSpec, value: str) -> tuple[str, str]:
    """Run a lightweight provider-specific key validation probe."""
    try:
        from aragora.security.safe_http import safe_get, safe_post
    except ImportError:
        return "skipped", "Stored key format looks valid; live validation is unavailable"

    try:
        if spec.name == "anthropic":
            response = safe_post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": value,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "openai":
            response = safe_get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {value}"},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "openrouter":
            response = safe_get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {value}"},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "gemini":
            response = safe_get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": value},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "grok":
            response = safe_get(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {value}"},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "mistral":
            response = safe_get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {value}"},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "kimi":
            response = safe_get(
                "https://api.moonshot.cn/v1/models",
                headers={"Authorization": f"Bearer {value}"},
                timeout=10.0,
            )
            return _status_from_response(response)

        if spec.name == "deepseek":
            return (
                "skipped",
                "DeepSeek keys are accepted for CLI/OpenRouter workflows; live validation is not implemented",
            )

        return "skipped", "No live validator implemented for this provider"
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
        return "error", f"Connection error during validation: {str(exc)[:80]}"


def _status_from_response(response) -> tuple[str, str]:
    """Translate an HTTP response into validation status."""
    status_code = getattr(response, "status_code", 0)
    if status_code == 200:
        return "valid", "API key is valid"
    if status_code in {400, 401, 403}:
        return "invalid", "Provider rejected the API key"
    return "error", f"Provider returned HTTP {status_code}"


__all__ = [
    "ProviderSpec",
    "ProviderStatus",
    "StoredKey",
    "ValidationReport",
    "get_provider_key",
    "get_supported_provider_names",
    "hydrate_env_from_secure_store",
    "list_provider_statuses",
    "mask_secret",
    "resolve_provider",
    "set_provider_key",
    "validate_provider_key",
]
