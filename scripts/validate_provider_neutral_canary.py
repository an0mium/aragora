#!/usr/bin/env python3
"""Validate provider-neutral canary inputs without reading secret values."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path

from aragora.config.secrets import SecretManager, SecretSourceError, SecretsConfig

_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}$")
_REQUIRED_FILES = frozenset(
    {
        "DATABASE_URL",
        "REDIS_URL",
        "ARAGORA_API_TOKEN",
        "ARAGORA_JWT_SECRET",
        "ARAGORA_ENCRYPTION_KEY",
        "odr-signing-key.pem",
    }
)
_PROVIDER_FILES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "MISTRAL_API_KEY",
    }
)
_MAX_FILE_BYTES = 64 * 1024


def validate_image(image: str) -> list[str]:
    if not _IMAGE_RE.fullmatch(image):
        return ["ARAGORA_IMAGE must be an immutable registry reference with @sha256:<64 hex>"]
    return []


def _validate_file(path: Path, trusted_uids: set[int]) -> list[str]:
    errors: list[str] = []
    try:
        metadata = path.lstat()
    except OSError:
        return [f"missing required custody file: {path.name}"]
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISREG(metadata.st_mode):
        errors.append(f"custody file is not regular: {path.name}")
    if metadata.st_nlink != 1:
        errors.append(f"custody file is not single-link: {path.name}")
    if metadata.st_uid not in trusted_uids:
        errors.append(f"custody file has untrusted ownership: {path.name}")
    if not mode & stat.S_IRUSR or mode & ~0o600:
        errors.append(f"custody file permissions are not owner-readable/owner-only: {path.name}")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_FILE_BYTES:
        errors.append(f"custody file size is outside 1..{_MAX_FILE_BYTES} bytes: {path.name}")
    return errors


def validate_secrets_directory(path_text: str) -> tuple[list[str], list[str]]:
    path = Path(path_text)
    if not path.is_absolute():
        return ["ARAGORA_SECRETS_DIR_HOST must be absolute"], []
    try:
        metadata = path.lstat()
    except OSError:
        return ["ARAGORA_SECRETS_DIR_HOST does not exist"], []
    errors: list[str] = []
    mode = stat.S_IMODE(metadata.st_mode)
    trusted_uids = {0, os.geteuid()}
    if not stat.S_ISDIR(metadata.st_mode):
        errors.append("ARAGORA_SECRETS_DIR_HOST is not a directory")
        return errors, []
    try:
        directory_fd = SecretManager(SecretsConfig(secrets_dir=str(path)))._open_secrets_directory()
        os.close(directory_fd)
    except SecretSourceError as exc:
        errors.append(str(exc))
    if metadata.st_uid not in trusted_uids:
        errors.append("ARAGORA_SECRETS_DIR_HOST has untrusted ownership")
    if mode & 0o077:
        errors.append("ARAGORA_SECRETS_DIR_HOST must use owner-only permissions")

    try:
        present = sorted(entry.name for entry in path.iterdir())
    except OSError:
        errors.append("ARAGORA_SECRETS_DIR_HOST cannot be enumerated")
        return errors, []
    for name in sorted(_REQUIRED_FILES):
        errors.extend(_validate_file(path / name, trusted_uids))
    provider_names = sorted(_PROVIDER_FILES.intersection(present))
    if not provider_names:
        errors.append("at least one managed AI provider key file is required")
    else:
        for name in provider_names:
            errors.extend(_validate_file(path / name, trusted_uids))
    return errors, present


def build_report(image: str, secrets_dir: str) -> dict[str, object]:
    errors = validate_image(image)
    secret_errors, present = validate_secrets_directory(secrets_dir)
    errors.extend(secret_errors)
    return {
        "ok": not errors,
        "image_digest_pinned": not validate_image(image),
        "required_secret_names_present": sorted(_REQUIRED_FILES.intersection(present)),
        "provider_secret_names_present": sorted(_PROVIDER_FILES.intersection(present)),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--secrets-dir", required=True)
    args = parser.parse_args(argv)
    report = build_report(args.image, args.secrets_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
