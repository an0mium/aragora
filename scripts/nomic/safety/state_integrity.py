"""
State file integrity protection for nomic loop.

Prevents tampering with state files that could allow:
- Skipping approval requirements
- Manipulating cycle counts
- Injecting malicious state data

Uses HMAC-SHA256 for integrity verification with a derived key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable for optional custom secret (for multi-machine deployments)
_STATE_SECRET_ENV = "NOMIC_STATE_SECRET"

# Default secret derivation source (machine-local)
_DEFAULT_SECRET_SOURCES = [
    "/etc/machine-id",  # Linux
    "/var/lib/dbus/machine-id",  # Linux fallback
]


def _get_state_secret() -> bytes:
    """Get or derive a secret for state file signing.

    Priority:
    1. Environment variable NOMIC_STATE_SECRET (for deployments)
    2. Machine ID file (Linux)
    3. Fallback to hostname + username hash (least secure)
    """
    # Check environment first
    env_secret = os.environ.get(_STATE_SECRET_ENV)
    if env_secret:
        return env_secret.encode("utf-8")

    # Try machine ID files
    for source_path in _DEFAULT_SECRET_SOURCES:
        try:
            with open(source_path) as f:
                machine_id = f.read().strip()
                if machine_id:
                    return hashlib.sha256(machine_id.encode()).digest()
        except (FileNotFoundError, PermissionError):
            continue

    # Fallback: derive from hostname + username (weakest option)
    import getpass
    import socket

    fallback_data = f"{socket.gethostname()}:{getpass.getuser()}:nomic-state-key"
    logger.warning(
        "[state-integrity] Using fallback secret derivation. "
        "Set NOMIC_STATE_SECRET for better security."
    )
    return hashlib.sha256(fallback_data.encode()).digest()


def compute_state_hmac(state_data: dict[str, Any]) -> str:
    """Compute HMAC-SHA256 for state data.

    Args:
        state_data: The state dictionary (without the hmac field)

    Returns:
        Hex-encoded HMAC signature
    """
    # Remove any existing hmac field to compute clean signature
    data_copy = {k: v for k, v in state_data.items() if k != "_hmac"}

    # Canonical JSON serialization (sorted keys, no spaces)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)

    secret = _get_state_secret()
    signature = hmac.new(secret, canonical.encode("utf-8"), hashlib.sha256)
    return signature.hexdigest()


def sign_state(state_data: dict[str, Any]) -> dict[str, Any]:
    """Add HMAC signature to state data.

    Args:
        state_data: The state dictionary to sign

    Returns:
        State dictionary with _hmac field added
    """
    # Compute HMAC without existing signature
    hmac_sig = compute_state_hmac(state_data)

    # Return new dict with signature
    signed = dict(state_data)
    signed["_hmac"] = hmac_sig
    return signed


def verify_state(state_data: dict[str, Any]) -> tuple[bool, str]:
    """Verify HMAC signature on state data.

    Args:
        state_data: The state dictionary to verify

    Returns:
        Tuple of (is_valid, error_message)
    """
    stored_hmac = state_data.get("_hmac")
    if not stored_hmac:
        return False, "State file missing integrity signature (_hmac)"

    expected_hmac = compute_state_hmac(state_data)

    if not hmac.compare_digest(stored_hmac, expected_hmac):
        return False, "State file integrity check failed - possible tampering detected"

    return True, ""


def save_state_secure(state_file: Path, state_data: dict[str, Any]) -> None:
    """Save state with integrity protection.

    Args:
        state_file: Path to the state file
        state_data: The state dictionary to save
    """
    signed_state = sign_state(state_data)

    # Write atomically via temp file
    temp_file = state_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(signed_state, f, indent=2, default=str)
        temp_file.replace(state_file)
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def load_state_secure(state_file: Path) -> tuple[dict[str, Any] | None, str]:
    """Load state with integrity verification.

    Args:
        state_file: Path to the state file

    Returns:
        Tuple of (state_data or None, error_message)
        On success, error_message is empty.
        On failure, state_data is None and error_message explains why.
    """
    if not state_file.exists():
        return None, "State file does not exist"

    try:
        with open(state_file) as f:
            state_data = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"State file is corrupted (invalid JSON): {e}"
    except PermissionError:
        return None, "Permission denied reading state file"
    except OSError as e:
        return None, f"Error reading state file: {e}"

    # Verify integrity
    is_valid, error = verify_state(state_data)
    if not is_valid:
        logger.error("[state-integrity] %s: %s", state_file, error)
        return None, error

    # Remove the hmac field before returning
    state_data.pop("_hmac", None)
    return state_data, ""


__all__ = [
    "compute_state_hmac",
    "load_state_secure",
    "save_state_secure",
    "sign_state",
    "verify_state",
]
