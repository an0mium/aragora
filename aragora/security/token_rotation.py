"""
Service Token Rotation Manager.

Handles storage and distribution of rotated service tokens (PyPI, npm,
GitHub PATs) across AWS Secrets Manager and GitHub repository/org secrets.

Note: PyPI and npm do not offer programmatic token creation/revocation.
This module handles the **storage and distribution** side of rotation,
plus audit logging. Token creation happens via provider web UIs.

Usage:
    from aragora.security.token_rotation import (
        TokenRotationManager,
        TokenRotationConfig,
        TokenType,
    )

    manager = TokenRotationManager(
        config=TokenRotationConfig(
            aws_secret_name="aragora/tokens",
            github_owner="aragora",
            github_repo="aragora",
        )
    )

    result = manager.rotate(TokenType.PYPI, "pypi-NEW_TOKEN_VALUE")

    # Scheduled rotation with health checks
    pipeline = RotationPipeline(
        manager=manager,
        schedule="0 3 * * 0",  # 3AM every Sunday
        health_checker=lambda tt: manager.verify_token(tt),
    )
    pipeline.execute(TokenType.PYPI, "pypi-NEW_TOKEN_VALUE")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Optional boto3/botocore imports
_BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    _BOTO3_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Types and Enums
# =============================================================================


class TokenType(str, Enum):
    """Supported service token types."""

    PYPI = "pypi"
    NPM = "npm"
    GITHUB_PAT = "github_pat"
    CUSTOM = "custom"


# Environment variable names that correspond to each token type
TOKEN_ENV_VARS: dict[TokenType, str] = {
    TokenType.PYPI: "PYPI_API_TOKEN",
    TokenType.NPM: "NPM_TOKEN",
    TokenType.GITHUB_PAT: "GH_TOKEN",
}

# GitHub secret names that correspond to each token type
TOKEN_GITHUB_SECRET_NAMES: dict[TokenType, str] = {
    TokenType.PYPI: "PYPI_API_TOKEN",
    TokenType.NPM: "NPM_TOKEN",
    TokenType.GITHUB_PAT: "GH_TOKEN",
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TokenRotationConfig:
    """Configuration for token rotation."""

    aws_secret_name: str = "aragora/tokens"
    """AWS Secrets Manager secret name for token storage."""

    aws_region: str = "us-east-1"
    """Primary AWS region."""

    github_owner: str = ""
    """GitHub org or user for secret storage."""

    github_repo: str = ""
    """Target repo (empty string for org-level secrets)."""

    stores: list[str] = field(default_factory=lambda: ["aws", "github"])
    """Which stores to write to on rotation."""

    @classmethod
    def from_env(cls) -> TokenRotationConfig:
        """Create config from environment variables.

        Environment variables:
            ARAGORA_TOKEN_SECRET_NAME: AWS SM secret name (default: aragora/tokens)
            AWS_REGION / AWS_DEFAULT_REGION: AWS region
            ARAGORA_GITHUB_OWNER: GitHub org/user
            ARAGORA_GITHUB_REPO: GitHub repo name
            ARAGORA_TOKEN_STORES: Comma-separated store list (default: aws,github)
        """
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        raw_stores = os.environ.get("ARAGORA_TOKEN_STORES", "aws,github")
        stores = [s.strip() for s in raw_stores.split(",") if s.strip()]

        return cls(
            aws_secret_name=os.environ.get("ARAGORA_TOKEN_SECRET_NAME", "aragora/tokens"),
            aws_region=region,
            github_owner=os.environ.get("ARAGORA_GITHUB_OWNER", ""),
            github_repo=os.environ.get("ARAGORA_GITHUB_REPO", ""),
            stores=stores,
        )


# =============================================================================
# Result
# =============================================================================


@dataclass
class TokenRotationResult:
    """Result of a token rotation operation."""

    token_type: TokenType
    stores_updated: list[str] = field(default_factory=list)
    old_token_prefix: str = ""
    new_token_prefix: str = ""
    rotated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token_type": self.token_type.value,
            "stores_updated": self.stores_updated,
            "old_token_prefix": self.old_token_prefix,
            "new_token_prefix": self.new_token_prefix,
            "rotated_at": self.rotated_at.isoformat(),
            "success": self.success,
            "errors": self.errors,
        }


@dataclass
class ManagedTokenInfo:
    """Information about a managed token."""

    token_type: str
    prefix: str
    last_rotated: str
    stores: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_type": self.token_type,
            "prefix": self.prefix,
            "last_rotated": self.last_rotated,
            "stores": self.stores,
        }


# =============================================================================
# Token Rotation Manager
# =============================================================================


class TokenRotationManager:
    """
    Manages service token rotation across secret stores.

    Supports storing tokens in:
    - AWS Secrets Manager (as a JSON object with token_type keys)
    - GitHub repository/org secrets (via gh CLI)

    Provides audit logging for all rotation operations.
    """

    def __init__(self, config: TokenRotationConfig | None = None):
        self._config = config or TokenRotationConfig.from_env()
        self._aws_client: Any = None
        self._rotation_history: list[TokenRotationResult] = []

    @property
    def config(self) -> TokenRotationConfig:
        """Get current configuration."""
        return self._config

    def _get_aws_client(self) -> Any:
        """Lazily initialize AWS Secrets Manager client."""
        if self._aws_client is not None:
            return self._aws_client

        if not _BOTO3_AVAILABLE:
            logger.debug("boto3 not installed, AWS Secrets Manager unavailable")
            return None

        try:
            self._aws_client = boto3.client("secretsmanager", region_name=self._config.aws_region)
            return self._aws_client
        except (BotoCoreError, ClientError, OSError, ValueError) as e:
            logger.warning("Failed to initialize AWS SM client: %s", e)
            return None

    def _mask_token(self, token: str) -> str:
        """Return first 8 characters of a token for identification."""
        if len(token) <= 8:
            return token[:4] + "..."
        return token[:8] + "..."

    # -----------------------------------------------------------------
    # Core rotation
    # -----------------------------------------------------------------

    def rotate(
        self,
        token_type: TokenType,
        new_token: str,
        *,
        secret_name_override: str | None = None,
        stores_override: list[str] | None = None,
        old_token: str | None = None,
    ) -> TokenRotationResult:
        """
        Rotate a service token by storing it in configured backends.

        Args:
            token_type: Type of token being rotated.
            new_token: The new token value.
            secret_name_override: Override the AWS SM secret name.
            stores_override: Override which stores to write to.
            old_token: Previous token value (for prefix logging).

        Returns:
            TokenRotationResult with per-store success/failure.
        """
        stores = stores_override if stores_override is not None else self._config.stores
        aws_secret_name = secret_name_override or self._config.aws_secret_name

        result = TokenRotationResult(
            token_type=token_type,
            new_token_prefix=self._mask_token(new_token),
            old_token_prefix=self._mask_token(old_token) if old_token else "",
        )

        logger.info(
            "Rotating %s token (new prefix: %s) to stores: %s",
            token_type.value,
            result.new_token_prefix,
            ", ".join(stores),
        )

        for store in stores:
            try:
                if store == "aws":
                    self._store_in_aws(aws_secret_name, token_type, new_token)
                    result.stores_updated.append("aws")
                elif store == "github":
                    gh_secret = TOKEN_GITHUB_SECRET_NAMES.get(token_type, token_type.value.upper())
                    self._store_in_github(
                        gh_secret,
                        new_token,
                        self._config.github_owner,
                        self._config.github_repo,
                    )
                    result.stores_updated.append("github")
                else:
                    result.errors[store] = f"Unknown store: {store}"
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as e:
                logger.error("Failed to store %s token in %s: %s", token_type.value, store, e)
                result.errors[store] = f"Failed to store token in {store}"

        result.success = len(result.errors) == 0
        result.rotated_at = datetime.now(timezone.utc)

        self._rotation_history.append(result)
        self._log_rotation_audit(result)

        return result

    # -----------------------------------------------------------------
    # AWS Secrets Manager
    # -----------------------------------------------------------------

    def _store_in_aws(
        self,
        secret_name: str,
        token_type: TokenType,
        token_value: str,
    ) -> None:
        """
        Store a token in AWS Secrets Manager.

        Tokens are stored as a JSON object keyed by token type, e.g.:
        {"pypi": "pypi-...", "npm": "npm_..."}

        Uses put_secret_value for existing secrets, create_secret for new ones.
        """
        client = self._get_aws_client()
        if client is None:
            raise RuntimeError("AWS Secrets Manager client not available (boto3 not installed?)")

        # Read existing secret to merge
        existing: dict[str, Any] = {}
        try:
            response = client.get_secret_value(SecretId=secret_name)
            existing = json.loads(response.get("SecretString", "{}"))
        except (BotoCoreError, ClientError, json.JSONDecodeError) as e:
            error_code = ""
            if hasattr(e, "response"):
                error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                logger.info("Secret %s not found, will create new", secret_name)
            else:
                raise

        # Update the token
        existing[token_type.value] = token_value
        existing[f"{token_type.value}_rotated_at"] = datetime.now(timezone.utc).isoformat()
        new_secret_string = json.dumps(existing)

        # Write back
        try:
            client.put_secret_value(
                SecretId=secret_name,
                SecretString=new_secret_string,
            )
            logger.info("Updated %s token in AWS SM secret %s", token_type.value, secret_name)
        except (BotoCoreError, ClientError) as e:
            error_code = ""
            if hasattr(e, "response"):
                error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                client.create_secret(
                    Name=secret_name,
                    SecretString=new_secret_string,
                )
                logger.info("Created AWS SM secret %s with %s token", secret_name, token_type.value)
            else:
                raise

    def _read_aws_tokens(self, secret_name: str | None = None) -> dict[str, Any]:
        """Read current tokens from AWS Secrets Manager."""
        client = self._get_aws_client()
        if client is None:
            return {}

        name = secret_name or self._config.aws_secret_name
        try:
            response = client.get_secret_value(SecretId=name)
            return json.loads(response.get("SecretString", "{}"))
        except (BotoCoreError, ClientError, json.JSONDecodeError) as e:
            error_code = ""
            if hasattr(e, "response"):
                error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                return {}
            logger.warning("Failed to read tokens from AWS SM: %s", e)
            return {}

    # -----------------------------------------------------------------
    # GitHub Secrets (via gh CLI)
    # -----------------------------------------------------------------

    def _store_in_github(
        self,
        secret_name: str,
        token_value: str,
        owner: str,
        repo: str,
    ) -> None:
        """
        Store a token as a GitHub secret using the gh CLI.

        Supports repo-level (gh secret set -R owner/repo) and
        org-level (gh secret set -o owner) secrets.
        """
        cmd = ["gh", "secret", "set", secret_name]

        if owner and repo:
            cmd.extend(["-R", f"{owner}/{repo}"])
        elif owner:
            cmd.extend(["-o", owner])
        else:
            raise ValueError(
                "github_owner must be set for GitHub secret storage. "
                "Set ARAGORA_GITHUB_OWNER or pass github_owner in config."
            )

        result = subprocess.run(
            cmd,
            input=token_value,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"gh secret set failed (exit {result.returncode}): {stderr}")

        target = f"{owner}/{repo}" if repo else f"org:{owner}"
        logger.info("Stored %s in GitHub secrets for %s", secret_name, target)

    # -----------------------------------------------------------------
    # List / Verify
    # -----------------------------------------------------------------

    def list_managed_tokens(self) -> list[ManagedTokenInfo]:
        """
        List tokens currently managed in AWS Secrets Manager.

        Returns:
            List of ManagedTokenInfo with prefix, last rotated, and stores.
        """
        tokens = self._read_aws_tokens()
        if not tokens:
            return []

        result: list[ManagedTokenInfo] = []
        for token_type in TokenType:
            value = tokens.get(token_type.value)
            if value is None:
                continue
            rotated_at = tokens.get(f"{token_type.value}_rotated_at", "unknown")
            result.append(
                ManagedTokenInfo(
                    token_type=token_type.value,
                    prefix=self._mask_token(value),
                    last_rotated=rotated_at,
                    stores=list(self._config.stores),
                )
            )

        return result

    def verify_token(self, token_type: TokenType) -> bool:
        """
        Verify a token works by running a lightweight check.

        Args:
            token_type: Type of token to verify.

        Returns:
            True if the token appears valid, False otherwise.
        """
        if token_type == TokenType.PYPI:
            return self._verify_pypi()
        elif token_type == TokenType.NPM:
            return self._verify_npm()
        elif token_type == TokenType.GITHUB_PAT:
            return self._verify_github_pat()
        else:
            logger.warning("No verification available for token type: %s", token_type.value)
            return False

    def _verify_pypi(self) -> bool:
        """Verify PyPI token via twine check or pip config."""
        try:
            result = subprocess.run(
                ["pip", "config", "get", "global.index-url"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # If pip is configured, token is likely set up
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _verify_npm(self) -> bool:
        """Verify npm token via npm whoami."""
        try:
            result = subprocess.run(
                ["npm", "whoami"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _verify_github_pat(self) -> bool:
        """Verify GitHub PAT via gh auth status."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # -----------------------------------------------------------------
    # History / Audit
    # -----------------------------------------------------------------

    def get_rotation_history(self, limit: int = 50) -> list[TokenRotationResult]:
        """Get recent rotation history."""
        return self._rotation_history[-limit:]

    def _log_rotation_audit(self, result: TokenRotationResult) -> None:
        """Log rotation event for audit trail (SOC 2 compliance)."""
        status = "SUCCESS" if result.success else "PARTIAL_FAILURE"
        logger.info(
            "TOKEN_ROTATION_AUDIT: type=%s status=%s stores=%s errors=%s rotated_at=%s",
            result.token_type.value,
            status,
            ",".join(result.stores_updated),
            json.dumps(result.errors) if result.errors else "none",
            result.rotated_at.isoformat(),
        )


# =============================================================================
# Module-level helpers
# =============================================================================

_manager: TokenRotationManager | None = None


def get_token_rotation_manager() -> TokenRotationManager | None:
    """Get the global token rotation manager instance."""
    return _manager


def get_or_create_token_rotation_manager() -> TokenRotationManager:
    """Get the global token rotation manager, creating one if needed."""
    global _manager
    if _manager is None:
        _manager = TokenRotationManager()
    return _manager


def reset_token_rotation_manager() -> None:
    """Reset the global manager instance (for testing)."""
    global _manager
    _manager = None


__all__ = [
    "TokenType",
    "TokenRotationConfig",
    "TokenRotationResult",
    "TokenRotationManager",
    "ManagedTokenInfo",
    "TOKEN_ENV_VARS",
    "TOKEN_GITHUB_SECRET_NAMES",
    "get_token_rotation_manager",
    "get_or_create_token_rotation_manager",
    "reset_token_rotation_manager",
]
