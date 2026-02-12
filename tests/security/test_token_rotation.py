"""
Tests for Service Token Rotation Manager.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

from aragora.security.token_rotation import (
    TokenType,
    TokenRotationConfig,
    TokenRotationResult,
    TokenRotationManager,
    ManagedTokenInfo,
    TOKEN_ENV_VARS,
    TOKEN_GITHUB_SECRET_NAMES,
    get_token_rotation_manager,
    get_or_create_token_rotation_manager,
    reset_token_rotation_manager,
)


class TestTokenType:
    """Tests for TokenType enum."""

    def test_values(self):
        assert TokenType.PYPI == "pypi"
        assert TokenType.NPM == "npm"
        assert TokenType.GITHUB_PAT == "github_pat"
        assert TokenType.CUSTOM == "custom"

    def test_env_var_mapping(self):
        assert TOKEN_ENV_VARS[TokenType.PYPI] == "PYPI_API_TOKEN"
        assert TOKEN_ENV_VARS[TokenType.NPM] == "NPM_TOKEN"
        assert TOKEN_ENV_VARS[TokenType.GITHUB_PAT] == "GH_TOKEN"

    def test_github_secret_mapping(self):
        assert TOKEN_GITHUB_SECRET_NAMES[TokenType.PYPI] == "PYPI_API_TOKEN"
        assert TOKEN_GITHUB_SECRET_NAMES[TokenType.NPM] == "NPM_TOKEN"


class TestTokenRotationConfig:
    """Tests for TokenRotationConfig."""

    def test_default_values(self):
        config = TokenRotationConfig()
        assert config.aws_secret_name == "aragora/tokens"
        assert config.aws_region == "us-east-1"
        assert config.github_owner == ""
        assert config.github_repo == ""
        assert config.stores == ["aws", "github"]

    def test_custom_values(self):
        config = TokenRotationConfig(
            aws_secret_name="my/tokens",
            aws_region="eu-west-1",
            github_owner="myorg",
            github_repo="myrepo",
            stores=["aws"],
        )
        assert config.aws_secret_name == "my/tokens"
        assert config.aws_region == "eu-west-1"
        assert config.github_owner == "myorg"
        assert config.github_repo == "myrepo"
        assert config.stores == ["aws"]

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_TOKEN_SECRET_NAME", "prod/tokens")
        monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
        monkeypatch.setenv("ARAGORA_GITHUB_OWNER", "aragora")
        monkeypatch.setenv("ARAGORA_GITHUB_REPO", "aragora")
        monkeypatch.setenv("ARAGORA_TOKEN_STORES", "aws")

        config = TokenRotationConfig.from_env()
        assert config.aws_secret_name == "prod/tokens"
        assert config.aws_region == "ap-southeast-1"
        assert config.github_owner == "aragora"
        assert config.github_repo == "aragora"
        assert config.stores == ["aws"]

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_TOKEN_SECRET_NAME", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("ARAGORA_GITHUB_OWNER", raising=False)
        monkeypatch.delenv("ARAGORA_GITHUB_REPO", raising=False)
        monkeypatch.delenv("ARAGORA_TOKEN_STORES", raising=False)

        config = TokenRotationConfig.from_env()
        assert config.aws_secret_name == "aragora/tokens"
        assert config.aws_region == "us-east-1"
        assert config.stores == ["aws", "github"]

    def test_from_env_aws_default_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

        config = TokenRotationConfig.from_env()
        assert config.aws_region == "eu-central-1"


class TestTokenRotationResult:
    """Tests for TokenRotationResult."""

    def test_default_result(self):
        result = TokenRotationResult(token_type=TokenType.PYPI)
        assert result.token_type == TokenType.PYPI
        assert result.stores_updated == []
        assert result.success is True
        assert result.errors == {}
        assert result.old_token_prefix == ""
        assert result.new_token_prefix == ""

    def test_to_dict(self):
        result = TokenRotationResult(
            token_type=TokenType.NPM,
            stores_updated=["aws", "github"],
            old_token_prefix="npm_old_...",
            new_token_prefix="npm_new_...",
            success=True,
        )
        d = result.to_dict()
        assert d["token_type"] == "npm"
        assert d["stores_updated"] == ["aws", "github"]
        assert d["success"] is True
        assert "rotated_at" in d

    def test_to_dict_with_errors(self):
        result = TokenRotationResult(
            token_type=TokenType.PYPI,
            stores_updated=["aws"],
            errors={"github": "gh not found"},
            success=False,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["errors"] == {"github": "gh not found"}

    def test_prefix_masking(self):
        manager = TokenRotationManager(
            config=TokenRotationConfig(stores=[])
        )
        assert manager._mask_token("pypi-abcdef123456") == "pypi-abc..."
        assert manager._mask_token("short") == "shor..."
        assert manager._mask_token("exactly8") == "exactly8..."


class TestManagedTokenInfo:
    """Tests for ManagedTokenInfo."""

    def test_to_dict(self):
        info = ManagedTokenInfo(
            token_type="pypi",
            prefix="pypi-abc...",
            last_rotated="2026-02-12T16:00:00+00:00",
            stores=["aws", "github"],
        )
        d = info.to_dict()
        assert d["token_type"] == "pypi"
        assert d["prefix"] == "pypi-abc..."
        assert d["stores"] == ["aws", "github"]


class TestTokenRotationManagerAWS:
    """Tests for AWS Secrets Manager integration."""

    def _make_manager(self, **config_kwargs):
        config = TokenRotationConfig(
            stores=["aws"],
            github_owner="testorg",
            **config_kwargs,
        )
        return TokenRotationManager(config=config)

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_store_in_aws_existing_secret(self, mock_boto3):
        """Should merge token into existing AWS SM secret."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Existing secret has an npm token
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"npm": "npm_existing"})
        }

        manager = self._make_manager()
        manager._aws_client = mock_client

        manager._store_in_aws("aragora/tokens", TokenType.PYPI, "pypi-newtoken123")

        # Should write merged JSON
        put_call = mock_client.put_secret_value.call_args
        written = json.loads(put_call.kwargs["SecretString"])
        assert written["pypi"] == "pypi-newtoken123"
        assert written["npm"] == "npm_existing"
        assert "pypi_rotated_at" in written

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_store_in_aws_new_secret(self, mock_boto3):
        """Should create new secret if not found."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Secret doesn't exist
        not_found = type("ClientError", (Exception,), {
            "response": {"Error": {"Code": "ResourceNotFoundException"}}
        })()
        mock_client.get_secret_value.side_effect = not_found

        # put_secret_value also fails with not found -> create
        mock_client.put_secret_value.side_effect = not_found
        mock_client.create_secret.return_value = {}

        manager = self._make_manager()
        manager._aws_client = mock_client

        manager._store_in_aws("aragora/tokens", TokenType.PYPI, "pypi-new")

        mock_client.create_secret.assert_called_once()
        create_call = mock_client.create_secret.call_args
        written = json.loads(create_call.kwargs["SecretString"])
        assert written["pypi"] == "pypi-new"

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_rotate_aws_only(self, mock_boto3):
        """Full rotation with AWS store only."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}

        manager = self._make_manager()
        manager._aws_client = mock_client

        result = manager.rotate(TokenType.PYPI, "pypi-brandnew123")

        assert result.success is True
        assert "aws" in result.stores_updated
        assert result.new_token_prefix == "pypi-bra..."

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", False)
    def test_aws_not_available(self):
        """Should fail gracefully when boto3 not installed."""
        manager = self._make_manager()
        manager._aws_client = None

        result = manager.rotate(TokenType.PYPI, "pypi-test")

        assert result.success is False
        assert "aws" in result.errors
        assert "not available" in result.errors["aws"].lower() or "boto3" in result.errors["aws"].lower()

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_read_aws_tokens(self, mock_boto3):
        """Should read and parse tokens from AWS SM."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({
                "pypi": "pypi-abc123",
                "pypi_rotated_at": "2026-02-12T00:00:00+00:00",
            })
        }

        manager = self._make_manager()
        manager._aws_client = mock_client

        tokens = manager._read_aws_tokens()
        assert tokens["pypi"] == "pypi-abc123"

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_read_aws_tokens_not_found(self, mock_boto3):
        """Should return empty dict when secret not found."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        not_found = type("ClientError", (Exception,), {
            "response": {"Error": {"Code": "ResourceNotFoundException"}}
        })()
        mock_client.get_secret_value.side_effect = not_found

        manager = self._make_manager()
        manager._aws_client = mock_client

        tokens = manager._read_aws_tokens()
        assert tokens == {}


class TestTokenRotationManagerGitHub:
    """Tests for GitHub Secrets integration."""

    def _make_manager(self, **config_kwargs):
        defaults = {
            "stores": ["github"],
            "github_owner": "testorg",
            "github_repo": "testrepo",
        }
        defaults.update(config_kwargs)
        return TokenRotationManager(config=TokenRotationConfig(**defaults))

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_store_in_github_repo_level(self, mock_run):
        """Should call gh secret set -R for repo-level secrets."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        manager = self._make_manager()
        manager._store_in_github("PYPI_API_TOKEN", "pypi-test", "testorg", "testrepo")

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd == ["gh", "secret", "set", "PYPI_API_TOKEN", "-R", "testorg/testrepo"]
        assert mock_run.call_args.kwargs["input"] == "pypi-test"

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_store_in_github_org_level(self, mock_run):
        """Should call gh secret set -o for org-level secrets."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        manager = self._make_manager(github_repo="")
        manager._store_in_github("PYPI_API_TOKEN", "pypi-test", "testorg", "")

        cmd = mock_run.call_args.args[0]
        assert cmd == ["gh", "secret", "set", "PYPI_API_TOKEN", "-o", "testorg"]

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_store_in_github_failure(self, mock_run):
        """Should raise on gh CLI failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="auth required")

        manager = self._make_manager()
        with pytest.raises(RuntimeError, match="gh secret set failed"):
            manager._store_in_github("PYPI_API_TOKEN", "pypi-test", "testorg", "testrepo")

    def test_store_in_github_no_owner(self):
        """Should raise when no owner configured."""
        manager = self._make_manager(github_owner="", github_repo="")
        with pytest.raises(ValueError, match="github_owner must be set"):
            manager._store_in_github("TOKEN", "value", "", "")

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_rotate_github_only(self, mock_run):
        """Full rotation with GitHub store only."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        manager = self._make_manager()
        result = manager.rotate(TokenType.PYPI, "pypi-newtoken")

        assert result.success is True
        assert "github" in result.stores_updated
        mock_run.assert_called_once()


class TestTokenRotationManagerMultiStore:
    """Tests for multi-store rotation."""

    @patch("aragora.security.token_rotation.subprocess.run")
    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_rotate_both_stores(self, mock_boto3, mock_run):
        """Should store in both AWS and GitHub."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        config = TokenRotationConfig(
            stores=["aws", "github"],
            github_owner="myorg",
            github_repo="myrepo",
        )
        manager = TokenRotationManager(config=config)
        manager._aws_client = mock_client

        result = manager.rotate(TokenType.NPM, "npm_newtoken123")

        assert result.success is True
        assert "aws" in result.stores_updated
        assert "github" in result.stores_updated

    @patch("aragora.security.token_rotation.subprocess.run")
    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_partial_failure(self, mock_boto3, mock_run):
        """AWS succeeds but GitHub fails -> partial failure."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}
        mock_run.return_value = MagicMock(returncode=1, stderr="not authenticated")

        config = TokenRotationConfig(
            stores=["aws", "github"],
            github_owner="myorg",
            github_repo="myrepo",
        )
        manager = TokenRotationManager(config=config)
        manager._aws_client = mock_client

        result = manager.rotate(TokenType.PYPI, "pypi-test")

        assert result.success is False
        assert "aws" in result.stores_updated
        assert "github" in result.errors

    def test_unknown_store(self):
        """Should report error for unknown store."""
        config = TokenRotationConfig(stores=["unknown_store"])
        manager = TokenRotationManager(config=config)

        result = manager.rotate(TokenType.PYPI, "pypi-test")

        assert result.success is False
        assert "unknown_store" in result.errors

    def test_stores_override(self):
        """Should respect stores_override parameter."""
        config = TokenRotationConfig(stores=["aws", "github"])
        manager = TokenRotationManager(config=config)

        # Override to empty stores -> nothing to do, success
        result = manager.rotate(TokenType.PYPI, "pypi-test", stores_override=[])
        assert result.success is True
        assert result.stores_updated == []


class TestTokenRotationManagerListVerify:
    """Tests for list and verify operations."""

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_list_managed_tokens(self, mock_boto3):
        """Should list tokens from AWS SM."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({
                "pypi": "pypi-abcdefgh12345",
                "pypi_rotated_at": "2026-02-12T00:00:00+00:00",
                "npm": "npm_xyz789",
                "npm_rotated_at": "2026-02-11T00:00:00+00:00",
            })
        }

        manager = TokenRotationManager(
            config=TokenRotationConfig(stores=["aws"])
        )
        manager._aws_client = mock_client

        tokens = manager.list_managed_tokens()
        assert len(tokens) == 2
        types = [t.token_type for t in tokens]
        assert "pypi" in types
        assert "npm" in types

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_list_managed_tokens_empty(self, mock_boto3):
        """Should return empty list when no tokens stored."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}

        manager = TokenRotationManager(config=TokenRotationConfig(stores=["aws"]))
        manager._aws_client = mock_client

        tokens = manager.list_managed_tokens()
        assert tokens == []

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_verify_github_pat(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        assert manager.verify_token(TokenType.GITHUB_PAT) is True

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_verify_github_pat_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        assert manager.verify_token(TokenType.GITHUB_PAT) is False

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_verify_npm(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        assert manager.verify_token(TokenType.NPM) is True

    @patch("aragora.security.token_rotation.subprocess.run")
    def test_verify_pypi(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        assert manager.verify_token(TokenType.PYPI) is True

    def test_verify_custom_unsupported(self):
        manager = TokenRotationManager(config=TokenRotationConfig(stores=[]))
        assert manager.verify_token(TokenType.CUSTOM) is False


class TestTokenRotationManagerHistory:
    """Tests for rotation history and audit."""

    def test_rotation_history(self):
        config = TokenRotationConfig(stores=[])
        manager = TokenRotationManager(config=config)

        manager.rotate(TokenType.PYPI, "pypi-one")
        manager.rotate(TokenType.NPM, "npm-two")

        history = manager.get_rotation_history()
        assert len(history) == 2
        assert history[0].token_type == TokenType.PYPI
        assert history[1].token_type == TokenType.NPM

    def test_rotation_history_limit(self):
        config = TokenRotationConfig(stores=[])
        manager = TokenRotationManager(config=config)

        for i in range(10):
            manager.rotate(TokenType.PYPI, f"pypi-{i}")

        history = manager.get_rotation_history(limit=3)
        assert len(history) == 3

    def test_old_token_prefix_in_result(self):
        config = TokenRotationConfig(stores=[])
        manager = TokenRotationManager(config=config)

        result = manager.rotate(
            TokenType.PYPI,
            "pypi-newtoken123",
            old_token="pypi-oldtoken456",
        )
        assert result.old_token_prefix == "pypi-old..."
        assert result.new_token_prefix == "pypi-new..."


class TestGlobalManager:
    """Tests for module-level manager functions."""

    def setup_method(self):
        reset_token_rotation_manager()

    def teardown_method(self):
        reset_token_rotation_manager()

    def test_get_returns_none_initially(self):
        assert get_token_rotation_manager() is None

    def test_get_or_create(self):
        manager = get_or_create_token_rotation_manager()
        assert manager is not None
        assert isinstance(manager, TokenRotationManager)

        # Same instance on second call
        manager2 = get_or_create_token_rotation_manager()
        assert manager2 is manager

    def test_reset(self):
        get_or_create_token_rotation_manager()
        assert get_token_rotation_manager() is not None

        reset_token_rotation_manager()
        assert get_token_rotation_manager() is None


class TestCLIIntegration:
    """Tests for CLI command functions."""

    @patch("aragora.security.token_rotation._BOTO3_AVAILABLE", True)
    @patch("aragora.security.token_rotation.boto3")
    def test_dry_run_skips_stores(self, mock_boto3):
        """Dry-run should not write to any stores."""
        config = TokenRotationConfig(stores=[])
        manager = TokenRotationManager(config=config)

        result = manager.rotate(TokenType.PYPI, "pypi-test")
        assert result.success is True
        assert result.stores_updated == []
        # boto3 should not have been called for storage
        mock_boto3.client.assert_not_called()
