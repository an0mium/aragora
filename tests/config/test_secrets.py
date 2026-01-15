"""
Tests for aragora.config.secrets module.

Tests cover:
- Secret retrieval from environment variables
- Secret retrieval from AWS Secrets Manager (mocked)
- Fallback behavior when secrets are not found
- Required secret validation
- Secret manager initialization
- AWS client lazy initialization
- Helper methods for auth and billing secrets
- Cache management
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None  # boto3 not installed

from aragora.config.secrets import (
    MANAGED_SECRETS,
    SecretManager,
    SecretsConfig,
    clear_secret_cache,
    get_required_secret,
    get_secret,
    get_secret_manager,
    reset_secret_manager,
)


class TestSecretsConfig:
    """Tests for SecretsConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = SecretsConfig()
        assert config.aws_region == "us-east-1"
        assert config.secret_name == "aragora/production"
        assert config.use_aws is False
        assert config.cache_ttl_seconds == 300

    def test_from_env_defaults(self):
        """Config loads defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = SecretsConfig.from_env()
            assert config.aws_region == "us-east-1"
            assert config.secret_name == "aragora/production"
            assert config.use_aws is False

    def test_from_env_with_values(self):
        """Config loads values from environment."""
        env = {
            "AWS_REGION": "eu-west-1",
            "ARAGORA_SECRET_NAME": "aragora/staging",
            "ARAGORA_USE_SECRETS_MANAGER": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SecretsConfig.from_env()
            assert config.aws_region == "eu-west-1"
            assert config.secret_name == "aragora/staging"
            assert config.use_aws is True

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "Yes"])
    def test_use_aws_truthy_values(self, value):
        """Config recognizes various truthy values for use_aws."""
        with patch.dict(os.environ, {"ARAGORA_USE_SECRETS_MANAGER": value}, clear=True):
            config = SecretsConfig.from_env()
            assert config.use_aws is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "invalid"])
    def test_use_aws_falsy_values(self, value):
        """Config treats non-truthy values as False for use_aws."""
        with patch.dict(os.environ, {"ARAGORA_USE_SECRETS_MANAGER": value}, clear=True):
            config = SecretsConfig.from_env()
            assert config.use_aws is False


class TestSecretManager:
    """Tests for SecretManager class."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global manager before and after each test."""
        reset_secret_manager()
        clear_secret_cache()
        yield
        reset_secret_manager()
        clear_secret_cache()

    def test_get_from_environment(self):
        """Secrets are retrieved from environment variables."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {"TEST_SECRET": "env_value"}):
            result = manager.get("TEST_SECRET")
            assert result == "env_value"

    def test_get_returns_default_when_not_found(self):
        """Default value is returned when secret not found."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {}, clear=True):
            result = manager.get("NONEXISTENT_SECRET", "default_value")
            assert result == "default_value"

    def test_get_returns_none_when_not_found_no_default(self):
        """None is returned when secret not found and no default."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {}, clear=True):
            result = manager.get("NONEXISTENT_SECRET")
            assert result is None

    def test_get_required_raises_when_missing(self):
        """Required secrets raise ValueError when not found."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Required secret 'MISSING_SECRET' not found"):
                manager.get_required("MISSING_SECRET")

    def test_get_required_returns_value_when_found(self):
        """Required secrets return value when found."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {"FOUND_SECRET": "found_value"}):
            result = manager.get_required("FOUND_SECRET")
            assert result == "found_value"

    def test_get_secrets_batch(self):
        """Multiple secrets can be retrieved at once."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {"SECRET_A": "value_a", "SECRET_B": "value_b"}):
            result = manager.get_secrets(["SECRET_A", "SECRET_B", "SECRET_C"])
            assert result == {
                "SECRET_A": "value_a",
                "SECRET_B": "value_b",
                "SECRET_C": None,
            }

    def test_is_configured_true(self):
        """is_configured returns True when secret exists."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {"CONFIGURED_SECRET": "value"}):
            assert manager.is_configured("CONFIGURED_SECRET") is True

    def test_is_configured_false(self):
        """is_configured returns False when secret missing."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        with patch.dict(os.environ, {}, clear=True):
            assert manager.is_configured("UNCONFIGURED_SECRET") is False


class TestSecretManagerAWS:
    """Tests for AWS Secrets Manager integration."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global manager before and after each test."""
        reset_secret_manager()
        clear_secret_cache()
        yield
        reset_secret_manager()
        clear_secret_cache()

    def test_aws_secrets_cached_on_init(self):
        """AWS secrets are loaded and cached during initialization."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"AWS_SECRET": "aws_value"})
        }

        with patch.object(manager, "_get_aws_client", return_value=mock_client):
            manager._initialize()
            result = manager.get("AWS_SECRET")
            assert result == "aws_value"

    def test_aws_cache_takes_precedence_over_env(self):
        """AWS cached secrets take precedence over environment variables."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)
        manager._cached_secrets = {"DUAL_SECRET": "aws_value"}
        manager._initialized = True

        with patch.dict(os.environ, {"DUAL_SECRET": "env_value"}):
            result = manager.get("DUAL_SECRET")
            assert result == "aws_value"

    def test_fallback_to_env_when_not_in_aws(self):
        """Falls back to environment when secret not in AWS cache."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)
        manager._cached_secrets = {}  # AWS has no secrets
        manager._initialized = True

        with patch.dict(os.environ, {"ENV_ONLY_SECRET": "env_value"}):
            result = manager.get("ENV_ONLY_SECRET")
            assert result == "env_value"

    def test_aws_client_lazy_initialization(self):
        """AWS client is lazily initialized only when needed."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        assert manager._aws_client is None

        with patch("boto3.client") as mock_boto:
            mock_boto.return_value = MagicMock()
            client = manager._get_aws_client()
            assert client is not None
            mock_boto.assert_called_once_with("secretsmanager", region_name="us-east-1")

    def test_aws_client_handles_missing_boto3(self):
        """Gracefully handles missing boto3 library."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        with patch.dict("sys.modules", {"boto3": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'boto3'")):
                client = manager._get_aws_client()
                assert client is None

    def test_aws_handles_resource_not_found(self):
        """Gracefully handles missing secret in AWS."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        mock_client = MagicMock()
        # Simulate ClientError for ResourceNotFoundException
        mock_client.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Secret not found"}},
            "GetSecretValue",
        )

        with patch.object(manager, "_get_aws_client", return_value=mock_client):
            secrets = manager._load_from_aws()
            assert secrets == {}

    def test_aws_handles_access_denied(self):
        """Gracefully handles access denied from AWS."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "GetSecretValue",
        )

        with patch.object(manager, "_get_aws_client", return_value=mock_client):
            secrets = manager._load_from_aws()
            assert secrets == {}

    def test_aws_handles_invalid_json(self):
        """Gracefully handles invalid JSON from AWS."""
        config = SecretsConfig(use_aws=True)
        manager = SecretManager(config)

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "not valid json"}

        with patch.object(manager, "_get_aws_client", return_value=mock_client):
            secrets = manager._load_from_aws()
            assert secrets == {}


class TestSecretManagerHelpers:
    """Tests for helper methods."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global manager before and after each test."""
        reset_secret_manager()
        clear_secret_cache()
        yield
        reset_secret_manager()
        clear_secret_cache()

    def test_get_auth_secrets(self):
        """Auth secrets helper returns expected keys."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        env = {
            "JWT_SECRET_KEY": "jwt_secret",
            "JWT_REFRESH_SECRET": "refresh_secret",
            "GOOGLE_OAUTH_CLIENT_ID": "google_id",
            "GOOGLE_OAUTH_CLIENT_SECRET": "google_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            result = manager.get_auth_secrets()
            assert result["JWT_SECRET_KEY"] == "jwt_secret"
            assert result["JWT_REFRESH_SECRET"] == "refresh_secret"
            assert result["GOOGLE_OAUTH_CLIENT_ID"] == "google_id"
            assert result["GOOGLE_OAUTH_CLIENT_SECRET"] == "google_secret"
            assert "GITHUB_OAUTH_CLIENT_ID" in result
            assert "GITHUB_OAUTH_CLIENT_SECRET" in result

    def test_get_billing_secrets(self):
        """Billing secrets helper returns expected keys."""
        config = SecretsConfig(use_aws=False)
        manager = SecretManager(config)

        env = {
            "STRIPE_SECRET_KEY": "sk_test_123",
            "STRIPE_WEBHOOK_SECRET": "whsec_123",
        }
        with patch.dict(os.environ, env, clear=True):
            result = manager.get_billing_secrets()
            assert result["STRIPE_SECRET_KEY"] == "sk_test_123"
            assert result["STRIPE_WEBHOOK_SECRET"] == "whsec_123"
            assert "STRIPE_PRICE_STARTER" in result
            assert "STRIPE_PRICE_PROFESSIONAL" in result
            assert "STRIPE_PRICE_ENTERPRISE" in result


class TestGlobalFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global manager before and after each test."""
        reset_secret_manager()
        clear_secret_cache()
        yield
        reset_secret_manager()
        clear_secret_cache()

    def test_get_secret_manager_singleton(self):
        """get_secret_manager returns singleton instance."""
        manager1 = get_secret_manager()
        manager2 = get_secret_manager()
        assert manager1 is manager2

    def test_reset_secret_manager(self):
        """reset_secret_manager clears the singleton."""
        manager1 = get_secret_manager()
        reset_secret_manager()
        manager2 = get_secret_manager()
        assert manager1 is not manager2

    def test_get_secret_function(self):
        """get_secret function works correctly."""
        with patch.dict(os.environ, {"FUNC_SECRET": "func_value"}):
            result = get_secret("FUNC_SECRET")
            assert result == "func_value"

    def test_get_secret_with_default(self):
        """get_secret function returns default when not found."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_secret("MISSING_FUNC_SECRET", "default")
            assert result == "default"

    def test_get_required_secret_function(self):
        """get_required_secret function works correctly."""
        with patch.dict(os.environ, {"REQUIRED_SECRET": "required_value"}):
            result = get_required_secret("REQUIRED_SECRET")
            assert result == "required_value"

    def test_get_required_secret_raises(self):
        """get_required_secret raises when secret missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_required_secret("MISSING_REQUIRED")

    def test_clear_secret_cache(self):
        """clear_secret_cache clears the lru_cache."""
        # Call get_secret to populate cache
        with patch.dict(os.environ, {"CACHED_SECRET": "cached_value"}):
            result1 = get_secret("CACHED_SECRET")
            assert result1 == "cached_value"

        # Clear cache
        clear_secret_cache()

        # Cache should be empty, function should work with new env
        with patch.dict(os.environ, {"CACHED_SECRET": "new_value"}):
            # Note: Due to singleton manager, value may still come from env
            result2 = get_secret("CACHED_SECRET")
            assert result2 == "new_value"


class TestManagedSecrets:
    """Tests for MANAGED_SECRETS constant."""

    def test_managed_secrets_is_frozenset(self):
        """MANAGED_SECRETS is immutable."""
        assert isinstance(MANAGED_SECRETS, frozenset)

    def test_managed_secrets_contains_expected_keys(self):
        """MANAGED_SECRETS contains all expected secret names."""
        expected = [
            "JWT_SECRET_KEY",
            "JWT_REFRESH_SECRET",
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GITHUB_OAUTH_CLIENT_ID",
            "GITHUB_OAUTH_CLIENT_SECRET",
            "STRIPE_SECRET_KEY",
            "STRIPE_WEBHOOK_SECRET",
            "DATABASE_URL",
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "REDIS_URL",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "SENTRY_DSN",
        ]
        for key in expected:
            assert key in MANAGED_SECRETS, f"Missing managed secret: {key}"

    def test_managed_secrets_count(self):
        """MANAGED_SECRETS contains expected number of secrets."""
        # At least 20 secrets should be managed
        assert len(MANAGED_SECRETS) >= 20
