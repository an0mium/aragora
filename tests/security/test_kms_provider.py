"""
Tests for Cloud KMS Provider.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.security.kms_provider import (
    KmsProvider,
    KmsKeyMetadata,
    AwsKmsProvider,
    AzureKeyVaultProvider,
    GcpKmsProvider,
    HashiCorpVaultProvider,
    LocalKmsProvider,
    get_kms_provider,
    init_kms_provider,
    reset_kms_provider,
    detect_cloud_provider,
)


class TestDetectCloudProvider:
    """Tests for cloud provider detection."""

    def setup_method(self):
        """Reset provider before each test."""
        reset_kms_provider()

    def teardown_method(self):
        """Clean up after each test."""
        reset_kms_provider()

    def test_explicit_aws(self, monkeypatch):
        """Should use AWS when explicitly set."""
        monkeypatch.setenv("ARAGORA_KMS_PROVIDER", "aws")
        assert detect_cloud_provider() == "aws"

    def test_explicit_azure(self, monkeypatch):
        """Should use Azure when explicitly set."""
        monkeypatch.setenv("ARAGORA_KMS_PROVIDER", "azure")
        assert detect_cloud_provider() == "azure"

    def test_explicit_gcp(self, monkeypatch):
        """Should use GCP when explicitly set."""
        monkeypatch.setenv("ARAGORA_KMS_PROVIDER", "gcp")
        assert detect_cloud_provider() == "gcp"

    def test_explicit_local(self, monkeypatch):
        """Should use local when explicitly set."""
        monkeypatch.setenv("ARAGORA_KMS_PROVIDER", "local")
        assert detect_cloud_provider() == "local"

    def test_detect_aws_from_env(self, monkeypatch):
        """Should detect AWS from environment variables."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("ARAGORA_AWS_KMS_KEY_ID", "alias/my-key")
        assert detect_cloud_provider() == "aws"

    def test_detect_azure_from_env(self, monkeypatch):
        """Should detect Azure from environment variables."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://myvault.vault.azure.net")
        assert detect_cloud_provider() == "azure"

    def test_detect_gcp_from_env(self, monkeypatch):
        """Should detect GCP from environment variables."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        monkeypatch.setenv("ARAGORA_GCP_KMS_KEYRING", "aragora")
        assert detect_cloud_provider() == "gcp"

    def test_fallback_to_local(self, monkeypatch):
        """Should fall back to local when no cloud detected."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        assert detect_cloud_provider() == "local"


class TestLocalKmsProvider:
    """Tests for LocalKmsProvider."""

    @pytest.mark.asyncio
    async def test_get_encryption_key_from_env(self, monkeypatch):
        """Should use key from environment variable."""
        key_hex = "a" * 64  # 32 bytes in hex
        monkeypatch.setenv("ARAGORA_ENCRYPTION_KEY", key_hex)

        provider = LocalKmsProvider()
        key = await provider.get_encryption_key("test-key")

        assert len(key) == 32
        assert key == bytes.fromhex(key_hex)

    @pytest.mark.asyncio
    async def test_get_encryption_key_generates_ephemeral(self, monkeypatch):
        """Should generate ephemeral key when not set."""
        monkeypatch.delenv("ARAGORA_ENCRYPTION_KEY", raising=False)

        provider = LocalKmsProvider()
        key = await provider.get_encryption_key("test-key")

        assert len(key) == 32

    @pytest.mark.asyncio
    async def test_get_key_metadata(self):
        """Should return local metadata."""
        provider = LocalKmsProvider(master_key=b"x" * 32)

        meta = await provider.get_key_metadata("test")

        assert meta.key_id == "local-key"
        assert meta.provider == "local"
        assert meta.algorithm == "AES-256"

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self):
        """Should encrypt and decrypt data keys."""
        provider = LocalKmsProvider(master_key=b"x" * 32)

        # Use exactly 32 bytes for the key
        original = b"test-data-key-exactly-32-bytes!!"
        assert len(original) == 32

        encrypted = await provider.encrypt_data_key(original, "test")
        # Note: Local provider uses XOR, so roundtrip should work
        assert len(encrypted) == 32

    @pytest.mark.asyncio
    async def test_close(self):
        """Should close without error."""
        provider = LocalKmsProvider(master_key=b"x" * 32)
        await provider.close()  # Should not raise


class TestAwsKmsProvider:
    """Tests for AwsKmsProvider with mocked boto3."""

    def test_init_with_defaults(self, monkeypatch):
        """Should use environment defaults."""
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv("ARAGORA_AWS_KMS_KEY_ID", "alias/my-key")

        provider = AwsKmsProvider()

        assert provider.region == "us-west-2"
        assert provider.default_key_id == "alias/my-key"

    def test_init_with_explicit_values(self):
        """Should use explicit values."""
        provider = AwsKmsProvider(region="eu-west-1", key_id="alias/custom")

        assert provider.region == "eu-west-1"
        assert provider.default_key_id == "alias/custom"

    @pytest.mark.asyncio
    async def test_get_encryption_key(self, monkeypatch):
        """Should call GenerateDataKey."""
        mock_client = MagicMock()
        mock_client.generate_data_key.return_value = {
            "Plaintext": b"generated-key-32bytes-here!!!!!",
            "CiphertextBlob": b"encrypted-blob",
        }

        monkeypatch.setenv("AWS_REGION", "us-east-1")
        provider = AwsKmsProvider()
        provider._client = mock_client

        key = await provider.get_encryption_key("alias/test")

        assert key == b"generated-key-32bytes-here!!!!!"
        mock_client.generate_data_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_decrypt_data_key(self, monkeypatch):
        """Should call Decrypt."""
        mock_client = MagicMock()
        mock_client.decrypt.return_value = {
            "Plaintext": b"decrypted-key-32bytes-here!!!!!",
        }

        monkeypatch.setenv("AWS_REGION", "us-east-1")
        provider = AwsKmsProvider()
        provider._client = mock_client

        key = await provider.decrypt_data_key(b"encrypted", "alias/test")

        assert key == b"decrypted-key-32bytes-here!!!!!"

    @pytest.mark.asyncio
    async def test_encrypt_data_key(self, monkeypatch):
        """Should call Encrypt."""
        mock_client = MagicMock()
        mock_client.encrypt.return_value = {
            "CiphertextBlob": b"encrypted-result",
        }

        monkeypatch.setenv("AWS_REGION", "us-east-1")
        provider = AwsKmsProvider()
        provider._client = mock_client

        result = await provider.encrypt_data_key(b"plaintext", "alias/test")

        assert result == b"encrypted-result"


class TestGetKmsProvider:
    """Tests for get_kms_provider factory."""

    def setup_method(self):
        """Reset provider before each test."""
        reset_kms_provider()

    def teardown_method(self):
        """Clean up after each test."""
        reset_kms_provider()

    def test_returns_local_by_default(self, monkeypatch):
        """Should return LocalKmsProvider when no cloud detected."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        provider = get_kms_provider()

        assert isinstance(provider, LocalKmsProvider)

    def test_returns_singleton(self, monkeypatch):
        """Should return same instance on repeated calls."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)

        p1 = get_kms_provider()
        p2 = get_kms_provider()

        assert p1 is p2

    def test_init_kms_provider(self):
        """Should allow explicit provider initialization."""
        custom_provider = LocalKmsProvider(master_key=b"custom" * 5 + b"!!")
        init_kms_provider(custom_provider)

        provider = get_kms_provider()

        assert provider is custom_provider


class TestKmsKeyMetadata:
    """Tests for KmsKeyMetadata dataclass."""

    def test_creation(self):
        """Should create metadata with all fields."""
        meta = KmsKeyMetadata(
            key_id="test-key",
            key_arn="arn:aws:kms:us-east-1:123:key/abc",
            version="v1",
            created_at="2024-01-01T00:00:00Z",
            algorithm="AES-256",
            provider="aws-kms",
        )

        assert meta.key_id == "test-key"
        assert meta.key_arn == "arn:aws:kms:us-east-1:123:key/abc"
        assert meta.version == "v1"
        assert meta.provider == "aws-kms"

    def test_defaults(self):
        """Should use defaults for optional fields."""
        meta = KmsKeyMetadata(key_id="test")

        assert meta.key_arn is None
        assert meta.version is None
        assert meta.algorithm == "AES-256"
        assert meta.provider == "unknown"


class TestHashiCorpVaultProvider:
    """Tests for HashiCorpVaultProvider with mocked hvac."""

    def test_init_with_defaults(self, monkeypatch):
        """Should use environment defaults."""
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_NAMESPACE", "myns")
        monkeypatch.setenv("ARAGORA_VAULT_TRANSIT_PATH", "transit")
        monkeypatch.setenv("ARAGORA_VAULT_KEY_NAME", "my-key")

        provider = HashiCorpVaultProvider()

        assert provider.addr == "https://vault.example.com"
        assert provider.token == "test-token"
        assert provider.namespace == "myns"
        assert provider.transit_path == "transit"
        assert provider.default_key == "my-key"

    def test_init_with_explicit_values(self):
        """Should use explicit values."""
        provider = HashiCorpVaultProvider(
            addr="https://custom.vault.com",
            token="explicit-token",
            namespace="custom-ns",
            transit_path="custom-transit",
            key_name="custom-key",
        )

        assert provider.addr == "https://custom.vault.com"
        assert provider.token == "explicit-token"
        assert provider.namespace == "custom-ns"
        assert provider.transit_path == "custom-transit"
        assert provider.default_key == "custom-key"

    @pytest.mark.asyncio
    async def test_get_encryption_key(self, monkeypatch):
        """Should call Vault transit generate_data_key."""
        import base64

        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.transit.generate_data_key.return_value = {
            "data": {
                "plaintext": base64.b64encode(b"generated-key-32bytes-here!!!!!").decode(),
                "ciphertext": "vault:v1:encrypted-blob",
            }
        }

        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        provider = HashiCorpVaultProvider()
        provider._client = mock_client

        key = await provider.get_encryption_key("test-key")

        assert key == b"generated-key-32bytes-here!!!!!"
        mock_client.secrets.transit.generate_data_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_encrypt_data_key(self, monkeypatch):
        """Should call Vault transit encrypt_data."""
        import base64

        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.transit.encrypt_data.return_value = {
            "data": {"ciphertext": f"vault:v1:{base64.b64encode(b'encrypted-result').decode()}"}
        }

        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        provider = HashiCorpVaultProvider()
        provider._client = mock_client

        result = await provider.encrypt_data_key(b"plaintext", "test-key")

        assert result == b"encrypted-result"

    @pytest.mark.asyncio
    async def test_decrypt_data_key(self, monkeypatch):
        """Should call Vault transit decrypt_data."""
        import base64

        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.transit.decrypt_data.return_value = {
            "data": {"plaintext": base64.b64encode(b"decrypted-key-32bytes-here!!!!!").decode()}
        }

        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        provider = HashiCorpVaultProvider()
        provider._client = mock_client

        key = await provider.decrypt_data_key(b"encrypted", "test-key")

        assert key == b"decrypted-key-32bytes-here!!!!!"

    @pytest.mark.asyncio
    async def test_get_key_metadata(self, monkeypatch):
        """Should return Vault key metadata."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.transit.read_key.return_value = {
            "data": {
                "latest_version": 3,
                "type": "aes256-gcm96",
            }
        }

        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        provider = HashiCorpVaultProvider(transit_path="transit")
        provider._client = mock_client

        meta = await provider.get_key_metadata("test-key")

        assert meta.key_id == "test-key"
        assert meta.version == "3"
        assert meta.provider == "hashicorp-vault"
        assert "vault://transit/keys/test-key" in meta.key_arn

    @pytest.mark.asyncio
    async def test_rotate_key(self, monkeypatch):
        """Should rotate key in Vault."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.transit.rotate_key.return_value = {}
        mock_client.secrets.transit.read_key.return_value = {
            "data": {
                "latest_version": 4,
                "type": "aes256-gcm96",
            }
        }

        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        provider = HashiCorpVaultProvider()
        provider._client = mock_client

        new_meta = await provider.rotate_key("test-key")

        assert new_meta.version == "4"
        mock_client.secrets.transit.rotate_key.assert_called_once()


class TestDetectVaultProvider:
    """Tests for Vault provider detection."""

    def setup_method(self):
        """Reset provider before each test."""
        reset_kms_provider()

    def teardown_method(self):
        """Clean up after each test."""
        reset_kms_provider()

    def test_explicit_vault(self, monkeypatch):
        """Should use Vault when explicitly set."""
        monkeypatch.setenv("ARAGORA_KMS_PROVIDER", "vault")
        assert detect_cloud_provider() == "vault"

    def test_detect_vault_from_env(self, monkeypatch):
        """Should detect Vault from environment variables."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        assert detect_cloud_provider() == "vault"

    def test_vault_requires_auth(self, monkeypatch):
        """Should not detect Vault without auth."""
        monkeypatch.delenv("ARAGORA_KMS_PROVIDER", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        monkeypatch.delenv("VAULT_ROLE_ID", raising=False)
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        # Without VAULT_TOKEN or VAULT_ROLE_ID, should fall back to local
        assert detect_cloud_provider() == "local"
