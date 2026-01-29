"""
Cloud KMS Provider for Encryption Key Management.

Provides integration with cloud key management services for secure
encryption key retrieval and rotation:
- AWS KMS (using boto3)
- Azure Key Vault (using azure-identity)
- GCP Cloud KMS (using google-cloud-kms)

The provider auto-detects which cloud platform to use based on
environment variables.

Usage:
    from aragora.security.kms_provider import get_kms_provider

    # Auto-detect and get the appropriate provider
    provider = get_kms_provider()

    # Retrieve encryption key
    key = await provider.get_encryption_key("aragora-master-key")

    # Decrypt a data key (envelope encryption)
    plaintext = await provider.decrypt_data_key(encrypted_key)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KmsKeyMetadata:
    """Metadata about a KMS key."""

    key_id: str
    key_arn: str | None = None
    version: str | None = None
    created_at: str | None = None
    algorithm: str = "AES-256"
    provider: str = "unknown"


class KmsProvider(ABC):
    """Abstract base class for cloud KMS providers."""

    @abstractmethod
    async def get_encryption_key(self, key_id: str) -> bytes:
        """
        Retrieve an encryption key from the KMS.

        Args:
            key_id: The key identifier (name, ARN, or resource ID)

        Returns:
            The raw key bytes (32 bytes for AES-256)
        """
        pass

    @abstractmethod
    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """
        Decrypt a data key using the KMS master key.

        Used in envelope encryption where data keys are encrypted
        by the KMS master key.

        Args:
            encrypted_key: The encrypted data key
            key_id: The master key ID used for encryption

        Returns:
            The decrypted data key bytes
        """
        pass

    @abstractmethod
    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """
        Encrypt a data key using the KMS master key.

        Args:
            plaintext_key: The plaintext data key
            key_id: The master key ID to use for encryption

        Returns:
            The encrypted data key bytes
        """
        pass

    @abstractmethod
    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Get metadata about a KMS key."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass


class AwsKmsProvider(KmsProvider):
    """
    AWS KMS provider using boto3.

    Environment variables:
    - AWS_REGION: AWS region (required)
    - AWS_ACCESS_KEY_ID: Access key (or use IAM role)
    - AWS_SECRET_ACCESS_KEY: Secret key (or use IAM role)
    - ARAGORA_AWS_KMS_KEY_ID: Default key ID/ARN
    """

    def __init__(self, region: str | None = None, key_id: str | None = None):
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.default_key_id = key_id or os.environ.get("ARAGORA_AWS_KMS_KEY_ID")
        self._client = None

    def _get_client(self):
        """Lazy initialize the KMS client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("kms", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 is required for AWS KMS. Install with: pip install boto3")
        return self._client

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Generate a data key using AWS KMS."""
        client = self._get_client()

        # Use GenerateDataKey to get a unique data key
        response = client.generate_data_key(
            KeyId=key_id or self.default_key_id,
            KeySpec="AES_256",
        )

        # Return the plaintext key (also stores encrypted version for envelope)
        plaintext: bytes = response["Plaintext"]
        return plaintext

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using AWS KMS."""
        client = self._get_client()

        response = client.decrypt(
            CiphertextBlob=encrypted_key,
            KeyId=key_id or self.default_key_id,
        )

        plaintext: bytes = response["Plaintext"]
        return plaintext

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using AWS KMS."""
        client = self._get_client()

        response = client.encrypt(
            KeyId=key_id or self.default_key_id,
            Plaintext=plaintext_key,
        )

        ciphertext: bytes = response["CiphertextBlob"]
        return ciphertext

    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Get metadata about an AWS KMS key."""
        client = self._get_client()

        response = client.describe_key(KeyId=key_id or self.default_key_id)
        meta = response["KeyMetadata"]

        return KmsKeyMetadata(
            key_id=meta["KeyId"],
            key_arn=meta["Arn"],
            version=None,  # AWS doesn't expose version like this
            created_at=meta["CreationDate"].isoformat() if meta.get("CreationDate") else None,
            algorithm="AES-256",
            provider="aws-kms",
        )

    async def close(self) -> None:
        """Close the client."""
        self._client = None


class AzureKeyVaultProvider(KmsProvider):
    """
    Azure Key Vault provider.

    Environment variables:
    - AZURE_KEY_VAULT_URL: Key vault URL (required)
    - AZURE_TENANT_ID: Azure tenant ID
    - AZURE_CLIENT_ID: Service principal client ID
    - AZURE_CLIENT_SECRET: Service principal secret
    - ARAGORA_AZURE_KEY_NAME: Default key name
    """

    def __init__(
        self,
        vault_url: str | None = None,
        key_name: str | None = None,
    ):
        self.vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
        self.default_key_name = key_name or os.environ.get(
            "ARAGORA_AZURE_KEY_NAME", "aragora-master-key"
        )
        self._client = None
        self._crypto_client = None

    def _get_clients(self):
        """Lazy initialize Azure clients."""
        if self._client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.keys import KeyClient

                credential = DefaultAzureCredential()
                self._client = KeyClient(
                    vault_url=self.vault_url,
                    credential=credential,
                )
            except ImportError:
                raise ImportError(
                    "azure-identity and azure-keyvault-keys are required for Azure KMS. "
                    "Install with: pip install azure-identity azure-keyvault-keys"
                )
        return self._client

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Generate a random key and encrypt with Azure Key Vault."""
        import secrets

        # Generate a random 256-bit key
        key = secrets.token_bytes(32)

        # In production, you'd want to store the encrypted version
        # For now, return the raw key (envelope encryption would store encrypted)
        return key

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm

            credential = DefaultAzureCredential()
            key_client = self._get_clients()
            key = key_client.get_key(key_id or self.default_key_name)

            crypto_client = CryptographyClient(key, credential=credential)
            result = crypto_client.decrypt(
                EncryptionAlgorithm.rsa_oaep_256,
                encrypted_key,
            )

            plaintext: bytes = result.plaintext
            return plaintext
        except Exception as e:
            logger.error(f"Azure Key Vault decrypt failed: {e}")
            raise

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm

            credential = DefaultAzureCredential()
            key_client = self._get_clients()
            key = key_client.get_key(key_id or self.default_key_name)

            crypto_client = CryptographyClient(key, credential=credential)
            result = crypto_client.encrypt(
                EncryptionAlgorithm.rsa_oaep_256,
                plaintext_key,
            )

            ciphertext: bytes = result.ciphertext
            return ciphertext
        except Exception as e:
            logger.error(f"Azure Key Vault encrypt failed: {e}")
            raise

    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Get metadata about an Azure Key Vault key."""
        key_client = self._get_clients()
        key = key_client.get_key(key_id or self.default_key_name)

        return KmsKeyMetadata(
            key_id=key.name,
            key_arn=key.id,
            version=key.properties.version,
            created_at=key.properties.created_on.isoformat() if key.properties.created_on else None,
            algorithm=str(key.key_type),
            provider="azure-keyvault",
        )

    async def close(self) -> None:
        """Close clients."""
        self._client = None
        self._crypto_client = None


class GcpKmsProvider(KmsProvider):
    """
    GCP Cloud KMS provider.

    Environment variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - ARAGORA_GCP_KMS_LOCATION: KMS location (default: global)
    - ARAGORA_GCP_KMS_KEYRING: Key ring name
    - ARAGORA_GCP_KMS_KEY: Key name
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        keyring: str | None = None,
        key_name: str | None = None,
    ):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("ARAGORA_GCP_KMS_LOCATION", "global")
        self.keyring = keyring or os.environ.get("ARAGORA_GCP_KMS_KEYRING", "aragora")
        self.default_key = key_name or os.environ.get("ARAGORA_GCP_KMS_KEY", "master-key")
        self._client = None

    def _get_client(self):
        """Lazy initialize GCP KMS client."""
        if self._client is None:
            try:
                from google.cloud import kms

                self._client = kms.KeyManagementServiceClient()
            except ImportError:
                raise ImportError(
                    "google-cloud-kms is required for GCP KMS. "
                    "Install with: pip install google-cloud-kms"
                )
        return self._client

    def _get_key_name(self, key_id: str | None = None) -> str:
        """Build full key resource name."""
        key = key_id or self.default_key
        return (
            f"projects/{self.project}/"
            f"locations/{self.location}/"
            f"keyRings/{self.keyring}/"
            f"cryptoKeys/{key}"
        )

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Generate a data key (GCP doesn't have GenerateDataKey like AWS)."""
        import secrets

        # Generate a random 256-bit key locally
        # In envelope encryption, this would be encrypted with GCP KMS
        return secrets.token_bytes(32)

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using GCP KMS."""
        client = self._get_client()

        response = client.decrypt(
            request={
                "name": self._get_key_name(key_id),
                "ciphertext": encrypted_key,
            }
        )

        plaintext: bytes = response.plaintext
        return plaintext

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using GCP KMS."""
        client = self._get_client()

        response = client.encrypt(
            request={
                "name": self._get_key_name(key_id),
                "plaintext": plaintext_key,
            }
        )

        ciphertext: bytes = response.ciphertext
        return ciphertext

    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Get metadata about a GCP KMS key."""
        client = self._get_client()

        response = client.get_crypto_key(request={"name": self._get_key_name(key_id)})

        return KmsKeyMetadata(
            key_id=response.name.split("/")[-1],
            key_arn=response.name,
            version=response.primary.name.split("/")[-1] if response.primary else None,
            created_at=response.create_time.isoformat() if response.create_time else None,
            algorithm=str(response.purpose),
            provider="gcp-kms",
        )

    async def close(self) -> None:
        """Close the client."""
        self._client = None


class HashiCorpVaultProvider(KmsProvider):
    """
    HashiCorp Vault provider for enterprise key management.

    Supports both Vault's Transit secrets engine (for encryption/decryption)
    and KV secrets engine (for key storage).

    Environment variables:
    - VAULT_ADDR: Vault server address (required)
    - VAULT_TOKEN: Authentication token (or use other auth methods)
    - VAULT_NAMESPACE: Namespace for enterprise Vault (optional)
    - ARAGORA_VAULT_TRANSIT_PATH: Transit engine mount path (default: transit)
    - ARAGORA_VAULT_KEY_NAME: Default key name (default: aragora-master-key)
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        namespace: str | None = None,
        transit_path: str | None = None,
        key_name: str | None = None,
    ):
        self.addr = addr or os.environ.get("VAULT_ADDR")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.namespace = namespace or os.environ.get("VAULT_NAMESPACE")
        self.transit_path = transit_path or os.environ.get("ARAGORA_VAULT_TRANSIT_PATH", "transit")
        self.default_key = key_name or os.environ.get(
            "ARAGORA_VAULT_KEY_NAME", "aragora-master-key"
        )
        self._client = None

    def _get_client(self):
        """Lazy initialize the Vault client."""
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(
                    url=self.addr,
                    token=self.token,
                    namespace=self.namespace,
                )

                if not self._client.is_authenticated():
                    raise ValueError("Vault authentication failed")

            except ImportError:
                raise ImportError(
                    "hvac is required for HashiCorp Vault. Install with: pip install hvac"
                )
        return self._client

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Generate a data key using Vault Transit engine.

        Uses Vault's datakey endpoint to generate a new data encryption key
        that is encrypted with the named key.
        """
        import base64

        client = self._get_client()
        key_name = key_id or self.default_key

        # Generate a new data key (returns plaintext + ciphertext)
        response = client.secrets.transit.generate_data_key(
            name=key_name,
            key_type="plaintext",  # Returns both plaintext and ciphertext
            mount_point=self.transit_path,
        )

        # Return the plaintext key (base64 encoded by Vault)
        plaintext_b64: str = response["data"]["plaintext"]
        return base64.b64decode(plaintext_b64)

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using Vault Transit engine."""
        import base64

        client = self._get_client()
        key_name = key_id or self.default_key

        # Vault expects base64 ciphertext prefixed with vault:v1:
        ciphertext_b64 = base64.b64encode(encrypted_key).decode("ascii")

        response = client.secrets.transit.decrypt_data(
            name=key_name,
            ciphertext=f"vault:v1:{ciphertext_b64}",
            mount_point=self.transit_path,
        )

        plaintext_b64: str = response["data"]["plaintext"]
        return base64.b64decode(plaintext_b64)

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using Vault Transit engine."""
        import base64

        client = self._get_client()
        key_name = key_id or self.default_key

        # Vault expects base64-encoded plaintext
        plaintext_b64 = base64.b64encode(plaintext_key).decode("ascii")

        response = client.secrets.transit.encrypt_data(
            name=key_name,
            plaintext=plaintext_b64,
            mount_point=self.transit_path,
        )

        # Response is vault:v1:base64ciphertext
        ciphertext: str = response["data"]["ciphertext"]
        # Strip the vault:v1: prefix and decode
        _, _, b64_part = ciphertext.partition("vault:v1:")
        return base64.b64decode(b64_part)

    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Get metadata about a Vault Transit key."""
        client = self._get_client()
        key_name = key_id or self.default_key

        response = client.secrets.transit.read_key(
            name=key_name,
            mount_point=self.transit_path,
        )

        data = response["data"]
        return KmsKeyMetadata(
            key_id=key_name,
            key_arn=f"vault://{self.transit_path}/keys/{key_name}",
            version=str(data.get("latest_version", 1)),
            created_at=None,  # Vault doesn't expose creation time easily
            algorithm=data.get("type", "aes256-gcm96"),
            provider="hashicorp-vault",
        )

    async def rotate_key(self, key_id: str) -> KmsKeyMetadata:
        """Rotate a key in Vault Transit engine.

        Creates a new version of the key. Old versions remain available
        for decryption but new encryptions use the new version.
        """
        client = self._get_client()
        key_name = key_id or self.default_key

        client.secrets.transit.rotate_key(
            name=key_name,
            mount_point=self.transit_path,
        )

        logger.info(f"Rotated Vault Transit key: {key_name}")
        return await self.get_key_metadata(key_name)

    async def close(self) -> None:
        """Close the Vault client."""
        self._client = None


class LocalKmsProvider(KmsProvider):
    """
    Local KMS provider for development and testing.

    Uses environment variable ARAGORA_ENCRYPTION_KEY directly.
    NOT for production use.
    """

    def __init__(self, master_key: bytes | None = None):
        self._master_key = master_key
        if self._master_key is None:
            key_hex = os.environ.get("ARAGORA_ENCRYPTION_KEY")
            if key_hex:
                self._master_key = bytes.fromhex(key_hex)

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Return the local master key."""
        if self._master_key is None:
            import secrets

            self._master_key = secrets.token_bytes(32)
            logger.warning(
                "No ARAGORA_ENCRYPTION_KEY set - generated ephemeral key. "
                "Data will not be recoverable after restart."
            )
        return self._master_key

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """XOR decrypt (for testing only)."""
        if self._master_key is None:
            raise ValueError("No master key available")

        # Simple XOR for local testing (NOT SECURE)
        return bytes(a ^ b for a, b in zip(encrypted_key, self._master_key * 2))[:32]

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """XOR encrypt (for testing only)."""
        if self._master_key is None:
            raise ValueError("No master key available")

        # Simple XOR for local testing (NOT SECURE)
        return bytes(a ^ b for a, b in zip(plaintext_key, self._master_key * 2))[:32]

    async def get_key_metadata(self, key_id: str) -> KmsKeyMetadata:
        """Return local key metadata."""
        return KmsKeyMetadata(
            key_id="local-key",
            key_arn=None,
            version="1",
            created_at=None,
            algorithm="AES-256",
            provider="local",
        )

    async def close(self) -> None:
        """No-op for local provider."""
        pass


# =============================================================================
# Provider Factory
# =============================================================================

_kms_provider: KmsProvider | None = None


def detect_cloud_provider() -> str:
    """
    Auto-detect which cloud provider to use based on environment.

    Detection order:
    1. Explicit ARAGORA_KMS_PROVIDER setting
    2. HashiCorp Vault (VAULT_ADDR)
    3. AWS (AWS_REGION or AWS_ACCESS_KEY_ID)
    4. Azure (AZURE_KEY_VAULT_URL)
    5. GCP (GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT)
    6. Local (fallback)
    """
    # Explicit setting takes priority
    explicit = os.environ.get("ARAGORA_KMS_PROVIDER", "").lower()
    if explicit in ("aws", "azure", "gcp", "vault", "local"):
        return explicit

    # HashiCorp Vault detection (check before cloud providers)
    if os.environ.get("VAULT_ADDR"):
        if os.environ.get("VAULT_TOKEN") or os.environ.get("VAULT_ROLE_ID"):
            return "vault"

    # AWS detection
    if os.environ.get("AWS_REGION") or os.environ.get("AWS_ACCESS_KEY_ID"):
        if os.environ.get("ARAGORA_AWS_KMS_KEY_ID"):
            return "aws"

    # Azure detection
    if os.environ.get("AZURE_KEY_VAULT_URL"):
        return "azure"

    # GCP detection
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_CLOUD_PROJECT"):
        if os.environ.get("ARAGORA_GCP_KMS_KEYRING"):
            return "gcp"

    # Fallback to local
    return "local"


def get_kms_provider() -> KmsProvider:
    """
    Get the global KMS provider instance.

    Auto-detects cloud provider from environment variables.
    """
    global _kms_provider

    if _kms_provider is None:
        provider_type = detect_cloud_provider()

        if provider_type == "aws":
            _kms_provider = AwsKmsProvider()
            logger.info("Using AWS KMS provider")
        elif provider_type == "azure":
            _kms_provider = AzureKeyVaultProvider()
            logger.info("Using Azure Key Vault provider")
        elif provider_type == "gcp":
            _kms_provider = GcpKmsProvider()
            logger.info("Using GCP Cloud KMS provider")
        elif provider_type == "vault":
            _kms_provider = HashiCorpVaultProvider()
            logger.info("Using HashiCorp Vault provider")
        else:
            _kms_provider = LocalKmsProvider()
            logger.info("Using local KMS provider (development only)")

    return _kms_provider


def init_kms_provider(provider: KmsProvider) -> None:
    """Initialize with a specific KMS provider."""
    global _kms_provider
    _kms_provider = provider


def reset_kms_provider() -> None:
    """Reset the global KMS provider (for testing)."""
    global _kms_provider
    _kms_provider = None


__all__ = [
    "KmsProvider",
    "KmsKeyMetadata",
    "AwsKmsProvider",
    "AzureKeyVaultProvider",
    "GcpKmsProvider",
    "HashiCorpVaultProvider",
    "LocalKmsProvider",
    "get_kms_provider",
    "init_kms_provider",
    "reset_kms_provider",
    "detect_cloud_provider",
]
