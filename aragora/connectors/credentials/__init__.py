"""
Credential Providers for Enterprise Connectors.

Provides pluggable credential management backends:
- EnvCredentialProvider: Environment variables (default)
- AWSSecretsManagerProvider: AWS Secrets Manager
- VaultCredentialProvider: HashiCorp Vault (future)

Usage:
    from aragora.connectors.credentials import (
        get_credential_provider,
        AWSSecretsManagerProvider,
    )

    # Use AWS Secrets Manager
    provider = AWSSecretsManagerProvider(
        secret_name="aragora/production/api-keys"
    )
    api_key = await provider.get_credential("anthropic_api_key")

    # Or use factory function with auto-detection
    provider = get_credential_provider()  # Auto-detects based on env

Environment Variables:
    CREDENTIAL_PROVIDER: Provider type (env, aws, vault). Default: env
    AWS_SECRET_NAME: AWS Secrets Manager secret name
    AWS_REGION: AWS region for Secrets Manager
"""

from aragora.connectors.credentials.providers import (
    CredentialProvider,
    EnvCredentialProvider,
    AWSSecretsManagerProvider,
    ChainedCredentialProvider,
    CachedCredentialProvider,
    get_credential_provider,
)

__all__ = [
    "CredentialProvider",
    "EnvCredentialProvider",
    "AWSSecretsManagerProvider",
    "ChainedCredentialProvider",
    "CachedCredentialProvider",
    "get_credential_provider",
]
