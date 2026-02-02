"""
Configuration Example

Demonstrates various configuration options for the Aragora SDK.
Shows environment variables, custom settings, and client options.

Usage:
    python examples/configuration.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import logging
import os

from aragora_sdk import AragoraAsyncClient, AragoraClient

# =============================================================================
# Environment Variables
# =============================================================================


def environment_variables() -> None:
    """Show environment variable configuration patterns."""
    print("=== Environment Variables ===\n")

    print("Required:")
    print("  ARAGORA_API_KEY - Your API key (required for authenticated requests)")
    print()

    print("Optional:")
    print("  ARAGORA_API_URL - Custom API URL (default: https://api.aragora.ai)")
    print("  ARAGORA_TIMEOUT - Request timeout in seconds (default: 30)")
    print("  ARAGORA_LOG_LEVEL - Logging level (DEBUG, INFO, WARNING, ERROR)")
    print()

    # Show current configuration
    print("Current Configuration:")
    print(f"  API URL: {os.environ.get('ARAGORA_API_URL', 'https://api.aragora.ai')}")
    api_key = os.environ.get("ARAGORA_API_KEY")
    key_display = f"***{api_key[-4:]}" if api_key else "not-set"
    print(f"  API Key: {key_display}")
    print(f"  Timeout: {os.environ.get('ARAGORA_TIMEOUT', '30')}s")
    print(f"  Log Level: {os.environ.get('ARAGORA_LOG_LEVEL', 'WARNING')}")


# =============================================================================
# Client Configuration
# =============================================================================


def client_configuration() -> None:
    """Show client initialization options."""
    print("\n=== Client Configuration Options ===\n")

    # Basic configuration
    print("1. Basic Configuration:")
    print("""
    client = AragoraClient(
        api_key="your-api-key",
    )
    """)

    # Full configuration
    print("2. Full Configuration:")
    print("""
    client = AragoraClient(
        base_url="https://api.aragora.ai",  # API endpoint
        api_key="your-api-key",             # Authentication
        timeout=60.0,                        # Request timeout (seconds)
        max_retries=3,                       # Automatic retry count
        headers={"X-Custom": "value"},       # Additional headers
    )
    """)

    # Environment-based configuration
    print("3. Environment-based Configuration:")
    print("""
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
        timeout=float(os.environ.get("ARAGORA_TIMEOUT", "30")),
    )
    """)


# =============================================================================
# Logging Configuration
# =============================================================================


def logging_configuration() -> None:
    """Show how to configure logging for debugging."""
    print("\n=== Logging Configuration ===\n")

    print("Enable debug logging to see API requests/responses:")
    print()

    # Configure logging
    print("1. Basic debug logging:")
    print("""
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # This will show all HTTP requests and responses
    """)

    # Targeted logging
    print("2. SDK-specific logging:")
    print("""
    import logging

    # Only enable for aragora_sdk
    logging.getLogger("aragora_sdk").setLevel(logging.DEBUG)

    # Or for HTTP client
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    """)

    # Custom handler
    print("3. Custom log handler:")
    print("""
    import logging

    # Create custom handler
    handler = logging.FileHandler("aragora.log")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Add to SDK logger
    logging.getLogger("aragora_sdk").addHandler(handler)
    """)

    # Demonstrate
    print("\nDemonstrating debug output:")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger("aragora_sdk")
    logger.info("This is an INFO message from the SDK")
    logger.debug("This DEBUG message only shows at DEBUG level")


# =============================================================================
# Custom HTTP Client
# =============================================================================


def custom_http_client() -> None:
    """Show how to use a custom HTTP client."""
    print("\n=== Custom HTTP Client ===\n")

    print("For advanced use cases, you can inject a custom httpx client:")
    print()

    print("1. Custom transport (proxies, certificates):")
    print("""
    import httpx

    # Create custom transport with proxy
    transport = httpx.HTTPTransport(proxy="http://proxy:8080")

    # Create httpx client with custom transport
    http_client = httpx.Client(
        transport=transport,
        verify="/path/to/custom/ca-bundle.crt",
    )

    # Use with Aragora client
    client = AragoraClient(
        api_key="your-key",
        http_client=http_client,
    )
    """)

    print("2. Custom timeout configuration:")
    print("""
    import httpx

    # Fine-grained timeout control
    timeout = httpx.Timeout(
        connect=5.0,    # Connection timeout
        read=30.0,      # Read timeout
        write=10.0,     # Write timeout
        pool=5.0,       # Pool timeout
    )

    http_client = httpx.Client(timeout=timeout)

    client = AragoraClient(
        api_key="your-key",
        http_client=http_client,
    )
    """)


# =============================================================================
# Multi-Environment Setup
# =============================================================================


def multi_environment_setup() -> None:
    """Show how to configure for multiple environments."""
    print("\n=== Multi-Environment Setup ===\n")

    print("Configure different environments using environment variables:")
    print()

    print("Development (.env.dev):")
    print("""
    ARAGORA_API_URL=http://localhost:8080
    ARAGORA_API_KEY=dev-key-12345
    ARAGORA_TIMEOUT=60
    ARAGORA_LOG_LEVEL=DEBUG
    """)

    print("Staging (.env.staging):")
    print("""
    ARAGORA_API_URL=https://staging.aragora.ai
    ARAGORA_API_KEY=staging-key-67890
    ARAGORA_TIMEOUT=30
    ARAGORA_LOG_LEVEL=INFO
    """)

    print("Production (.env.prod):")
    print("""
    ARAGORA_API_URL=https://api.aragora.ai
    ARAGORA_API_KEY=prod-key-secret
    ARAGORA_TIMEOUT=30
    ARAGORA_LOG_LEVEL=WARNING
    """)

    print("Loading with python-dotenv:")
    print("""
    from dotenv import load_dotenv
    import os

    # Load environment-specific .env file
    env = os.environ.get("ENV", "dev")
    load_dotenv(f".env.{env}")

    # Create client with loaded config
    client = AragoraClient(
        base_url=os.environ["ARAGORA_API_URL"],
        api_key=os.environ["ARAGORA_API_KEY"],
    )
    """)


# =============================================================================
# Configuration Factory
# =============================================================================


class AragoraConfig:
    """Configuration factory for different environments."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        debug: bool = False,
    ):
        self.api_key = api_key or os.environ.get("ARAGORA_API_KEY")
        self.base_url = base_url or os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai")
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug

    def create_client(self) -> AragoraClient:
        """Create a configured sync client."""
        if self.debug:
            logging.getLogger("aragora_sdk").setLevel(logging.DEBUG)

        return AragoraClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def create_async_client(self) -> AragoraAsyncClient:
        """Create a configured async client."""
        if self.debug:
            logging.getLogger("aragora_sdk").setLevel(logging.DEBUG)

        return AragoraAsyncClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @classmethod
    def development(cls) -> AragoraConfig:
        """Create development configuration."""
        return cls(
            base_url="http://localhost:8080",
            timeout=60.0,
            debug=True,
        )

    @classmethod
    def production(cls) -> AragoraConfig:
        """Create production configuration."""
        return cls(
            timeout=30.0,
            max_retries=3,
            debug=False,
        )


def configuration_factory() -> None:
    """Demonstrate the configuration factory pattern."""
    print("\n=== Configuration Factory Pattern ===\n")

    print("Use a configuration factory for clean environment management:")
    print("""
    # Development
    config = AragoraConfig.development()
    client = config.create_client()

    # Production
    config = AragoraConfig.production()
    client = config.create_client()

    # Custom
    config = AragoraConfig(
        api_key="custom-key",
        base_url="https://custom.aragora.ai",
        timeout=45.0,
        debug=True,
    )
    client = config.create_client()
    """)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run configuration demonstrations."""
    print("Aragora SDK Configuration Examples")
    print("=" * 60)

    environment_variables()
    client_configuration()
    logging_configuration()
    custom_http_client()
    multi_environment_setup()
    configuration_factory()

    print("\n" + "=" * 60)
    print("Configuration examples complete!")
    print("\nKey Takeaways:")
    print("  1. Use environment variables for sensitive data (API keys)")
    print("  2. Configure logging for debugging")
    print("  3. Use configuration factories for multi-environment setups")
    print("  4. Inject custom HTTP clients for advanced networking needs")


if __name__ == "__main__":
    main()
