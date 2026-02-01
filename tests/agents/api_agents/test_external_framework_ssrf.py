"""Tests for SSRF protection in ExternalFrameworkAgent.

Tests cover:
- Rejection of localhost and loopback URLs
- Rejection of private IP ranges (10.x, 172.16.x, 192.168.x)
- Rejection of cloud metadata endpoints (169.254.169.254)
- Rejection of dangerous protocols (file://, gopher://)
- Acceptance of valid external URLs
- Domain allowlist via ARAGORA_GATEWAY_ALLOWED_DOMAINS env var
- IPv6 loopback rejection
- SSRF protection in is_available() method
- SSRF protection in generate() method
- SSRF protection in critique() method
- SSRF protection in vote() method
"""

import os

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.agents.api_agents.external_framework import (
    ExternalFrameworkAgent,
    ExternalFrameworkConfig,
)
from aragora.security.ssrf_protection import SSRFValidationError


class TestSSRFProtectionInit:
    """SSRF protection tests for ExternalFrameworkAgent initialization."""

    def test_rejects_localhost_url(self):
        """Agent should reject localhost URLs."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://localhost:8000")

    def test_rejects_127_0_0_1(self):
        """Agent should reject 127.0.0.1 loopback address."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://127.0.0.1:8000")

    def test_rejects_private_ip_10(self):
        """Agent should reject 10.x.x.x private IP range."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://10.0.0.1:8000")

    def test_rejects_private_ip_172(self):
        """Agent should reject 172.16.x.x private IP range."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://172.16.0.1:8000")

    def test_rejects_private_ip_192(self):
        """Agent should reject 192.168.x.x private IP range."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://192.168.1.1:8000")

    def test_rejects_cloud_metadata_ip(self):
        """Agent should reject AWS/GCP/Azure cloud metadata IP."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://169.254.169.254")

    def test_rejects_file_protocol(self):
        """Agent should reject file:// protocol."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="file:///etc/passwd")

    def test_rejects_gopher_protocol(self):
        """Agent should reject gopher:// protocol."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="gopher://evil.com")

    def test_rejects_0_0_0_0(self):
        """Agent should reject 0.0.0.0 address."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://0.0.0.0:8000")

    def test_rejects_ipv6_loopback(self):
        """Agent should reject IPv6 loopback address."""
        with pytest.raises(SSRFValidationError):
            ExternalFrameworkAgent(base_url="http://[::1]:8000")

    def test_accepts_valid_external_https_url(self):
        """Agent should accept valid external HTTPS URLs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        assert agent.base_url == "https://api.example.com"

    def test_accepts_valid_external_http_url(self):
        """Agent should accept valid external HTTP URLs with ports."""
        agent = ExternalFrameworkAgent(base_url="http://api.example.com:8080")
        assert agent.base_url == "http://api.example.com:8080"

    def test_accepts_valid_subdomain_url(self):
        """Agent should accept valid URLs with subdomains."""
        agent = ExternalFrameworkAgent(base_url="https://framework.agents.example.com")
        assert agent.base_url == "https://framework.agents.example.com"


class TestSSRFDomainAllowlist:
    """Tests for ARAGORA_GATEWAY_ALLOWED_DOMAINS environment variable."""

    def test_domain_allowlist_permits_listed_domain(self):
        """Should allow domains in ARAGORA_GATEWAY_ALLOWED_DOMAINS."""
        with patch.dict(
            os.environ,
            {"ARAGORA_GATEWAY_ALLOWED_DOMAINS": "trusted.com,api.trusted.com"},
        ):
            agent = ExternalFrameworkAgent(base_url="https://trusted.com")
            assert agent.base_url == "https://trusted.com"

    def test_domain_allowlist_blocks_unlisted_domain(self):
        """Should block domains not in ARAGORA_GATEWAY_ALLOWED_DOMAINS."""
        with patch.dict(
            os.environ,
            {"ARAGORA_GATEWAY_ALLOWED_DOMAINS": "trusted.com"},
        ):
            with pytest.raises(SSRFValidationError):
                ExternalFrameworkAgent(base_url="https://untrusted.com")

    def test_domain_allowlist_empty_allows_all_public(self):
        """Empty allowlist should allow all public domains."""
        with patch.dict(
            os.environ,
            {"ARAGORA_GATEWAY_ALLOWED_DOMAINS": ""},
        ):
            agent = ExternalFrameworkAgent(base_url="https://any-public-domain.com")
            assert agent.base_url == "https://any-public-domain.com"

    def test_domain_allowlist_with_whitespace(self):
        """Should handle whitespace in domain list."""
        with patch.dict(
            os.environ,
            {"ARAGORA_GATEWAY_ALLOWED_DOMAINS": " trusted.com , api.trusted.com "},
        ):
            agent = ExternalFrameworkAgent(base_url="https://trusted.com")
            assert agent.base_url == "https://trusted.com"


class TestSSRFIsAvailable:
    """SSRF protection tests for is_available() method."""

    @pytest.mark.asyncio
    async def test_is_available_rejects_unsafe_health_url(self):
        """is_available should return False for unsafe URLs."""
        # Create agent with safe URL first, then tamper with base_url
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        # Simulate base_url being changed to an unsafe value
        agent.base_url = "http://127.0.0.1"
        agent.config.base_url = "http://127.0.0.1"

        result = await agent.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_rejects_metadata_url(self):
        """is_available should return False for cloud metadata URLs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent.base_url = "http://169.254.169.254"
        agent.config.base_url = "http://169.254.169.254"

        result = await agent.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_with_safe_url_proceeds(self):
        """is_available should proceed normally with safe URLs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.is_available()

        assert result is True


class TestSSRFGenerate:
    """SSRF protection tests for generate() method."""

    @pytest.mark.asyncio
    async def test_generate_rejects_unsafe_url(self):
        """generate should raise SSRFValidationError for unsafe URLs."""
        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            enable_circuit_breaker=False,
        )
        # Tamper with base_url after init
        agent.base_url = "http://10.0.0.1"
        agent.config.base_url = "http://10.0.0.1"

        with pytest.raises(SSRFValidationError):
            await agent.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_with_safe_url_proceeds(self):
        """generate should proceed normally with safe URLs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "test output"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.generate("Test prompt")

        assert result == "test output"


class TestSSRFCritique:
    """SSRF protection tests for critique() method."""

    @pytest.mark.asyncio
    async def test_critique_rejects_unsafe_url(self):
        """critique should raise SSRFValidationError for unsafe URLs."""
        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            enable_circuit_breaker=False,
        )
        # Tamper with base_url after init
        agent.base_url = "http://192.168.1.1"
        agent.config.base_url = "http://192.168.1.1"

        with pytest.raises(SSRFValidationError):
            await agent.critique(
                proposal="Test proposal",
                task="Test task",
                target_agent="test-agent",
            )


class TestSSRFVote:
    """SSRF protection tests for vote() method."""

    @pytest.mark.asyncio
    async def test_vote_rejects_unsafe_url(self):
        """vote should raise SSRFValidationError for unsafe URLs."""
        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            enable_circuit_breaker=False,
        )
        # Tamper with base_url after init
        agent.base_url = "http://172.16.0.1"
        agent.config.base_url = "http://172.16.0.1"

        proposals = {
            "agent1": "Proposal A",
            "agent2": "Proposal B",
        }

        with pytest.raises(SSRFValidationError):
            await agent.vote(proposals, "Test task")


class TestSSRFValidateEndpointUrlMethod:
    """Tests for the _validate_endpoint_url helper method directly."""

    def test_validate_blocks_private_ips(self):
        """_validate_endpoint_url should raise for private IPs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        with pytest.raises(SSRFValidationError):
            agent._validate_endpoint_url("http://10.0.0.1/api")

    def test_validate_allows_public_urls(self):
        """_validate_endpoint_url should not raise for public URLs."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        # Should not raise
        agent._validate_endpoint_url("https://api.example.com/generate")

    def test_validate_blocks_localhost_variants(self):
        """_validate_endpoint_url should block all localhost variants."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        for url in [
            "http://localhost/api",
            "http://127.0.0.1/api",
            "http://0.0.0.0/api",
            "http://[::1]/api",
        ]:
            with pytest.raises(SSRFValidationError):
                agent._validate_endpoint_url(url)

    def test_validate_error_includes_url(self):
        """SSRFValidationError should include the blocked URL."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        with pytest.raises(SSRFValidationError) as exc_info:
            agent._validate_endpoint_url("http://127.0.0.1:8000/api")

        assert exc_info.value.url == "http://127.0.0.1:8000/api"

    def test_validate_error_message_describes_issue(self):
        """SSRFValidationError message should describe the blocking reason."""
        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        with pytest.raises(SSRFValidationError, match="Unsafe external framework URL blocked"):
            agent._validate_endpoint_url("http://localhost:8000/api")
