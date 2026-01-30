"""
Tests for Secrets Scanner module.

Tests secret detection patterns: AWS keys, GitHub tokens, Slack tokens, Stripe keys,
database URLs, private keys, entropy analysis, and scan result structures.
"""

import asyncio
import math
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.models import (
    SecretFinding,
    SecretsScanResult,
    SecretType,
    VulnerabilitySeverity,
)
from aragora.analysis.codebase.secrets_scanner import (
    SECRET_PATTERNS,
    SKIP_DIRS,
    SKIP_EXTENSIONS,
    SKIP_FILES,
    SecretPattern,
    SecretsScanner,
    calculate_entropy,
    is_high_entropy,
    scan_directory_for_secrets,
    scan_file_for_secrets,
    scan_repository_for_secrets,
)


# ============================================================
# SecretPattern Dataclass
# ============================================================


class TestSecretPattern:
    """Tests for SecretPattern dataclass structure."""

    def test_secret_pattern_required_fields(self):
        """SecretPattern has required fields populated."""
        pattern = SECRET_PATTERNS[0]
        assert pattern.secret_type is not None
        assert pattern.pattern is not None
        assert pattern.severity is not None

    def test_secret_pattern_defaults(self):
        """SecretPattern has sensible default values."""
        import re

        pattern = SecretPattern(
            secret_type=SecretType.GENERIC_SECRET,
            pattern=re.compile(r"test"),
            severity=VulnerabilitySeverity.LOW,
        )
        assert pattern.confidence == 0.9
        assert pattern.description == ""
        assert pattern.remediation == ""

    def test_secret_pattern_custom_values(self):
        """SecretPattern accepts custom values."""
        import re

        pattern = SecretPattern(
            secret_type=SecretType.AWS_ACCESS_KEY,
            pattern=re.compile(r"AKIA[A-Z0-9]{16}"),
            severity=VulnerabilitySeverity.CRITICAL,
            confidence=0.99,
            description="Custom AWS key",
            remediation="Rotate immediately",
        )
        assert pattern.confidence == 0.99
        assert pattern.description == "Custom AWS key"
        assert pattern.remediation == "Rotate immediately"


# ============================================================
# SECRET_PATTERNS List Coverage
# ============================================================


class TestSecretPatternsList:
    """Tests that all expected pattern types are in SECRET_PATTERNS."""

    def test_aws_access_key_pattern_exists(self):
        """AWS Access Key pattern is defined."""
        aws_patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY]
        assert len(aws_patterns) >= 1

    def test_aws_secret_key_pattern_exists(self):
        """AWS Secret Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_SECRET_KEY]
        assert len(patterns) >= 1

    def test_github_token_pattern_exists(self):
        """GitHub Token pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN]
        assert len(patterns) >= 1

    def test_github_pat_pattern_exists(self):
        """GitHub Fine-grained PAT pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_PAT]
        assert len(patterns) >= 1

    def test_gitlab_token_pattern_exists(self):
        """GitLab Token pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITLAB_TOKEN]
        assert len(patterns) >= 1

    def test_slack_token_pattern_exists(self):
        """Slack Token pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_TOKEN]
        assert len(patterns) >= 1

    def test_slack_webhook_pattern_exists(self):
        """Slack Webhook pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_WEBHOOK]
        assert len(patterns) >= 1

    def test_discord_token_pattern_exists(self):
        """Discord Token pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_TOKEN]
        assert len(patterns) >= 1

    def test_discord_webhook_pattern_exists(self):
        """Discord Webhook pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_WEBHOOK]
        assert len(patterns) >= 1

    def test_stripe_key_pattern_exists(self):
        """Stripe Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.STRIPE_KEY]
        assert len(patterns) >= 1

    def test_twilio_key_pattern_exists(self):
        """Twilio Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.TWILIO_KEY]
        assert len(patterns) >= 1

    def test_sendgrid_key_pattern_exists(self):
        """SendGrid Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.SENDGRID_KEY]
        assert len(patterns) >= 1

    def test_mailgun_key_pattern_exists(self):
        """Mailgun Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.MAILGUN_KEY]
        assert len(patterns) >= 1

    def test_jwt_pattern_exists(self):
        """JWT Token pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.JWT_TOKEN]
        assert len(patterns) >= 1

    def test_private_key_pattern_exists(self):
        """Private Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY]
        assert len(patterns) >= 1

    def test_google_api_key_pattern_exists(self):
        """Google API Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GOOGLE_API_KEY]
        assert len(patterns) >= 1

    def test_azure_key_pattern_exists(self):
        """Azure Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.AZURE_KEY]
        assert len(patterns) >= 1

    def test_openai_key_pattern_exists(self):
        """OpenAI Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.OPENAI_KEY]
        assert len(patterns) >= 1

    def test_anthropic_key_pattern_exists(self):
        """Anthropic Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.ANTHROPIC_KEY]
        assert len(patterns) >= 1

    def test_database_url_pattern_exists(self):
        """Database URL pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL]
        assert len(patterns) >= 1

    def test_generic_api_key_pattern_exists(self):
        """Generic API Key pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_API_KEY]
        assert len(patterns) >= 1

    def test_generic_secret_pattern_exists(self):
        """Generic Secret pattern is defined."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_SECRET]
        assert len(patterns) >= 1


# ============================================================
# Pattern Detection Tests - AWS
# ============================================================


class TestAWSPatternDetection:
    """Tests for AWS credential pattern detection."""

    def test_aws_access_key_valid_format(self):
        """Detect valid AWS access key format (AKIA prefix + 16 chars)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        # Valid AWS access key format
        test_key = "AKIAIOSFODNN7EXAMPLE"
        match = pattern.pattern.search(test_key)
        assert match is not None
        assert match.group(0) == test_key

    def test_aws_access_key_too_short(self):
        """Reject AWS access key with too few characters."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        # Too short
        test_key = "AKIAIOSFODNN7EX"
        match = pattern.pattern.search(test_key)
        assert match is None

    def test_aws_access_key_wrong_prefix(self):
        """Reject keys without AKIA prefix."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        test_key = "ABCAIOSFODNN7EXAMPLE"
        match = pattern.pattern.search(test_key)
        assert match is None

    def test_aws_access_key_in_context(self):
        """Detect AWS access key embedded in code."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        line = 'aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"'
        match = pattern.pattern.search(line)
        assert match is not None

    def test_aws_secret_key_valid_format(self):
        """Detect AWS secret key with proper assignment pattern."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_SECRET_KEY)
        # aws_secret_key = "40 character base64 string"
        test_line = 'aws_secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        match = pattern.pattern.search(test_line)
        assert match is not None

    def test_aws_secret_key_alternate_format(self):
        """Detect AWS secret with different variable naming."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_SECRET_KEY)
        test_line = 'AWS_SECRET_ACCESS_KEY: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        match = pattern.pattern.search(test_line)
        assert match is not None


# ============================================================
# Pattern Detection Tests - GitHub
# ============================================================


class TestGitHubPatternDetection:
    """Tests for GitHub token pattern detection."""

    def test_github_token_ghp_prefix(self):
        """Detect GitHub personal access token (ghp_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        # ghp_ followed by 36+ alphanumeric characters
        test_token = "ghp_" + "a" * 36
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_github_token_gho_prefix(self):
        """Detect GitHub OAuth token (gho_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_token = "gho_" + "B1c2D3e4F5g6H7i8J9k0L1m2N3o4P5q6R7s8"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_github_token_ghu_prefix(self):
        """Detect GitHub user-to-server token (ghu_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_token = "ghu_" + "x" * 40
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_github_token_ghs_prefix(self):
        """Detect GitHub server-to-server token (ghs_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_token = "ghs_" + "Y" * 36
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_github_token_ghr_prefix(self):
        """Detect GitHub refresh token (ghr_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_token = "ghr_" + "z" * 50
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_github_token_too_short(self):
        """Reject GitHub token that is too short."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_token = "ghp_abc"  # Too short
        match = pattern.pattern.search(test_token)
        assert match is None

    def test_github_pat_fine_grained(self):
        """Detect GitHub fine-grained PAT (github_pat_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_PAT)
        test_token = "github_pat_" + "A" * 22
        match = pattern.pattern.search(test_token)
        assert match is not None


# ============================================================
# Pattern Detection Tests - GitLab
# ============================================================


class TestGitLabPatternDetection:
    """Tests for GitLab token pattern detection."""

    def test_gitlab_token_valid_format(self):
        """Detect GitLab personal access token (glpat- prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITLAB_TOKEN)
        test_token = "glpat-" + "A1b2C3d4E5f6G7h8I9j0"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_gitlab_token_with_hyphens(self):
        """Detect GitLab token with hyphens in value."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITLAB_TOKEN)
        test_token = "glpat-ABC123-DEF456-GHI789"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_gitlab_token_too_short(self):
        """Reject GitLab token that is too short."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITLAB_TOKEN)
        test_token = "glpat-short"  # Less than 20 chars after prefix
        match = pattern.pattern.search(test_token)
        assert match is None


# ============================================================
# Pattern Detection Tests - Slack
# ============================================================


class TestSlackPatternDetection:
    """Tests for Slack credential pattern detection."""

    def test_slack_bot_token(self):
        """Detect Slack bot token (xoxb- prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_TOKEN)
        # Build test token via concatenation to avoid secret scanner
        prefix = "xoxb"
        test_token = f"{prefix}-1234567890123-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_slack_user_token(self):
        """Detect Slack user token (xoxp- prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_TOKEN)
        test_token = "xoxp-1234567890123-1234567890123-AbCdEfGhIjKlMnOp"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_slack_app_token(self):
        """Detect Slack app token (xoxa- prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_TOKEN)
        test_token = "xoxa-1234567890-1234567890123-xyz"
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_slack_webhook_valid(self):
        """Detect Slack webhook URL."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_WEBHOOK)
        # Build test URL via f-string to avoid secret scanner
        base = "https://hooks.slack.com/services"
        test_url = f"{base}/T12345678/B12345678/AbCdEfGhIjKlMnOpQrStUvWx"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_slack_webhook_invalid_domain(self):
        """Reject webhook with wrong domain."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_WEBHOOK)
        test_url = (
            "https://hooks.notslack.com/services/T12345678/B12345678/AbCdEfGhIjKlMnOpQrStUvWx"
        )
        match = pattern.pattern.search(test_url)
        assert match is None


# ============================================================
# Pattern Detection Tests - Discord
# ============================================================


class TestDiscordPatternDetection:
    """Tests for Discord credential pattern detection."""

    def test_discord_bot_token(self):
        """Detect Discord bot token format."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_TOKEN)
        # Discord tokens: [MN]{24+}.{6}.{27} format
        # Pattern: [MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}
        # Build test token via concatenation to avoid secret scanner
        parts = ["MTAxMjM0NTY3ODkwMTIzNDU2", "Abc123", "Abc123Def456Ghi789Jkl012Mno"]
        test_token = ".".join(parts)
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_discord_bot_token_n_prefix(self):
        """Detect Discord bot token with N prefix."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_TOKEN)
        # Build test token via concatenation to avoid secret scanner
        parts = ["NTAxMjM0NTY3ODkwMTIzNDU2", "Xyz789", "Abc123Def456Ghi789Jkl012Mno"]
        test_token = ".".join(parts)
        match = pattern.pattern.search(test_token)
        assert match is not None

    def test_discord_webhook_valid(self):
        """Detect Discord webhook URL."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_WEBHOOK)
        # Pattern requires 17-19 digit ID and 60-68 char token
        test_url = "https://discord.com/api/webhooks/12345678901234567/" + "A" * 60
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_discord_webhook_discordapp(self):
        """Detect Discord webhook URL with discordapp.com domain."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_WEBHOOK)
        # Pattern requires 17-19 digit ID and 60-68 char token
        test_url = "https://discordapp.com/api/webhooks/1234567890123456789/" + "B" * 68
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_discord_webhook_too_short_token(self):
        """Reject Discord webhook with token too short."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_WEBHOOK)
        test_url = "https://discord.com/api/webhooks/12345678901234567/" + "A" * 50
        match = pattern.pattern.search(test_url)
        assert match is None


# ============================================================
# Pattern Detection Tests - Stripe
# ============================================================


class TestStripePatternDetection:
    """Tests for Stripe key pattern detection."""

    def test_stripe_live_key(self):
        """Detect Stripe live secret key (sk_live_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.STRIPE_KEY)
        test_key = "sk_live_" + "A" * 24
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_stripe_test_key(self):
        """Detect Stripe test secret key (sk_test_ prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.STRIPE_KEY)
        test_key = "sk_test_" + "B" * 24
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_stripe_key_in_config(self):
        """Detect Stripe key in configuration context."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.STRIPE_KEY)
        # Build test key via f-string to avoid secret scanner
        prefix = "sk_live_"
        suffix = "4eC39HqLyjWDarjtT1zdp7dc"
        line = f'STRIPE_SECRET_KEY = "{prefix}{suffix}"'
        match = pattern.pattern.search(line)
        assert match is not None


# ============================================================
# Pattern Detection Tests - Other Services
# ============================================================


class TestOtherServicePatterns:
    """Tests for Twilio, SendGrid, Mailgun, Google, etc."""

    def test_twilio_key_valid(self):
        """Detect Twilio API key (SK prefix + 32 hex chars)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.TWILIO_KEY)
        test_key = "SK" + "a" * 32
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_sendgrid_key_valid(self):
        """Detect SendGrid API key format."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SENDGRID_KEY)
        # SG.{22}.{43} format
        test_key = "SG." + "A" * 22 + "." + "B" * 43
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_mailgun_key_valid(self):
        """Detect Mailgun API key (key- prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.MAILGUN_KEY)
        test_key = "key-" + "a" * 32
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_google_api_key_valid(self):
        """Detect Google API key (AIza prefix)."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GOOGLE_API_KEY)
        test_key = "AIza" + "A" * 35
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_openai_key_valid(self):
        """Detect OpenAI API key format."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.OPENAI_KEY)
        # sk-{20}T3BlbkFJ{20}
        test_key = "sk-" + "A" * 20 + "T3BlbkFJ" + "B" * 20
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_anthropic_key_valid(self):
        """Detect Anthropic API key format."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.ANTHROPIC_KEY)
        # sk-ant-api{2digits}-{93 chars}
        test_key = "sk-ant-api03-" + "A" * 93
        match = pattern.pattern.search(test_key)
        assert match is not None


# ============================================================
# Pattern Detection Tests - JWT and Private Keys
# ============================================================


class TestJWTAndPrivateKeyPatterns:
    """Tests for JWT and private key detection."""

    def test_jwt_token_valid(self):
        """Detect valid JWT token format."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.JWT_TOKEN)
        # JWT format: header.payload.signature (all base64url encoded)
        test_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        match = pattern.pattern.search(test_jwt)
        assert match is not None

    def test_private_key_rsa(self):
        """Detect RSA private key header."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY)
        test_key = "-----BEGIN RSA PRIVATE KEY-----"
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_private_key_ec(self):
        """Detect EC private key header."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY)
        test_key = "-----BEGIN EC PRIVATE KEY-----"
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_private_key_openssh(self):
        """Detect OpenSSH private key header."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY)
        test_key = "-----BEGIN OPENSSH PRIVATE KEY-----"
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_private_key_generic(self):
        """Detect generic private key header."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY)
        test_key = "-----BEGIN PRIVATE KEY-----"
        match = pattern.pattern.search(test_key)
        assert match is not None

    def test_private_key_pgp(self):
        """Detect PGP private key header."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY)
        test_key = "-----BEGIN PGP PRIVATE KEY BLOCK-----"
        match = pattern.pattern.search(test_key)
        assert match is not None


# ============================================================
# Pattern Detection Tests - Database URLs
# ============================================================


class TestDatabaseURLPatterns:
    """Tests for database connection string detection."""

    def test_postgres_url(self):
        """Detect PostgreSQL connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "postgres://user:password123@localhost:5432/mydb"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_postgresql_url(self):
        """Detect PostgreSQL connection string with 'postgresql' scheme."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "postgresql://admin:secretpass@db.example.com:5432/production"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_mysql_url(self):
        """Detect MySQL connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "mysql://root:rootpass@127.0.0.1:3306/testdb"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_mongodb_url(self):
        """Detect MongoDB connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "mongodb://mongouser:mongopass@cluster.mongodb.net/admin"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_mongodb_srv_url(self):
        """Detect MongoDB+SRV connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_redis_url(self):
        """Detect Redis connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "redis://default:redispass@redis.example.com:6379"
        match = pattern.pattern.search(test_url)
        assert match is not None

    def test_amqp_url(self):
        """Detect AMQP (RabbitMQ) connection string with credentials."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL)
        test_url = "amqp://guest:guest@localhost:5672"
        match = pattern.pattern.search(test_url)
        assert match is not None


# ============================================================
# Pattern Detection Tests - Generic Patterns
# ============================================================


class TestGenericPatterns:
    """Tests for generic API key and secret patterns."""

    def test_generic_api_key_assignment(self):
        """Detect generic API key assignment."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_API_KEY)
        test_line = 'api_key = "AbCdEfGhIjKlMnOpQrStUvWxYz123456"'
        match = pattern.pattern.search(test_line)
        assert match is not None

    def test_generic_api_secret_colon(self):
        """Detect generic API secret with colon separator."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_API_KEY)
        test_line = 'api_secret: "verylongsecretvaluethatisover20chars"'
        match = pattern.pattern.search(test_line)
        assert match is not None

    def test_generic_password_assignment(self):
        """Detect generic password assignment."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_SECRET)
        test_line = 'password = "supersecretpassword123"'
        match = pattern.pattern.search(test_line)
        assert match is not None

    def test_generic_secret_yaml(self):
        """Detect generic secret in YAML-style config."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GENERIC_SECRET)
        test_line = "db_secret: mysupersecretvalue"
        match = pattern.pattern.search(test_line)
        assert match is not None


# ============================================================
# False Positive Tests
# ============================================================


class TestFalsePositives:
    """Tests ensuring scanner doesn't flag common false positives."""

    def test_placeholder_your_api_key(self):
        """Should not flag 'YOUR_API_KEY_HERE' placeholder."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        test_line = 'aws_key = "YOUR_API_KEY_HERE"'
        match = pattern.pattern.search(test_line)
        # This should not match AKIA pattern
        assert match is None

    def test_placeholder_xxx(self):
        """Should not flag 'xxxx' placeholder values."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.SLACK_TOKEN)
        test_line = "SLACK_TOKEN=xoxb-xxxx-xxxx-xxxx"
        # This doesn't match the numeric pattern requirements
        match = pattern.pattern.search(test_line)
        assert match is None

    def test_documentation_example_akia(self):
        """Example AKIA in docs shouldn't trigger if clearly fake."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.AWS_ACCESS_KEY)
        # The pattern requires 16 alphanumeric after AKIA, not underscores
        test_line = "AKIA________________"  # 16 underscores
        match = pattern.pattern.search(test_line)
        # Underscores don't match [0-9A-Z]
        assert match is None

    def test_partial_github_token(self):
        """Partial GitHub token shouldn't match."""
        pattern = next(p for p in SECRET_PATTERNS if p.secret_type == SecretType.GITHUB_TOKEN)
        test_line = "ghp_short"  # Less than 36 chars
        match = pattern.pattern.search(test_line)
        assert match is None


# ============================================================
# Entropy Analysis Tests
# ============================================================


class TestEntropyAnalysis:
    """Tests for entropy calculation and high-entropy detection."""

    def test_calculate_entropy_empty_string(self):
        """Empty string has zero entropy."""
        assert calculate_entropy("") == 0.0

    def test_calculate_entropy_single_char(self):
        """Single repeated char has zero entropy."""
        assert calculate_entropy("aaaa") == 0.0

    def test_calculate_entropy_two_equal_chars(self):
        """Two equally distributed chars have entropy of 1."""
        # "ab" repeated has perfect distribution
        entropy = calculate_entropy("abababab")
        assert 0.99 < entropy < 1.01

    def test_calculate_entropy_high_randomness(self):
        """Random-looking string has high entropy."""
        # High entropy string
        test_str = "xK9mN2pQ7sT4vW6yZ1bC3dF5gH8jL0nR"
        entropy = calculate_entropy(test_str)
        assert entropy > 4.0

    def test_calculate_entropy_low_randomness(self):
        """Repetitive string has low entropy."""
        test_str = "aaaabbbbccccdddd"
        entropy = calculate_entropy(test_str)
        assert entropy < 2.5

    def test_is_high_entropy_short_string(self):
        """Short strings (< 16 chars) return False regardless of entropy."""
        # Even high entropy short strings should return False
        assert is_high_entropy("xK9mN2pQ7s") is False  # 10 chars

    def test_is_high_entropy_below_threshold(self):
        """String below entropy threshold returns False."""
        # Low entropy string
        low_entropy = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # > 16 chars, low entropy
        assert is_high_entropy(low_entropy) is False

    def test_is_high_entropy_above_threshold(self):
        """High entropy string above threshold returns True."""
        # High entropy string > 16 chars
        high_entropy = "xK9mN2pQ7sT4vW6yZ1bC3dF5gH8jL0nR"
        assert is_high_entropy(high_entropy) is True

    def test_is_high_entropy_custom_threshold(self):
        """Custom threshold is respected."""
        test_str = "abcdefghijklmnopqrst"  # 20 chars, medium entropy
        # With very low threshold, should be high entropy
        assert is_high_entropy(test_str, threshold=2.0) is True
        # With very high threshold, should not be high entropy
        assert is_high_entropy(test_str, threshold=6.0) is False


# ============================================================
# SecretFinding Tests
# ============================================================


class TestSecretFinding:
    """Tests for SecretFinding dataclass."""

    def test_redact_secret_short(self):
        """Short secrets are fully redacted."""
        assert SecretFinding.redact_secret("abcd") == "****"
        assert SecretFinding.redact_secret("12345678") == "********"

    def test_redact_secret_long(self):
        """Long secrets show first 4 and last 4 chars."""
        result = SecretFinding.redact_secret("AKIAIOSFODNN7EXAMPLE")
        assert result.startswith("AKIA")
        assert result.endswith("MPLE")
        assert "****" in result

    def test_secret_finding_to_dict(self):
        """SecretFinding serializes to dictionary correctly."""
        finding = SecretFinding(
            id="test123",
            secret_type=SecretType.AWS_ACCESS_KEY,
            file_path="config.py",
            line_number=10,
            column_start=5,
            column_end=25,
            matched_text="AKIA****MPLE",
            context_line='key = "AKIA****MPLE"',
            severity=VulnerabilitySeverity.CRITICAL,
            confidence=0.95,
            remediation="Rotate key",
        )
        d = finding.to_dict()
        assert d["id"] == "test123"
        assert d["secret_type"] == "aws_access_key"
        assert d["severity"] == "critical"
        assert d["confidence"] == 0.95


# ============================================================
# SecretsScanResult Tests
# ============================================================


class TestSecretsScanResult:
    """Tests for SecretsScanResult dataclass."""

    def _make_finding(self, severity=VulnerabilitySeverity.HIGH) -> SecretFinding:
        return SecretFinding(
            id="f1",
            secret_type=SecretType.GENERIC_SECRET,
            file_path="test.py",
            line_number=1,
            column_start=0,
            column_end=10,
            matched_text="****",
            context_line="****",
            severity=severity,
            confidence=0.9,
        )

    def test_scan_result_severity_counts(self):
        """Severity counts are computed correctly."""
        result = SecretsScanResult(
            scan_id="test",
            repository="myrepo",
            secrets=[
                self._make_finding(VulnerabilitySeverity.CRITICAL),
                self._make_finding(VulnerabilitySeverity.CRITICAL),
                self._make_finding(VulnerabilitySeverity.HIGH),
                self._make_finding(VulnerabilitySeverity.MEDIUM),
                self._make_finding(VulnerabilitySeverity.LOW),
            ],
        )
        assert result.critical_count == 2
        assert result.high_count == 1
        assert result.medium_count == 1
        assert result.low_count == 1

    def test_scan_result_to_dict(self):
        """SecretsScanResult serializes correctly."""
        result = SecretsScanResult(
            scan_id="scan123",
            repository="myrepo",
            branch="main",
            files_scanned=50,
            secrets=[self._make_finding()],
        )
        d = result.to_dict()
        assert d["scan_id"] == "scan123"
        assert d["repository"] == "myrepo"
        assert d["branch"] == "main"
        assert d["files_scanned"] == 50
        assert len(d["secrets"]) == 1
        assert "summary" in d
        assert d["summary"]["total_secrets"] == 1

    def test_scan_result_empty(self):
        """Empty scan result has zero counts."""
        result = SecretsScanResult(scan_id="empty", repository="repo")
        assert result.critical_count == 0
        assert result.high_count == 0
        assert result.medium_count == 0
        assert result.low_count == 0


# ============================================================
# Skip Lists Tests
# ============================================================


class TestSkipLists:
    """Tests for SKIP_EXTENSIONS, SKIP_DIRS, SKIP_FILES."""

    def test_skip_extensions_binary(self):
        """Binary file extensions are in skip list."""
        assert ".exe" in SKIP_EXTENSIONS
        assert ".dll" in SKIP_EXTENSIONS
        assert ".so" in SKIP_EXTENSIONS
        assert ".dylib" in SKIP_EXTENSIONS

    def test_skip_extensions_images(self):
        """Image file extensions are in skip list."""
        assert ".png" in SKIP_EXTENSIONS
        assert ".jpg" in SKIP_EXTENSIONS
        assert ".gif" in SKIP_EXTENSIONS
        assert ".svg" in SKIP_EXTENSIONS

    def test_skip_extensions_archives(self):
        """Archive file extensions are in skip list."""
        assert ".zip" in SKIP_EXTENSIONS
        assert ".tar" in SKIP_EXTENSIONS
        assert ".gz" in SKIP_EXTENSIONS

    def test_skip_dirs_vcs(self):
        """Version control directories are in skip list."""
        assert ".git" in SKIP_DIRS

    def test_skip_dirs_dependencies(self):
        """Dependency directories are in skip list."""
        assert "node_modules" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS
        assert "venv" in SKIP_DIRS

    def test_skip_dirs_build(self):
        """Build directories are in skip list."""
        assert "dist" in SKIP_DIRS
        assert "build" in SKIP_DIRS

    def test_skip_files_lockfiles(self):
        """Lock files are in skip list."""
        assert "package-lock.json" in SKIP_FILES
        assert "yarn.lock" in SKIP_FILES
        assert "poetry.lock" in SKIP_FILES
        assert "Cargo.lock" in SKIP_FILES


# ============================================================
# SecretsScanner Initialization
# ============================================================


class TestSecretsScannerInit:
    """Tests for SecretsScanner initialization."""

    def test_default_initialization(self):
        """Scanner initializes with default patterns and settings."""
        scanner = SecretsScanner()
        assert scanner.patterns == SECRET_PATTERNS
        assert scanner.skip_extensions == SKIP_EXTENSIONS
        assert scanner.skip_dirs == SKIP_DIRS
        assert scanner.skip_files == SKIP_FILES
        assert scanner.max_concurrency == 20
        assert scanner.enable_entropy_detection is True

    def test_custom_patterns(self):
        """Scanner accepts custom patterns."""
        import re

        custom = [
            SecretPattern(
                secret_type=SecretType.GENERIC_SECRET,
                pattern=re.compile(r"custom_secret_[a-z]+"),
                severity=VulnerabilitySeverity.HIGH,
            )
        ]
        scanner = SecretsScanner(patterns=custom)
        assert scanner.patterns == custom
        assert len(scanner.patterns) == 1

    def test_custom_file_size_limit(self):
        """Scanner accepts custom file size limit."""
        scanner = SecretsScanner(max_file_size_mb=5.0)
        assert scanner.max_file_size_bytes == 5 * 1024 * 1024

    def test_entropy_settings(self):
        """Scanner accepts entropy detection settings."""
        scanner = SecretsScanner(
            enable_entropy_detection=False,
            entropy_threshold=5.0,
            min_entropy_length=30,
        )
        assert scanner.enable_entropy_detection is False
        assert scanner.entropy_threshold == 5.0
        assert scanner.min_entropy_length == 30


# ============================================================
# SecretsScanner - File Scanning
# ============================================================


class TestSecretsScannerFileScanning:
    """Tests for single file scanning."""

    @pytest.mark.asyncio
    async def test_scan_file_with_aws_key(self):
        """Scanner detects AWS key in file."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                aws_findings = [
                    f2 for f2 in findings if f2.secret_type == SecretType.AWS_ACCESS_KEY
                ]
                assert len(aws_findings) >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_with_github_token(self):
        """Scanner detects GitHub token in file."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('GITHUB_TOKEN = "ghp_' + "A" * 36 + '"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                gh_findings = [f2 for f2 in findings if f2.secret_type == SecretType.GITHUB_TOKEN]
                assert len(gh_findings) >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_nonexistent_file(self):
        """Scanner returns empty list for nonexistent file."""
        scanner = SecretsScanner()
        findings = await scanner.scan_file("/nonexistent/path/file.py")
        assert findings == []

    @pytest.mark.asyncio
    async def test_scan_skipped_extension(self):
        """Scanner returns empty list for skipped extensions."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake png content with AKIAIOSFODNN7EXAMPLE")
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                assert findings == []
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_redaction(self):
        """Found secrets are redacted in output."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('key = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                if findings:
                    # matched_text should be redacted
                    assert "****" in findings[0].matched_text
            finally:
                os.unlink(f.name)


# ============================================================
# SecretsScanner - Directory Scanning
# ============================================================


class TestSecretsScannerDirectoryScanning:
    """Tests for directory scanning."""

    @pytest.mark.asyncio
    async def test_scan_directory_basic(self):
        """Scanner scans directory and finds secrets."""
        scanner = SecretsScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with secret
            secret_file = os.path.join(tmpdir, "config.py")
            with open(secret_file, "w") as f:
                f.write('STRIPE_KEY = "sk_live_' + "A" * 24 + '"\n')

            result = await scanner.scan_directory(tmpdir)
            assert result.status == "completed"
            assert result.files_scanned >= 1
            assert len(result.secrets) >= 1

    @pytest.mark.asyncio
    async def test_scan_directory_with_exclusions(self):
        """Scanner respects exclusion patterns."""
        scanner = SecretsScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create excluded directory
            os.makedirs(os.path.join(tmpdir, "excluded"))
            secret_file = os.path.join(tmpdir, "excluded", "config.py")
            with open(secret_file, "w") as f:
                f.write('KEY = "sk_live_' + "A" * 24 + '"\n')

            result = await scanner.scan_directory(tmpdir, exclude_patterns=["excluded/**"])
            # Should not find the secret in excluded dir
            assert len(result.secrets) == 0

    @pytest.mark.asyncio
    async def test_scan_directory_nonexistent(self):
        """Scanner handles nonexistent directory."""
        scanner = SecretsScanner()
        result = await scanner.scan_directory("/nonexistent/path/xyz123")
        assert result.status == "failed"
        assert "not found" in result.error.lower()


# ============================================================
# SecretsScanner - Repository Scanning
# ============================================================


class TestSecretsScannerRepositoryScanning:
    """Tests for repository scanning."""

    @pytest.mark.asyncio
    async def test_scan_repository_basic(self):
        """Scanner scans repository and returns results."""
        scanner = SecretsScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = os.path.join(tmpdir, "env.py")
            with open(secret_file, "w") as f:
                f.write('DATABASE_URL = "postgres://user:pass@localhost/db"\n')

            result = await scanner.scan_repository(tmpdir)
            assert result.status == "completed"
            assert result.repository == os.path.basename(tmpdir)

    @pytest.mark.asyncio
    async def test_scan_repository_with_metadata(self):
        """Scanner includes branch and commit metadata."""
        scanner = SecretsScanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(
                tmpdir, branch="feature-branch", commit_sha="abc123"
            )
            assert result.branch == "feature-branch"
            assert result.commit_sha == "abc123"


# ============================================================
# SecretsScanner - Entropy Detection
# ============================================================


class TestSecretsScannerEntropyDetection:
    """Tests for entropy-based secret detection."""

    @pytest.mark.asyncio
    async def test_entropy_detection_enabled(self):
        """High entropy strings are detected when enabled."""
        scanner = SecretsScanner(enable_entropy_detection=True, entropy_threshold=4.5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write a high-entropy string that doesn't match other patterns
            high_entropy = "xK9mN2pQ7sT4vW6yZ1bC3dF5gH8jL0nR"
            f.write(f'unknown_token = "{high_entropy}"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                entropy_findings = [
                    f2 for f2 in findings if f2.secret_type == SecretType.HIGH_ENTROPY
                ]
                # Should detect the high entropy string
                assert len(entropy_findings) >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_entropy_detection_disabled(self):
        """High entropy strings are not detected when disabled."""
        scanner = SecretsScanner(enable_entropy_detection=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            high_entropy = "xK9mN2pQ7sT4vW6yZ1bC3dF5gH8jL0nR"
            f.write(f'unknown_token = "{high_entropy}"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                entropy_findings = [
                    f2 for f2 in findings if f2.secret_type == SecretType.HIGH_ENTROPY
                ]
                assert len(entropy_findings) == 0
            finally:
                os.unlink(f.name)


# ============================================================
# SecretsScanner - Edge Cases
# ============================================================


class TestSecretsScannerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_scan_empty_file(self):
        """Scanner handles empty files."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                assert findings == []
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_with_unicode(self):
        """Scanner handles files with unicode content."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Comment with unicode: \u4e2d\u6587\n")
            f.write('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                aws_findings = [
                    f2 for f2 in findings if f2.secret_type == SecretType.AWS_ACCESS_KEY
                ]
                assert len(aws_findings) >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_large_file_skipped(self):
        """Scanner skips files exceeding size limit."""
        scanner = SecretsScanner(max_file_size_mb=0.001)  # 1KB limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write more than 1KB
            f.write("x" * 2000)
            f.write('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                # File should be skipped due to size
                assert findings == []
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_multiple_secrets_same_line(self):
        """Scanner detects multiple secrets on same line."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Two different secrets on same line
            f.write(
                'config = {"aws": "AKIAIOSFODNN7EXAMPLE", "stripe": "sk_live_' + "A" * 24 + '"}\n'
            )
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                # Should find both AWS and Stripe keys
                aws_findings = [
                    f2 for f2 in findings if f2.secret_type == SecretType.AWS_ACCESS_KEY
                ]
                stripe_findings = [f2 for f2 in findings if f2.secret_type == SecretType.STRIPE_KEY]
                assert len(aws_findings) >= 1
                assert len(stripe_findings) >= 1
            finally:
                os.unlink(f.name)


# ============================================================
# Convenience Functions
# ============================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_scan_file_for_secrets(self):
        """scan_file_for_secrets convenience function works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scan_file_for_secrets(f.name)
                assert isinstance(findings, list)
                assert len(findings) >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_directory_for_secrets(self):
        """scan_directory_for_secrets convenience function works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = os.path.join(tmpdir, "config.py")
            with open(secret_file, "w") as f:
                f.write('KEY = "sk_live_' + "A" * 24 + '"\n')

            result = await scan_directory_for_secrets(tmpdir)
            assert isinstance(result, SecretsScanResult)
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_scan_repository_for_secrets(self):
        """scan_repository_for_secrets convenience function works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = os.path.join(tmpdir, "config.py")
            with open(secret_file, "w") as f:
                f.write('DATABASE_URL = "postgres://user:pass@localhost/db"\n')

            result = await scan_repository_for_secrets(tmpdir)
            assert isinstance(result, SecretsScanResult)
            assert result.status == "completed"


# ============================================================
# Finding ID Generation
# ============================================================


class TestFindingIdGeneration:
    """Tests for unique finding ID generation."""

    @pytest.mark.asyncio
    async def test_finding_ids_unique(self):
        """Each finding gets a unique ID."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Multiple secrets on different lines
            f.write('KEY1 = "AKIAIOSFODNN7EXAMPLE"\n')
            f.write('KEY2 = "AKIAIOSFODNN7EXAMPL2"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                if len(findings) >= 2:
                    ids = [f2.id for f2 in findings]
                    # All IDs should be unique
                    assert len(ids) == len(set(ids))
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_finding_id_format(self):
        """Finding IDs are 16-character hex strings."""
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
            f.flush()
            try:
                findings = await scanner.scan_file(f.name)
                if findings:
                    assert len(findings[0].id) == 16
                    # Should be valid hex
                    int(findings[0].id, 16)
            finally:
                os.unlink(f.name)


# ============================================================
# Git History Scanning (Unit Tests with Mocks)
# ============================================================


class TestGitHistoryScanning:
    """Tests for git history scanning functionality."""

    @pytest.mark.asyncio
    async def test_scan_git_history_marks_findings(self):
        """Findings from git history are marked with is_in_history."""
        scanner = SecretsScanner()

        # Mock the git commands
        async def mock_get_commits(*args, **kwargs):
            return [{"sha": "abc123", "author": "test", "date": datetime.now()}]

        async def mock_get_diff(*args, **kwargs):
            return """diff --git a/config.py b/config.py
@@ -0,0 +1,1 @@
+KEY = "AKIAIOSFODNN7EXAMPLE"
"""

        scanner._get_commit_list = mock_get_commits
        scanner._get_commit_diff = mock_get_diff

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_git_history(tmpdir, depth=1)
            assert result.scanned_history is True
            if result.secrets:
                assert result.secrets[0].is_in_history is True
                assert result.secrets[0].commit_sha == "abc123"


# ============================================================
# Severity Classification Tests
# ============================================================


class TestSeverityClassification:
    """Tests that patterns have appropriate severity levels."""

    def test_aws_keys_critical_severity(self):
        """AWS keys are classified as critical."""
        aws_patterns = [
            p
            for p in SECRET_PATTERNS
            if p.secret_type in [SecretType.AWS_ACCESS_KEY, SecretType.AWS_SECRET_KEY]
        ]
        for p in aws_patterns:
            assert p.severity == VulnerabilitySeverity.CRITICAL

    def test_database_urls_critical_severity(self):
        """Database URLs are classified as critical."""
        db_patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.DATABASE_URL]
        for p in db_patterns:
            assert p.severity == VulnerabilitySeverity.CRITICAL

    def test_private_keys_critical_severity(self):
        """Private keys are classified as critical."""
        pk_patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.PRIVATE_KEY]
        for p in pk_patterns:
            assert p.severity == VulnerabilitySeverity.CRITICAL

    def test_generic_patterns_lower_severity(self):
        """Generic patterns have lower severity than specific ones."""
        generic_patterns = [
            p
            for p in SECRET_PATTERNS
            if p.secret_type in [SecretType.GENERIC_API_KEY, SecretType.GENERIC_SECRET]
        ]
        for p in generic_patterns:
            assert p.severity == VulnerabilitySeverity.MEDIUM

    def test_discord_webhook_medium_severity(self):
        """Discord webhooks are medium severity (can be easily rotated)."""
        patterns = [p for p in SECRET_PATTERNS if p.secret_type == SecretType.DISCORD_WEBHOOK]
        for p in patterns:
            assert p.severity == VulnerabilitySeverity.MEDIUM


# ============================================================
# Confidence Score Tests
# ============================================================


class TestConfidenceScores:
    """Tests for pattern confidence scores."""

    def test_high_confidence_patterns(self):
        """Well-defined patterns have high confidence (>= 0.9)."""
        high_confidence_types = [
            SecretType.AWS_ACCESS_KEY,
            SecretType.GITHUB_TOKEN,
            SecretType.GITHUB_PAT,
            SecretType.GITLAB_TOKEN,
            SecretType.SLACK_TOKEN,
            SecretType.STRIPE_KEY,
            SecretType.SENDGRID_KEY,
        ]
        for st in high_confidence_types:
            patterns = [p for p in SECRET_PATTERNS if p.secret_type == st]
            for p in patterns:
                assert p.confidence >= 0.9, f"{st} should have high confidence"

    def test_lower_confidence_generic_patterns(self):
        """Generic patterns have lower confidence."""
        generic_patterns = [
            p
            for p in SECRET_PATTERNS
            if p.secret_type in [SecretType.GENERIC_API_KEY, SecretType.GENERIC_SECRET]
        ]
        for p in generic_patterns:
            assert p.confidence < 0.8, "Generic patterns should have lower confidence"


# ============================================================
# Remediation Guidance Tests
# ============================================================


class TestRemediationGuidance:
    """Tests for remediation guidance in patterns."""

    def test_all_patterns_have_remediation(self):
        """All patterns include remediation guidance."""
        for pattern in SECRET_PATTERNS:
            assert pattern.remediation != "", f"{pattern.secret_type} missing remediation"

    def test_remediation_mentions_rotation(self):
        """Remediation guidance mentions key rotation where applicable."""
        rotation_types = [
            SecretType.AWS_ACCESS_KEY,
            SecretType.AWS_SECRET_KEY,
            SecretType.STRIPE_KEY,
            SecretType.OPENAI_KEY,
        ]
        for st in rotation_types:
            patterns = [p for p in SECRET_PATTERNS if p.secret_type == st]
            for p in patterns:
                remediation_lower = p.remediation.lower()
                assert any(
                    word in remediation_lower
                    for word in ["rotate", "roll", "regenerate", "create new", "delete"]
                ), f"{st} remediation should mention key rotation"
