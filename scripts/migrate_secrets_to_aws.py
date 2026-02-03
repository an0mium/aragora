#!/usr/bin/env python3
"""
Secrets Migration Tool: .env to AWS Secrets Manager.

DEPRECATED: This script is superseded by the unified secrets_manager.py.
Use instead:
    python scripts/secrets_manager.py migrate -e staging      # Migrate to staging
    python scripts/secrets_manager.py migrate -e production   # Migrate to production

This script is kept for backward compatibility.

---

This script automates the migration from .env files to AWS Secrets Manager.
It handles what can be automated and provides clear instructions for manual steps.

What this script automates:
- Reading existing .env values
- Generating new JWT/encryption secrets
- Creating/updating AWS Secrets Manager secret
- Validating API keys still work
- Verifying the migration succeeded

What requires human action:
- Rotating API keys at provider dashboards (Anthropic, OpenAI, etc.)
- OAuth client secrets (Google, GitHub)

Usage:
    # Dry run - see what would happen
    python scripts/migrate_secrets_to_aws.py --dry-run

    # Migrate to staging
    python scripts/migrate_secrets_to_aws.py --environment staging

    # Migrate to production
    python scripts/migrate_secrets_to_aws.py --environment production

    # Migrate and rotate internal secrets (JWT, encryption keys)
    python scripts/migrate_secrets_to_aws.py --environment staging --rotate-internal

    # After manually rotating API keys, validate them
    python scripts/migrate_secrets_to_aws.py --validate-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


# =============================================================================
# Secret Definitions
# =============================================================================


@dataclass
class SecretDefinition:
    """Definition of a secret to migrate."""

    env_name: str
    aws_name: str
    category: str  # "internal", "api_key", "oauth", "database", "billing"
    can_auto_generate: bool = False
    can_auto_rotate: bool = False  # Provider has API for rotation
    provider: str | None = None
    rotation_url: str | None = None
    description: str = ""
    required: bool = False
    min_length: int = 0


# Secrets that we manage
SECRETS: list[SecretDefinition] = [
    # === INTERNAL SECRETS (can be auto-generated) ===
    SecretDefinition(
        env_name="ARAGORA_JWT_SECRET",
        aws_name="ARAGORA_JWT_SECRET",
        category="internal",
        can_auto_generate=True,
        description="JWT signing secret (64 chars)",
        required=True,
        min_length=32,
    ),
    SecretDefinition(
        env_name="JWT_SECRET_KEY",
        aws_name="JWT_SECRET_KEY",
        category="internal",
        can_auto_generate=True,
        description="JWT secret key alias",
        min_length=32,
    ),
    SecretDefinition(
        env_name="JWT_REFRESH_SECRET",
        aws_name="JWT_REFRESH_SECRET",
        category="internal",
        can_auto_generate=True,
        description="JWT refresh token secret",
        min_length=32,
    ),
    SecretDefinition(
        env_name="ARAGORA_ENCRYPTION_KEY",
        aws_name="ARAGORA_ENCRYPTION_KEY",
        category="internal",
        can_auto_generate=True,
        description="Data encryption key (32 bytes base64)",
        min_length=32,
    ),
    SecretDefinition(
        env_name="ARAGORA_AUDIT_SIGNING_KEY",
        aws_name="ARAGORA_AUDIT_SIGNING_KEY",
        category="internal",
        can_auto_generate=True,
        description="Audit log signing key",
        min_length=32,
    ),
    # === API KEYS (require manual rotation) ===
    SecretDefinition(
        env_name="ANTHROPIC_API_KEY",
        aws_name="ANTHROPIC_API_KEY",
        category="api_key",
        provider="Anthropic",
        rotation_url="https://console.anthropic.com/settings/keys",
        description="Claude API key",
        required=True,
    ),
    SecretDefinition(
        env_name="OPENAI_API_KEY",
        aws_name="OPENAI_API_KEY",
        category="api_key",
        provider="OpenAI",
        rotation_url="https://platform.openai.com/api-keys",
        description="GPT API key",
    ),
    SecretDefinition(
        env_name="OPENROUTER_API_KEY",
        aws_name="OPENROUTER_API_KEY",
        category="api_key",
        provider="OpenRouter",
        rotation_url="https://openrouter.ai/keys",
        description="OpenRouter fallback key",
    ),
    SecretDefinition(
        env_name="MISTRAL_API_KEY",
        aws_name="MISTRAL_API_KEY",
        category="api_key",
        provider="Mistral",
        rotation_url="https://console.mistral.ai/api-keys",
        description="Mistral API key",
    ),
    SecretDefinition(
        env_name="GEMINI_API_KEY",
        aws_name="GEMINI_API_KEY",
        category="api_key",
        provider="Google",
        rotation_url="https://aistudio.google.com/apikey",
        description="Gemini API key",
    ),
    SecretDefinition(
        env_name="XAI_API_KEY",
        aws_name="XAI_API_KEY",
        category="api_key",
        provider="xAI",
        rotation_url="https://console.x.ai/team/api-keys",
        description="Grok API key",
    ),
    SecretDefinition(
        env_name="DEEPSEEK_API_KEY",
        aws_name="DEEPSEEK_API_KEY",
        category="api_key",
        provider="DeepSeek",
        rotation_url="https://platform.deepseek.com/api_keys",
        description="DeepSeek API key",
    ),
    SecretDefinition(
        env_name="ELEVENLABS_API_KEY",
        aws_name="ELEVENLABS_API_KEY",
        category="api_key",
        provider="ElevenLabs",
        rotation_url="https://elevenlabs.io/app/settings/api-keys",
        description="ElevenLabs TTS key",
    ),
    # === OAUTH SECRETS (require manual rotation) ===
    SecretDefinition(
        env_name="GOOGLE_OAUTH_CLIENT_ID",
        aws_name="GOOGLE_OAUTH_CLIENT_ID",
        category="oauth",
        provider="Google",
        rotation_url="https://console.cloud.google.com/apis/credentials",
        description="Google OAuth client ID",
    ),
    SecretDefinition(
        env_name="GOOGLE_OAUTH_CLIENT_SECRET",
        aws_name="GOOGLE_OAUTH_CLIENT_SECRET",
        category="oauth",
        provider="Google",
        rotation_url="https://console.cloud.google.com/apis/credentials",
        description="Google OAuth client secret",
    ),
    SecretDefinition(
        env_name="GITHUB_OAUTH_CLIENT_ID",
        aws_name="GITHUB_OAUTH_CLIENT_ID",
        category="oauth",
        provider="GitHub",
        rotation_url="https://github.com/settings/developers",
        description="GitHub OAuth client ID",
    ),
    SecretDefinition(
        env_name="GITHUB_OAUTH_CLIENT_SECRET",
        aws_name="GITHUB_OAUTH_CLIENT_SECRET",
        category="oauth",
        provider="GitHub",
        rotation_url="https://github.com/settings/developers",
        description="GitHub OAuth client secret",
    ),
    # === DATABASE SECRETS ===
    SecretDefinition(
        env_name="DATABASE_URL",
        aws_name="DATABASE_URL",
        category="database",
        description="PostgreSQL connection string",
    ),
    SecretDefinition(
        env_name="SUPABASE_URL",
        aws_name="SUPABASE_URL",
        category="database",
        provider="Supabase",
        rotation_url="https://supabase.com/dashboard/project/_/settings/api",
        description="Supabase project URL",
    ),
    SecretDefinition(
        env_name="SUPABASE_KEY",
        aws_name="SUPABASE_KEY",
        category="database",
        provider="Supabase",
        rotation_url="https://supabase.com/dashboard/project/_/settings/api",
        description="Supabase anon key",
    ),
    SecretDefinition(
        env_name="SUPABASE_SERVICE_ROLE_KEY",
        aws_name="SUPABASE_SERVICE_ROLE_KEY",
        category="database",
        provider="Supabase",
        rotation_url="https://supabase.com/dashboard/project/_/settings/api",
        description="Supabase service role key",
    ),
    SecretDefinition(
        env_name="REDIS_URL",
        aws_name="REDIS_URL",
        category="database",
        description="Redis connection string",
    ),
    # === BILLING SECRETS ===
    SecretDefinition(
        env_name="STRIPE_SECRET_KEY",
        aws_name="STRIPE_SECRET_KEY",
        category="billing",
        provider="Stripe",
        rotation_url="https://dashboard.stripe.com/apikeys",
        description="Stripe secret key",
    ),
    SecretDefinition(
        env_name="STRIPE_WEBHOOK_SECRET",
        aws_name="STRIPE_WEBHOOK_SECRET",
        category="billing",
        provider="Stripe",
        rotation_url="https://dashboard.stripe.com/webhooks",
        description="Stripe webhook signing secret",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================


def generate_secret(length: int = 48) -> str:
    """Generate a cryptographically secure secret."""
    return secrets.token_urlsafe(length)


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load secrets from .env file."""
    if not env_path.exists():
        return {}

    secrets_dict = {}
    content = env_path.read_text()

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            key, value = match.groups()
            # Remove quotes if present
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            secrets_dict[key] = value

    return secrets_dict


def get_aws_client(region: str):
    """Get AWS Secrets Manager client."""
    try:
        import boto3

        return boto3.client("secretsmanager", region_name=region)
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        sys.exit(1)


def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured."""
    try:
        import boto3

        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        logger.info(f"AWS Account: {identity['Account']}")
        logger.info(f"AWS User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        logger.error(f"AWS credentials not configured: {e}")
        return False


# =============================================================================
# Validation Functions
# =============================================================================


async def validate_anthropic_key(key: str) -> bool:
    """Validate Anthropic API key."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10.0,
            )
            # 401 = invalid, 400 = valid but bad request
            return response.status_code != 401
    except Exception as e:
        logger.debug(f"Anthropic validation error: {e}")
        return False


async def validate_openai_key(key: str) -> bool:
    """Validate OpenAI API key."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10.0,
            )
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"OpenAI validation error: {e}")
        return False


async def validate_openrouter_key(key: str) -> bool:
    """Validate OpenRouter API key."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10.0,
            )
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"OpenRouter validation error: {e}")
        return False


VALIDATORS = {
    "ANTHROPIC_API_KEY": validate_anthropic_key,
    "OPENAI_API_KEY": validate_openai_key,
    "OPENROUTER_API_KEY": validate_openrouter_key,
}


async def validate_secrets(secrets_dict: dict[str, str]) -> dict[str, bool]:
    """Validate all secrets that have validators."""
    results = {}

    for key, validator in VALIDATORS.items():
        if key in secrets_dict and secrets_dict[key]:
            logger.info(f"Validating {key}...")
            try:
                results[key] = await validator(secrets_dict[key])
                status = f"{GREEN}VALID{RESET}" if results[key] else f"{RED}INVALID{RESET}"
                logger.info(f"  {key}: {status}")
            except Exception as e:
                results[key] = False
                logger.error(f"  {key}: {RED}ERROR{RESET} - {e}")

    return results


# =============================================================================
# Migration Functions
# =============================================================================


@dataclass
class MigrationPlan:
    """Plan for migrating secrets."""

    to_migrate: dict[str, str] = field(default_factory=dict)
    to_generate: list[str] = field(default_factory=list)
    to_rotate_manually: list[SecretDefinition] = field(default_factory=list)
    missing_required: list[SecretDefinition] = field(default_factory=list)
    aws_secret_name: str = ""
    aws_region: str = ""


def create_migration_plan(
    env_secrets: dict[str, str],
    environment: str,
    rotate_internal: bool = False,
) -> MigrationPlan:
    """Create a migration plan."""
    plan = MigrationPlan(
        aws_secret_name=f"aragora/{environment}",
        aws_region=os.environ.get("AWS_REGION", "us-east-2"),
    )

    for secret_def in SECRETS:
        env_value = env_secrets.get(secret_def.env_name)

        # Check if it's a required secret that's missing
        if secret_def.required and not env_value:
            if secret_def.can_auto_generate:
                plan.to_generate.append(secret_def.env_name)
            else:
                plan.missing_required.append(secret_def)
            continue

        # Skip if no value
        if not env_value:
            continue

        # Auto-generate internal secrets if rotating
        if secret_def.can_auto_generate and rotate_internal:
            plan.to_generate.append(secret_def.env_name)
        else:
            plan.to_migrate[secret_def.aws_name] = env_value

        # Track secrets that need manual rotation
        if secret_def.category in ("api_key", "oauth", "billing") and secret_def.rotation_url:
            plan.to_rotate_manually.append(secret_def)

    return plan


def execute_migration(plan: MigrationPlan, dry_run: bool = False) -> bool:
    """Execute the migration plan."""

    # Build the secret value
    secret_value = dict(plan.to_migrate)

    # Generate new internal secrets
    for secret_name in plan.to_generate:
        new_value = generate_secret(48)
        secret_value[secret_name] = new_value
        logger.info(f"{GREEN}Generated{RESET} new value for {secret_name}")

    if dry_run:
        logger.info(f"\n{YELLOW}DRY RUN{RESET} - Would create/update secret:")
        logger.info(f"  Name: {plan.aws_secret_name}")
        logger.info(f"  Region: {plan.aws_region}")
        logger.info(f"  Keys: {list(secret_value.keys())}")
        return True

    # Create or update AWS secret
    client = get_aws_client(plan.aws_region)
    secret_string = json.dumps(secret_value)

    try:
        # Try to update existing secret
        client.put_secret_value(
            SecretId=plan.aws_secret_name,
            SecretString=secret_string,
        )
        logger.info(f"{GREEN}Updated{RESET} existing secret: {plan.aws_secret_name}")
    except client.exceptions.ResourceNotFoundException:
        # Create new secret
        client.create_secret(
            Name=plan.aws_secret_name,
            SecretString=secret_string,
            Description=f"Aragora {plan.aws_secret_name.split('/')[-1]} secrets",
            Tags=[
                {"Key": "Application", "Value": "aragora"},
                {"Key": "Environment", "Value": plan.aws_secret_name.split("/")[-1]},
                {"Key": "ManagedBy", "Value": "migrate_secrets_to_aws.py"},
            ],
        )
        logger.info(f"{GREEN}Created{RESET} new secret: {plan.aws_secret_name}")

    return True


def print_manual_rotation_instructions(secrets_to_rotate: list[SecretDefinition]) -> None:
    """Print instructions for manual rotation."""
    if not secrets_to_rotate:
        return

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}MANUAL ROTATION REQUIRED{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print("""
The following secrets require manual rotation at provider dashboards.
After rotating, run this script again with --update-key to add the new keys.
""")

    # Group by provider
    by_provider: dict[str, list[SecretDefinition]] = {}
    for secret in secrets_to_rotate:
        provider = secret.provider or "Unknown"
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(secret)

    for provider, provider_secrets in by_provider.items():
        url = provider_secrets[0].rotation_url
        print(f"\n{BLUE}{BOLD}{provider}{RESET}")
        print(f"  URL: {url}")
        print("  Secrets:")
        for s in provider_secrets:
            print(f"    - {s.env_name}: {s.description}")

    print(f"""
{BOLD}After rotating keys:{RESET}

1. Update your local .env file with new keys
2. Run: python scripts/migrate_secrets_to_aws.py --environment <env> --update-from-env
3. Or update individual keys:
   python scripts/migrate_secrets_to_aws.py --update-key ANTHROPIC_API_KEY=sk-ant-...
""")


def verify_aws_secret(secret_name: str, region: str) -> dict[str, str] | None:
    """Verify AWS secret exists and return its keys."""
    try:
        client = get_aws_client(region)
        response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response["SecretString"])
        return secret_dict
    except Exception as e:
        logger.error(f"Could not read AWS secret: {e}")
        return None


# =============================================================================
# CLI Commands
# =============================================================================


async def cmd_migrate(args: argparse.Namespace) -> int:
    """Migrate secrets from .env to AWS."""

    # Check AWS credentials
    if not args.dry_run and not check_aws_credentials():
        return 1

    # Load .env file
    env_path = Path(args.env_file)
    env_secrets = load_env_file(env_path)

    if not env_secrets:
        logger.warning(f"No secrets found in {env_path}")
        if not args.dry_run:
            return 1

    logger.info(f"Loaded {len(env_secrets)} secrets from {env_path}")

    # Create migration plan
    plan = create_migration_plan(
        env_secrets,
        args.environment,
        rotate_internal=args.rotate_internal,
    )

    # Report missing required secrets
    if plan.missing_required:
        logger.error(f"\n{RED}Missing required secrets:{RESET}")
        for s in plan.missing_required:
            logger.error(f"  - {s.env_name}: {s.description}")
        if not args.dry_run:
            return 1

    # Execute migration
    print(f"\n{BOLD}Migration Plan:{RESET}")
    print(f"  Environment: {args.environment}")
    print(f"  AWS Secret: {plan.aws_secret_name}")
    print(f"  Region: {plan.aws_region}")
    print(f"  Secrets to migrate: {len(plan.to_migrate)}")
    print(f"  Secrets to generate: {len(plan.to_generate)}")
    print(f"  Secrets needing manual rotation: {len(plan.to_rotate_manually)}")

    if not execute_migration(plan, dry_run=args.dry_run):
        return 1

    # Print manual rotation instructions
    print_manual_rotation_instructions(plan.to_rotate_manually)

    # Verify if not dry run
    if not args.dry_run:
        print(f"\n{BOLD}Verification:{RESET}")
        aws_secrets = verify_aws_secret(plan.aws_secret_name, plan.aws_region)
        if aws_secrets:
            print(f"  {GREEN}SUCCESS{RESET} - Secret contains {len(aws_secrets)} keys")

            # Validate API keys
            if args.validate:
                print(f"\n{BOLD}Validating API keys...{RESET}")
                await validate_secrets(aws_secrets)

    return 0


async def cmd_validate(args: argparse.Namespace) -> int:
    """Validate secrets in AWS."""

    if not check_aws_credentials():
        return 1

    secret_name = f"aragora/{args.environment}"
    region = os.environ.get("AWS_REGION", "us-east-2")

    aws_secrets = verify_aws_secret(secret_name, region)
    if not aws_secrets:
        logger.error(f"Could not read secret: {secret_name}")
        return 1

    print(f"\n{BOLD}Validating secrets in {secret_name}...{RESET}\n")
    results = await validate_secrets(aws_secrets)

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n{RED}Failed validations:{RESET} {', '.join(failed)}")
        return 1

    print(f"\n{GREEN}All validations passed!{RESET}")
    return 0


async def cmd_update_key(args: argparse.Namespace) -> int:
    """Update a single key in AWS Secrets Manager."""

    if not check_aws_credentials():
        return 1

    # Parse key=value
    if "=" not in args.key_value:
        logger.error("Format: KEY_NAME=value")
        return 1

    key_name, key_value = args.key_value.split("=", 1)

    secret_name = f"aragora/{args.environment}"
    region = os.environ.get("AWS_REGION", "us-east-2")

    # Get existing secret
    aws_secrets = verify_aws_secret(secret_name, region)
    if not aws_secrets:
        logger.error(f"Secret {secret_name} does not exist")
        return 1

    # Update
    aws_secrets[key_name] = key_value

    client = get_aws_client(region)
    client.put_secret_value(
        SecretId=secret_name,
        SecretString=json.dumps(aws_secrets),
    )

    logger.info(f"{GREEN}Updated{RESET} {key_name} in {secret_name}")

    # Validate if it's an API key
    if key_name in VALIDATORS:
        print(f"\n{BOLD}Validating {key_name}...{RESET}")
        is_valid = await VALIDATORS[key_name](key_value)
        if is_valid:
            print(f"  {GREEN}VALID{RESET}")
        else:
            print(f"  {RED}INVALID{RESET} - Check the key and try again")
            return 1

    return 0


async def cmd_status(args: argparse.Namespace) -> int:
    """Show current secrets status."""

    if not check_aws_credentials():
        return 1

    secret_name = f"aragora/{args.environment}"
    region = os.environ.get("AWS_REGION", "us-east-2")

    print(f"\n{BOLD}AWS Secrets Manager Status{RESET}")
    print(f"  Secret: {secret_name}")
    print(f"  Region: {region}")

    aws_secrets = verify_aws_secret(secret_name, region)

    if not aws_secrets:
        print(f"\n  {RED}Secret does not exist{RESET}")
        print(f"\n  Run: python scripts/migrate_secrets_to_aws.py --environment {args.environment}")
        return 1

    print(f"\n{BOLD}Configured Secrets:{RESET}")

    # Group by category
    for category in ["internal", "api_key", "oauth", "database", "billing"]:
        category_secrets = [s for s in SECRETS if s.category == category]
        if not category_secrets:
            continue

        print(f"\n  {BLUE}{category.upper()}{RESET}")
        for secret_def in category_secrets:
            if secret_def.aws_name in aws_secrets:
                value = aws_secrets[secret_def.aws_name]
                # Mask the value
                if len(value) > 8:
                    masked = value[:4] + "..." + value[-4:]
                else:
                    masked = "****"
                print(f"    {GREEN}[SET]{RESET} {secret_def.aws_name}: {masked}")
            else:
                if secret_def.required:
                    print(f"    {RED}[MISSING]{RESET} {secret_def.aws_name} (required)")
                else:
                    print(f"    {YELLOW}[NOT SET]{RESET} {secret_def.aws_name}")

    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate secrets from .env to AWS Secrets Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current status
  python scripts/migrate_secrets_to_aws.py status --environment staging

  # Dry run migration
  python scripts/migrate_secrets_to_aws.py migrate --environment staging --dry-run

  # Full migration with new internal secrets
  python scripts/migrate_secrets_to_aws.py migrate --environment staging --rotate-internal

  # Validate API keys in AWS
  python scripts/migrate_secrets_to_aws.py validate --environment staging

  # Update a single key after manual rotation
  python scripts/migrate_secrets_to_aws.py update-key --environment staging ANTHROPIC_API_KEY=sk-ant-...
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate secrets to AWS")
    migrate_parser.add_argument(
        "--environment",
        "-e",
        required=True,
        choices=["staging", "production"],
        help="Target environment",
    )
    migrate_parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    migrate_parser.add_argument(
        "--rotate-internal",
        action="store_true",
        help="Generate new JWT/encryption secrets",
    )
    migrate_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate API keys after migration",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate secrets in AWS")
    validate_parser.add_argument(
        "--environment",
        "-e",
        required=True,
        choices=["staging", "production"],
        help="Environment to validate",
    )

    # update-key command
    update_parser = subparsers.add_parser("update-key", help="Update a single key")
    update_parser.add_argument(
        "--environment",
        "-e",
        required=True,
        choices=["staging", "production"],
        help="Target environment",
    )
    update_parser.add_argument(
        "key_value",
        help="Key=Value to update (e.g., ANTHROPIC_API_KEY=sk-ant-...)",
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Show secrets status")
    status_parser.add_argument(
        "--environment",
        "-e",
        required=True,
        choices=["staging", "production"],
        help="Environment to check",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run appropriate command
    if args.command == "migrate":
        return asyncio.run(cmd_migrate(args))
    elif args.command == "validate":
        return asyncio.run(cmd_validate(args))
    elif args.command == "update-key":
        return asyncio.run(cmd_update_key(args))
    elif args.command == "status":
        return asyncio.run(cmd_status(args))

    return 0


if __name__ == "__main__":
    sys.exit(main())
