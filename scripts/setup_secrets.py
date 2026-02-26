#!/usr/bin/env python3
"""
Aragora Secrets Setup Helper.

Interactive script to check, store, and verify secrets in AWS Secrets Manager.
Designed for non-developer users who need to configure API keys.

SECURITY: Secrets are NEVER passed as command-line arguments (they leak to
shell history and `ps` output). All secret input uses hidden prompts or
clipboard reading with automatic clipboard clearing.

Usage:
    python scripts/setup_secrets.py check          # Show what's configured
    python scripts/setup_secrets.py store KEY       # Store a secret (prompts securely)
    python scripts/setup_secrets.py store-all       # Interactive guided setup
    python scripts/setup_secrets.py github          # Show GitHub secrets needed
    python scripts/setup_secrets.py sync-github     # Sync AWS SM secrets to GitHub Actions
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Secure input helpers
# ---------------------------------------------------------------------------


def _read_secret(prompt: str = "Paste value") -> str:
    """Read a secret value securely (hidden input, no echo)."""
    return getpass.getpass(f"    {prompt} (input is hidden): ").strip()


def _read_from_clipboard() -> str | None:
    """Read from system clipboard (macOS pbpaste)."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _clear_clipboard() -> None:
    """Clear the system clipboard (macOS pbcopy)."""
    try:
        subprocess.run(
            ["pbcopy"],
            input=b"",
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def _read_secret_with_clipboard_option(prompt: str = "Paste value") -> str:
    """Read a secret, offering clipboard paste with auto-clear."""
    clipboard = _read_from_clipboard()
    if clipboard and len(clipboard) > 8:
        preview = clipboard[:4] + "..." + clipboard[-4:]
        print(f"    Clipboard contains: {preview}")
        choice = input("    Use clipboard value? (Y/n): ").strip().lower()
        if choice in ("", "y", "yes"):
            _clear_clipboard()
            print("    Clipboard cleared.")
            return clipboard

    return _read_secret(prompt)


# ---------------------------------------------------------------------------
# Secret catalog: every secret Aragora uses, how to get it, and priority
# ---------------------------------------------------------------------------


@dataclass
class SecretInfo:
    """Metadata about a managed secret."""

    name: str
    description: str
    how_to_get: str
    url: str
    priority: str  # "required", "recommended", "optional"
    category: str


SECRET_CATALOG: list[SecretInfo] = [
    # --- AI Provider Keys (required: at least one) ---
    SecretInfo(
        name="ANTHROPIC_API_KEY",
        description="Claude API key (primary AI provider)",
        how_to_get="Sign in, go to API Keys, click Create Key",
        url="https://console.anthropic.com/settings/keys",
        priority="required",
        category="AI Providers",
    ),
    SecretInfo(
        name="OPENAI_API_KEY",
        description="OpenAI API key (GPT models)",
        how_to_get="Sign in, go to API Keys, click Create new secret key",
        url="https://platform.openai.com/api-keys",
        priority="required",
        category="AI Providers",
    ),
    SecretInfo(
        name="OPENROUTER_API_KEY",
        description="OpenRouter API key (failover for all providers)",
        how_to_get="Sign in, go to Keys, click Create Key",
        url="https://openrouter.ai/keys",
        priority="recommended",
        category="AI Providers",
    ),
    SecretInfo(
        name="GEMINI_API_KEY",
        description="Google Gemini API key",
        how_to_get="Sign in, click Create API Key, select project",
        url="https://aistudio.google.com/apikey",
        priority="recommended",
        category="AI Providers",
    ),
    SecretInfo(
        name="XAI_API_KEY",
        description="xAI Grok API key",
        how_to_get="Sign in, go to API Keys, create new key",
        url="https://console.x.ai/",
        priority="recommended",
        category="AI Providers",
    ),
    SecretInfo(
        name="MISTRAL_API_KEY",
        description="Mistral API key (Mistral Large, Codestral)",
        how_to_get="Sign in, go to API Keys, create new key",
        url="https://console.mistral.ai/api-keys/",
        priority="optional",
        category="AI Providers",
    ),
    SecretInfo(
        name="DEEPSEEK_API_KEY",
        description="DeepSeek API key (via OpenRouter recommended instead)",
        how_to_get="Sign in, go to API Keys",
        url="https://platform.deepseek.com/api_keys",
        priority="optional",
        category="AI Providers",
    ),
    # --- Database ---
    SecretInfo(
        name="SUPABASE_URL",
        description="Supabase project URL",
        how_to_get="Go to Project Settings > API, copy Project URL",
        url="https://supabase.com/dashboard/projects",
        priority="required",
        category="Database",
    ),
    SecretInfo(
        name="SUPABASE_KEY",
        description="Supabase anon/public key",
        how_to_get="Go to Project Settings > API, copy anon/public key",
        url="https://supabase.com/dashboard/projects",
        priority="required",
        category="Database",
    ),
    SecretInfo(
        name="SUPABASE_SERVICE_ROLE_KEY",
        description="Supabase service role key (admin access)",
        how_to_get="Go to Project Settings > API, copy service_role key",
        url="https://supabase.com/dashboard/projects",
        priority="required",
        category="Database",
    ),
    SecretInfo(
        name="SUPABASE_DB_PASSWORD",
        description="Supabase database password",
        how_to_get="Go to Project Settings > Database, copy password",
        url="https://supabase.com/dashboard/projects",
        priority="required",
        category="Database",
    ),
    # --- Deployment (Vercel) ---
    SecretInfo(
        name="VERCEL_TOKEN",
        description="Vercel deployment token",
        how_to_get="Go to Settings > Tokens, create new token with 'Full Account' scope",
        url="https://vercel.com/account/tokens",
        priority="required",
        category="Deployment",
    ),
    SecretInfo(
        name="VERCEL_ORG_ID",
        description="Vercel organization/team ID",
        how_to_get="Go to Settings > General, copy 'Team ID' (starts with 'team_')",
        url="https://vercel.com/account",
        priority="required",
        category="Deployment",
    ),
    SecretInfo(
        name="VERCEL_PROJECT_ID",
        description="Vercel project ID for aragora-live",
        how_to_get="Go to your project > Settings > General, copy 'Project ID' (starts with 'prj_')",
        url="https://vercel.com/dashboard",
        priority="required",
        category="Deployment",
    ),
    # --- Authentication ---
    SecretInfo(
        name="JWT_SECRET_KEY",
        description="JWT signing key (auto-generated if not set)",
        how_to_get=(
            "Auto-generated during rotation. "
            'Or: python -c "import secrets; print(secrets.token_hex(64))"'
        ),
        url="",
        priority="required",
        category="Authentication",
    ),
    SecretInfo(
        name="GOOGLE_OAUTH_CLIENT_ID",
        description="Google OAuth client ID for login",
        how_to_get="Create OAuth 2.0 credentials, copy Client ID",
        url="https://console.cloud.google.com/apis/credentials",
        priority="recommended",
        category="Authentication",
    ),
    SecretInfo(
        name="GOOGLE_OAUTH_CLIENT_SECRET",
        description="Google OAuth client secret for login",
        how_to_get="Create OAuth 2.0 credentials, copy Client Secret",
        url="https://console.cloud.google.com/apis/credentials",
        priority="recommended",
        category="Authentication",
    ),
    SecretInfo(
        name="GITHUB_OAUTH_CLIENT_ID",
        description="GitHub OAuth app client ID",
        how_to_get="Create OAuth App, copy Client ID",
        url="https://github.com/settings/developers",
        priority="optional",
        category="Authentication",
    ),
    SecretInfo(
        name="GITHUB_OAUTH_CLIENT_SECRET",
        description="GitHub OAuth app client secret",
        how_to_get="Create OAuth App, copy Client Secret",
        url="https://github.com/settings/developers",
        priority="optional",
        category="Authentication",
    ),
    # --- Billing ---
    SecretInfo(
        name="STRIPE_SECRET_KEY",
        description="Stripe secret key for billing",
        how_to_get="Go to Developers > API Keys, copy Secret key",
        url="https://dashboard.stripe.com/apikeys",
        priority="optional",
        category="Billing",
    ),
    SecretInfo(
        name="STRIPE_WEBHOOK_SECRET",
        description="Stripe webhook signing secret",
        how_to_get="Go to Developers > Webhooks, click endpoint, copy Signing secret",
        url="https://dashboard.stripe.com/webhooks",
        priority="optional",
        category="Billing",
    ),
    # --- Monitoring ---
    SecretInfo(
        name="SENTRY_DSN",
        description="Sentry error tracking DSN",
        how_to_get="Go to Project Settings > Client Keys (DSN), copy DSN",
        url="https://sentry.io/settings/",
        priority="optional",
        category="Monitoring",
    ),
    # --- Other ---
    SecretInfo(
        name="ELEVENLABS_API_KEY",
        description="ElevenLabs TTS API key (voice features)",
        how_to_get="Sign in, go to Profile > API Key",
        url="https://elevenlabs.io/",
        priority="optional",
        category="Other",
    ),
]

# Secrets that only go in GitHub (not AWS SM)
GITHUB_ONLY_SECRETS = [
    {
        "name": "VERCEL_TOKEN",
        "description": "Vercel deployment token (same value as AWS SM)",
        "how": "Copy from Vercel: https://vercel.com/account/tokens",
    },
    {
        "name": "VERCEL_ORG_ID",
        "description": "Vercel org/team ID",
        "how": "Copy from Vercel: Settings > General > Team ID",
    },
    {
        "name": "VERCEL_PROJECT_ID",
        "description": "Vercel project ID",
        "how": "Copy from Vercel: Project > Settings > General > Project ID",
    },
]

GITHUB_VARIABLES = [
    {
        "name": "LIVE_FRONTEND_BASE_URL",
        "description": "Public URL of the frontend",
        "value": "https://aragora.ai",
    },
]


def _check_env(name: str) -> str | None:
    """Check if a secret is set in environment."""
    return os.environ.get(name)


def _check_aws(name: str) -> str | None:
    """Check if a secret is in AWS Secrets Manager."""
    try:
        from aragora.config.secrets import get_secret_manager

        manager = get_secret_manager()
        if not manager.config.use_aws:
            return None
        manager._initialize()
        return manager._cached_secrets.get(name)
    except Exception:
        return None


def cmd_check() -> None:
    """Show status of all secrets."""
    print("=" * 70)
    print("ARAGORA SECRETS STATUS")
    print("=" * 70)

    current_category = ""
    configured = 0
    missing_required = 0
    total = len(SECRET_CATALOG)

    for secret in SECRET_CATALOG:
        if secret.category != current_category:
            current_category = secret.category
            print(f"\n  {current_category}")
            print(f"  {'-' * 40}")

        in_env = _check_env(secret.name)
        in_aws = _check_aws(secret.name)

        if in_aws:
            status = "OK (AWS SM)"
            configured += 1
        elif in_env:
            status = "OK (env var)"
            configured += 1
        else:
            if secret.priority == "required":
                status = "MISSING"
                missing_required += 1
            elif secret.priority == "recommended":
                status = "not set (recommended)"
            else:
                status = "not set"

        priority_tag = {"required": "*", "recommended": "+", "optional": " "}[secret.priority]
        print(f"  {priority_tag} {secret.name:<40} {status}")

    print(f"\n{'=' * 70}")
    print(f"  {configured}/{total} configured")
    if missing_required:
        print(f"  {missing_required} REQUIRED secrets missing")
    print("\n  * = required   + = recommended")
    print(f"{'=' * 70}")


def _store_to_aws(name: str, value: str) -> None:
    """Store a single secret in AWS Secrets Manager."""
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    secret_name = os.environ.get("ARAGORA_SECRET_NAME", "aragora/production")

    client = boto3.client("secretsmanager", region_name=region)

    # Load existing secret bundle
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secrets_bundle: dict[str, str] = json.loads(response["SecretString"])
        print(f"  Loaded existing secret bundle: {secret_name}")
    except client.exceptions.ResourceNotFoundException:
        secrets_bundle = {}
        print(f"  Creating new secret bundle: {secret_name}")
    except Exception as e:
        print(f"ERROR: Could not access AWS Secrets Manager: {e}")
        sys.exit(1)

    # Update the value
    secrets_bundle[name] = value

    # Write back -- always use update_secret since the bundle already exists
    # (or was just created via get_secret_value above)
    try:
        try:
            client.update_secret(
                SecretId=secret_name,
                SecretString=json.dumps(secrets_bundle),
            )
        except client.exceptions.ResourceNotFoundException:
            # Bundle doesn't exist yet -- create it
            client.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secrets_bundle),
            )
        print(f"  Stored {name} in {secret_name}")
    except Exception as e:
        print(f"ERROR: Failed to store secret: {e}")
        sys.exit(1)


def cmd_store(name: str) -> None:
    """Securely store a single secret in AWS Secrets Manager.

    Reads the value from clipboard (with auto-clear) or hidden prompt.
    NEVER accepts the value as a CLI argument.
    """
    # Validate the secret name
    known_names = {s.name for s in SECRET_CATALOG}
    if name not in known_names:
        print(f"  Warning: '{name}' is not in the standard catalog.")
        choice = input("  Store it anyway? (y/N): ").strip().lower()
        if choice not in ("y", "yes"):
            print("  Cancelled.")
            return

    # Show info about this secret if we know it
    info = next((s for s in SECRET_CATALOG if s.name == name), None)
    if info:
        print(f"\n  {name}: {info.description}")
        if info.url:
            print(f"  Get it here: {info.url}")
        print(f"  How: {info.how_to_get}")
        print()

    value = _read_secret_with_clipboard_option()
    if not value:
        print("  No value provided. Cancelled.")
        return

    _store_to_aws(name, value)
    print(f"  Done. {name} is now in AWS Secrets Manager.")


def cmd_store_all() -> None:
    """Interactive guided setup for all secrets."""
    print("=" * 70)
    print("ARAGORA SECRETS SETUP (Interactive)")
    print("=" * 70)
    print()
    print("This will walk you through setting up each secret.")
    print("For each one, copy the value to your clipboard then press Enter,")
    print("or type the value (hidden). Press Enter with nothing to skip.")
    print()

    stored = 0
    skipped = 0

    current_category = ""
    for secret in SECRET_CATALOG:
        if secret.category != current_category:
            current_category = secret.category
            print(f"\n--- {current_category} ---\n")

        # Check if already configured
        in_aws = _check_aws(secret.name)
        in_env = _check_env(secret.name)
        if in_aws:
            print(f"  {secret.name}: already in AWS SM (skipping)")
            continue

        priority_label = f"[{secret.priority.upper()}]"
        print(f"  {priority_label} {secret.name}")
        print(f"    {secret.description}")
        if secret.url:
            print(f"    Get it here: {secret.url}")
        print(f"    How: {secret.how_to_get}")
        if in_env:
            print("    (currently set in env var)")

        value = _read_secret_with_clipboard_option()
        if value:
            _store_to_aws(secret.name, value)
            stored += 1
        else:
            skipped += 1
            print("    Skipped.")

    print(f"\n{'=' * 70}")
    print(f"  Stored: {stored}   Skipped: {skipped}")
    print(f"{'=' * 70}")


def cmd_github() -> None:
    """Show GitHub-specific secrets and variables needed."""
    print("=" * 70)
    print("GITHUB SECRETS & VARIABLES NEEDED")
    print("=" * 70)
    print()
    print("These must be added at: https://github.com/an0mium/aragora/settings/secrets/actions")
    print()
    print("SECRETS (Settings > Secrets and variables > Actions > New repository secret):")
    print()
    for s in GITHUB_ONLY_SECRETS:
        print(f"  Name:  {s['name']}")
        print(f"  Info:  {s['description']}")
        print(f"  How:   {s['how']}")
        print()
    print("VARIABLES (Settings > Secrets and variables > Actions > Variables tab > New variable):")
    print()
    for v in GITHUB_VARIABLES:
        print(f"  Name:   {v['name']}")
        print(f"  Value:  {v['value']}")
        print(f"  Info:   {v['description']}")
        print()


# ---------------------------------------------------------------------------
# GitHub secrets sync
# ---------------------------------------------------------------------------

# Secrets to sync from AWS SM → GitHub Actions secrets
GITHUB_SYNC_SECRETS: list[str] = [
    # AI Providers
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROK_API_KEY",
    # Database
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_DB_PASSWORD",
    # Deployment
    "VERCEL_TOKEN",
    "VERCEL_ORG_ID",
    "VERCEL_PROJECT_ID",
    # Auth
    "JWT_SECRET_KEY",
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GITHUB_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_CLIENT_SECRET",
    # Billing / Monitoring
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "SENTRY_DSN",
    # Other
    "ELEVENLABS_API_KEY",
]

# Variables (non-secret config) to set in GitHub Actions
GITHUB_SYNC_VARIABLES: dict[str, str] = {
    "LIVE_FRONTEND_BASE_URL": "https://aragora.ai",
    "AWS_CI_ENABLED": "true",
}

GITHUB_REPO = "an0mium/aragora"

# GitHub doesn't allow secret names starting with "GITHUB_".
# Remap to GH_ prefix; workflows must reference ${{ secrets.GH_OAUTH_CLIENT_ID }}.
GITHUB_SECRET_RENAMES: dict[str, str] = {
    "GITHUB_OAUTH_CLIENT_ID": "GH_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_CLIENT_SECRET": "GH_OAUTH_CLIENT_SECRET",
}


def _gh_available() -> bool:
    """Check if gh CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _gh_set_secret(name: str, value: str) -> bool:
    """Set a GitHub Actions secret via gh CLI (value passed via stdin)."""
    try:
        result = subprocess.run(
            ["gh", "secret", "set", name, "--repo", GITHUB_REPO],
            input=value,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _gh_set_variable(name: str, value: str) -> bool:
    """Set a GitHub Actions variable via gh CLI."""
    # Try update first, then create
    result = subprocess.run(
        ["gh", "variable", "set", name, "--repo", GITHUB_REPO, "--body", value],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _load_aws_secrets_bundle() -> dict[str, str]:
    """Load the full secrets bundle from AWS Secrets Manager."""
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    secret_name = os.environ.get("ARAGORA_SECRET_NAME", "aragora/production")

    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception as e:
        print(f"ERROR: Could not load secrets from AWS SM: {e}")
        sys.exit(1)


def cmd_sync_github() -> None:
    """Sync secrets from AWS Secrets Manager to GitHub Actions."""
    print("=" * 70)
    print("SYNC: AWS Secrets Manager → GitHub Actions")
    print("=" * 70)

    # Check gh CLI
    if not _gh_available():
        print("\nERROR: gh CLI is not installed or not authenticated.")
        print("  Install: https://cli.github.com/")
        print("  Auth:    gh auth login")
        sys.exit(1)

    print(f"\n  Repository: {GITHUB_REPO}")

    # Load all secrets from AWS SM
    print("  Loading secrets from AWS Secrets Manager...")
    bundle = _load_aws_secrets_bundle()
    print(f"  Found {len(bundle)} secrets in AWS SM\n")

    # Sync secrets
    synced = 0
    skipped = 0
    missing = 0

    print("  SECRETS:")
    for name in GITHUB_SYNC_SECRETS:
        value = bundle.get(name)
        if not value:
            # Fall back to env var
            value = os.environ.get(name)

        if not value:
            print(f"    SKIP  {name:<40} (not in AWS SM or env)")
            missing += 1
            continue

        # Remap names that GitHub reserves (e.g. GITHUB_* prefix)
        gh_name = GITHUB_SECRET_RENAMES.get(name, name)
        ok = _gh_set_secret(gh_name, value)
        if ok:
            suffix = f" (as {gh_name})" if gh_name != name else ""
            print(f"    SET   {name}{suffix}")
            synced += 1
        else:
            print(f"    FAIL  {name}")
            skipped += 1

    # Sync variables
    print("\n  VARIABLES:")
    for name, value in GITHUB_SYNC_VARIABLES.items():
        ok = _gh_set_variable(name, value)
        if ok:
            print(f"    SET   {name} = {value}")
        else:
            print(f"    FAIL  {name}")

    print(f"\n{'=' * 70}")
    print(f"  Synced: {synced}   Skipped: {skipped}   Missing: {missing}")
    print(f"{'=' * 70}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "check":
        cmd_check()
    elif command == "store":
        if len(sys.argv) == 4:
            # Reject: value passed as CLI argument is a security risk
            print("ERROR: Do NOT pass secret values as command-line arguments!")
            print("       They leak to shell history (~/.zsh_history) and `ps` output.")
            print()
            print(f"  Safe usage:  python scripts/setup_secrets.py store {sys.argv[2]}")
            print("  (You'll be prompted to paste the value securely)")
            sys.exit(1)
        elif len(sys.argv) == 3:
            cmd_store(sys.argv[2])
        else:
            print("Usage: python scripts/setup_secrets.py store SECRET_NAME")
            sys.exit(1)
    elif command == "store-all":
        cmd_store_all()
    elif command == "github":
        cmd_github()
    elif command == "sync-github":
        cmd_sync_github()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
