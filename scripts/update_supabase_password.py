#!/usr/bin/env python3
"""
Supabase Password Update Script

Updates Supabase database password across all backends:
- AWS Secrets Manager (us-east-1, us-east-2)
- GitHub Secrets
- Local .env

Usage:
    python scripts/update_supabase_password.py              # Interactive prompt
    python scripts/update_supabase_password.py --password "new-password"  # Direct
    python scripts/update_supabase_password.py --dry-run    # Preview only
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import sys
from pathlib import Path

# Reuse backends from rotate_keys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.rotate_keys import (
    AWSSecretsBackend,
    GitHubSecretsBackend,
    LocalEnvBackend,
)

# Project reference for Supabase
PROJECT_REF = "etwxrexpyvqykqqjaxsz"

# DSN patterns for password replacement
# Direct connection: postgres://postgres:password@db.ref.supabase.co:5432/postgres
# Pooler connection: postgres://postgres.ref:password@host:port/postgres
DSN_PASSWORD_PATTERN = re.compile(r"(postgres://postgres(?:\.[^:]+)?:)([^@]+)(@)")


def update_dsn_password(dsn: str, new_password: str) -> str:
    """Replace password in a Postgres DSN."""
    return DSN_PASSWORD_PATTERN.sub(rf"\g<1>{new_password}\g<3>", dsn)


def main():
    parser = argparse.ArgumentParser(
        description="Update Supabase database password across all backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--password", "-p", help="New password (prompts if not provided)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SUPABASE PASSWORD UPDATE")
    print("=" * 60)

    # Get the new password
    if args.password:
        new_password = args.password
    else:
        new_password = getpass.getpass("\nEnter new Supabase database password: ")
        if not new_password:
            print("No password provided. Exiting.")
            sys.exit(1)

    print(f"\nPassword: {'*' * len(new_password)}")

    # Initialize backends
    print("\n[1/3] Initializing backends...")
    backends: dict[str, object] = {}

    # AWS us-east-1
    try:
        be = AWSSecretsBackend("us-east-1")
        be.client.list_secrets(MaxResults=1)
        backends["aws-east-1"] = be
        print("  ✓ AWS Secrets Manager (us-east-1)")
    except Exception as e:
        print(f"  ✗ AWS us-east-1: {str(e)[:50]}")

    # AWS us-east-2
    try:
        be = AWSSecretsBackend("us-east-2")
        be.client.list_secrets(MaxResults=1)
        backends["aws-east-2"] = be
        print("  ✓ AWS Secrets Manager (us-east-2)")
    except Exception as e:
        print(f"  ✗ AWS us-east-2: {str(e)[:50]}")

    # GitHub
    try:
        gh = GitHubSecretsBackend()
        if gh.repo:
            backends["github"] = gh
            print(f"  ✓ GitHub Secrets ({gh.repo})")
        else:
            print("  ✗ GitHub: could not detect repo")
    except Exception as e:
        print(f"  ✗ GitHub: {e}")

    # Local .env
    backends["local"] = LocalEnvBackend()
    print("  ✓ Local .env")

    if len(backends) < 2:
        print("\n  ⚠ Warning: Some backends unavailable")

    # Secrets to update
    secrets_to_update = [
        "SUPABASE_DB_PASSWORD",
        "SUPABASE_POSTGRES_DSN",
        "ARAGORA_POSTGRES_DSN",
    ]

    # Update each backend
    print("\n[2/3] Updating secrets...")

    for backend_name, backend in backends.items():
        print(f"\n  {backend_name}:")

        for secret_name in secrets_to_update:
            if args.dry_run:
                print(f"    would update {secret_name}")
                continue

            try:
                if secret_name == "SUPABASE_DB_PASSWORD":
                    # Direct password value
                    new_value = new_password
                else:
                    # DSN - use direct connection format (not pooler)
                    # Direct: postgres://postgres:pass@db.REF.supabase.co:5432/postgres
                    if backend_name == "github":
                        # Can't read GitHub secrets, construct direct connection DSN
                        new_value = f"postgres://postgres:{new_password}@db.{PROJECT_REF}.supabase.co:5432/postgres"
                    else:
                        current = backend.get_secret(secret_name)
                        if current and "postgres://" in current:
                            new_value = update_dsn_password(current, new_password)
                        else:
                            # Construct direct connection DSN
                            new_value = f"postgres://postgres:{new_password}@db.{PROJECT_REF}.supabase.co:5432/postgres"

                # Set the secret
                success = backend.set_secret(secret_name, new_value)
                symbol = "✓" if success else "✗"
                print(f"    {symbol} {secret_name}")

            except Exception as e:
                print(f"    ✗ {secret_name}: {str(e)[:40]}")

    # Final summary
    print("\n[3/3] Summary")
    if args.dry_run:
        print("  (dry-run mode - no changes made)")
    else:
        print("  ✓ Password updated across all available backends")
        print("\n  Next steps:")
        print('  1. Verify connection: python -c "import asyncpg; ..."')
        print("  2. Re-run tests: pytest tests/test_supabase.py")

    print()


if __name__ == "__main__":
    main()
