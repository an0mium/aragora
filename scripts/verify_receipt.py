#!/usr/bin/env python3
"""
Offline Receipt Signature Verification Tool.

Verifies cryptographic signatures on decision receipts without requiring
server access. Supports HMAC-SHA256, RSA-SHA256, and Ed25519 algorithms.

Usage:
    python verify_receipt.py receipt.json --key <signing_key>
    python verify_receipt.py receipt.json --key-file key.pem
    python verify_receipt.py receipt.json --key-env ARAGORA_RECEIPT_SIGNING_KEY

Examples:
    # Verify with HMAC key (hex-encoded)
    python verify_receipt.py receipt.json --key abc123def456...

    # Verify with RSA/Ed25519 public key file
    python verify_receipt.py receipt.json --key-file public_key.pem

    # Verify using environment variable
    export ARAGORA_RECEIPT_SIGNING_KEY=abc123...
    python verify_receipt.py receipt.json --key-env ARAGORA_RECEIPT_SIGNING_KEY

Exit codes:
    0 - Signature valid
    1 - Signature invalid
    2 - Error (missing file, invalid format, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_receipt(path: str) -> dict:
    """Load and parse receipt JSON file."""
    with open(path) as f:
        return json.load(f)


def get_key(args: argparse.Namespace) -> str | bytes | None:
    """Extract signing key from arguments."""
    if args.key:
        return args.key
    if args.key_env:
        key = os.environ.get(args.key_env)
        if not key:
            print(f"Error: Environment variable {args.key_env} not set", file=sys.stderr)
            sys.exit(2)
        return key
    if args.key_file:
        with open(args.key_file, "rb") as f:
            return f.read()
    return None


def create_signer(algorithm: str, key: str | bytes | None):
    """Create appropriate signer based on algorithm and key."""
    from aragora.gauntlet.signing import (
        Ed25519Signer,
        HMACSigner,
        ReceiptSigner,
        RSASigner,
    )

    if algorithm == "HMAC-SHA256":
        if not key:
            print("Error: HMAC-SHA256 requires a signing key", file=sys.stderr)
            print("Use --key, --key-env, or --key-file to provide it", file=sys.stderr)
            sys.exit(2)
        # Convert hex string to bytes if needed
        if isinstance(key, str):
            try:
                key_bytes = bytes.fromhex(key)
            except ValueError:
                key_bytes = key.encode("utf-8")
        else:
            key_bytes = key
        backend = HMACSigner(secret_key=key_bytes)
        return ReceiptSigner(backend=backend)

    elif algorithm == "RSA-SHA256":
        if not key:
            print("Error: RSA-SHA256 requires a public key file", file=sys.stderr)
            print("Use --key-file to provide the PEM-encoded public key", file=sys.stderr)
            sys.exit(2)

        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            print("Error: RSA verification requires 'cryptography' package", file=sys.stderr)
            print("Install with: pip install cryptography", file=sys.stderr)
            sys.exit(2)

        if isinstance(key, str):
            key = key.encode("utf-8")

        public_key = serialization.load_pem_public_key(key)
        backend = RSASigner(public_key=public_key)
        return ReceiptSigner(backend=backend)

    elif algorithm == "Ed25519":
        if not key:
            print("Error: Ed25519 requires a public key file", file=sys.stderr)
            print("Use --key-file to provide the PEM-encoded public key", file=sys.stderr)
            sys.exit(2)

        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            print("Error: Ed25519 verification requires 'cryptography' package", file=sys.stderr)
            print("Install with: pip install cryptography", file=sys.stderr)
            sys.exit(2)

        if isinstance(key, str):
            key = key.encode("utf-8")

        public_key = serialization.load_pem_public_key(key)
        backend = Ed25519Signer(public_key=public_key)
        return ReceiptSigner(backend=backend)

    else:
        print(f"Error: Unknown algorithm: {algorithm}", file=sys.stderr)
        print("Supported algorithms: HMAC-SHA256, RSA-SHA256, Ed25519", file=sys.stderr)
        sys.exit(2)


def print_receipt_info(data: dict, verbose: bool = False):
    """Print receipt information."""
    receipt = data.get("receipt", {})
    metadata = data.get("signature_metadata", {})

    print("\nReceipt Information:")
    print(f"  Decision ID: {receipt.get('decision_id', 'N/A')}")
    print(f"  Verdict: {receipt.get('verdict', 'N/A')}")
    print(f"  Confidence: {receipt.get('confidence', 'N/A')}")

    print("\nSignature Metadata:")
    print(f"  Algorithm: {metadata.get('algorithm', 'N/A')}")
    print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"  Key ID: {metadata.get('key_id', 'N/A')}")
    print(f"  Version: {metadata.get('version', 'N/A')}")

    # Print signatory info if present
    signatory = metadata.get("signatory")
    if signatory:
        print("\nSignatory Information:")
        print(f"  Name: {signatory.get('name', 'N/A')}")
        print(f"  Email: {signatory.get('email', 'N/A')}")
        if signatory.get("title"):
            print(f"  Title: {signatory['title']}")
        if signatory.get("organization"):
            print(f"  Organization: {signatory['organization']}")
        if signatory.get("role"):
            print(f"  Role: {signatory['role']}")
        if signatory.get("department"):
            print(f"  Department: {signatory['department']}")

    if verbose:
        print("\nFull Receipt Data:")
        print(json.dumps(receipt, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Verify decision receipt signatures offline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "receipt_file",
        help="Path to the signed receipt JSON file",
    )

    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--key",
        help="Signing key (HMAC: hex-encoded secret)",
    )
    key_group.add_argument(
        "--key-env",
        metavar="VAR",
        help="Environment variable containing the signing key",
    )
    key_group.add_argument(
        "--key-file",
        metavar="FILE",
        help="File containing signing key (PEM format for RSA/Ed25519)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed receipt information",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show pass/fail result",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    # Load receipt
    try:
        data = load_receipt(args.receipt_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.receipt_file}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}", file=sys.stderr)
        sys.exit(2)

    # Validate receipt structure
    if "signature" not in data:
        print("Error: Receipt file missing 'signature' field", file=sys.stderr)
        sys.exit(2)
    if "signature_metadata" not in data:
        print("Error: Receipt file missing 'signature_metadata' field", file=sys.stderr)
        sys.exit(2)
    if "receipt" not in data:
        print("Error: Receipt file missing 'receipt' field", file=sys.stderr)
        sys.exit(2)

    # Get algorithm from metadata
    algorithm = data["signature_metadata"].get("algorithm")
    if not algorithm:
        print("Error: Missing algorithm in signature_metadata", file=sys.stderr)
        sys.exit(2)

    # Get key
    key = get_key(args)

    # Create signer and verify
    try:
        from aragora.gauntlet.signing import SignedReceipt

        signer = create_signer(algorithm, key)
        signed_receipt = SignedReceipt.from_dict(data)
        is_valid = signer.verify(signed_receipt)

    except ImportError as e:
        print(f"Error: Missing required package: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: Verification failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Output result
    if args.json_output:
        result = {
            "valid": is_valid,
            "receipt_file": args.receipt_file,
            "algorithm": algorithm,
            "key_id": data["signature_metadata"].get("key_id"),
            "timestamp": data["signature_metadata"].get("timestamp"),
        }
        if data["signature_metadata"].get("signatory"):
            result["signatory"] = data["signature_metadata"]["signatory"]
        print(json.dumps(result, indent=2))
    elif args.quiet:
        print("VALID" if is_valid else "INVALID")
    else:
        if is_valid:
            print("\n[VALID] Signature verification successful")
        else:
            print("\n[INVALID] Signature verification failed")

        if not args.quiet:
            print_receipt_info(data, verbose=args.verbose)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
