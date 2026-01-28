"""
SDK Code Generator CLI.

Generates Python and TypeScript SDK namespaces from OpenAPI specification.

Usage:
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --output sdk/python/aragora/namespaces/
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --namespace debates --output /tmp/sdk_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .openapi_parser import OpenAPIParser
from .python_generator import PythonGenerator


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sdk_generator",
        description="Generate SDK namespaces from OpenAPI specification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all Python namespaces
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --output sdk/python/aragora/namespaces/

    # Generate specific namespace
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --namespace debates --output /tmp/sdk_test

    # List available namespaces
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --list

    # Dry run (show what would be generated)
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --dry-run
        """,
    )

    parser.add_argument(
        "--openapi",
        "-i",
        type=Path,
        required=True,
        help="Path to OpenAPI specification (JSON or YAML)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for generated files (default: current directory)",
    )

    parser.add_argument(
        "--namespace",
        "-n",
        type=str,
        default=None,
        help="Generate only specific namespace (default: all)",
    )

    parser.add_argument(
        "--language",
        "-l",
        choices=["python", "typescript"],
        default="python",
        help="Target language (default: python)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available namespaces and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate OpenAPI file exists
    if not args.openapi.exists():
        print(f"Error: OpenAPI file not found: {args.openapi}", file=sys.stderr)
        return 1

    # Parse OpenAPI spec
    if args.verbose:
        print(f"Parsing OpenAPI spec: {args.openapi}")

    try:
        api_parser = OpenAPIParser(args.openapi)
        api_parser.parse()
    except Exception as e:
        print(f"Error parsing OpenAPI spec: {e}", file=sys.stderr)
        return 1

    # Get namespaces
    namespaces = api_parser.get_endpoints_by_namespace()

    if args.verbose:
        print(f"Found {len(namespaces)} namespaces with {len(api_parser.endpoints)} endpoints")

    # List mode
    if args.list:
        print(f"\nAvailable namespaces ({len(namespaces)}):\n")
        for ns, endpoints in sorted(namespaces.items()):
            print(f"  {ns:30} ({len(endpoints):3} endpoints)")
        print(f"\nTotal: {len(api_parser.endpoints)} endpoints")
        return 0

    # Filter to specific namespace if requested
    if args.namespace:
        if args.namespace not in namespaces:
            print(f"Error: Namespace '{args.namespace}' not found", file=sys.stderr)
            print(f"Available namespaces: {', '.join(sorted(namespaces.keys()))}", file=sys.stderr)
            return 1
        namespaces = {args.namespace: namespaces[args.namespace]}

    # Set output directory
    output_dir = args.output or Path.cwd()

    # Generate based on language
    if args.language == "python":
        return _generate_python(
            api_parser,
            namespaces,
            output_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    elif args.language == "typescript":
        print("Error: TypeScript generator not yet implemented", file=sys.stderr)
        print("Use --language python for now", file=sys.stderr)
        return 1

    return 0


def _generate_python(
    api_parser: OpenAPIParser,
    namespaces: dict,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Generate Python SDK namespaces."""
    if dry_run:
        print(f"\nDry run - would generate {len(namespaces)} namespace(s):\n")
        for ns, endpoints in sorted(namespaces.items()):
            print(f"  {output_dir / f'{ns}.py'}")
            if verbose:
                for ep in endpoints:
                    print(f"    - {ep.method:6} {ep.path}")
        return 0

    if verbose:
        print(f"\nGenerating Python SDK to: {output_dir}\n")

    try:
        generator = PythonGenerator(api_parser, output_dir)

        # Generate filtered namespaces only
        generated = 0
        for ns, endpoints in sorted(namespaces.items()):
            content = generator.generate_namespace(ns, endpoints)
            generator._write_namespace(ns, content)
            generated += 1
            if verbose:
                print(f"  Generated: {ns}.py ({len(endpoints)} endpoints)")
            else:
                print(f"  {ns}.py")

        print(f"\nGenerated {generated} namespace file(s)")
        return 0

    except Exception as e:
        print(f"Error generating SDK: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
