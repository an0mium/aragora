"""
Doctor command - Health checks for Aragora.
"""

import os
import sys


def main() -> int:
    """Run health checks."""
    print("ARAGORA HEALTH CHECK")
    print("=" * 50)

    checks: list[tuple[str, str, bool | None]] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(("Python", py_ver, sys.version_info >= (3, 10)))

    # Required packages
    for pkg in ["aiohttp", "pydantic"]:
        try:
            __import__(pkg)
            checks.append((pkg, "installed", True))
        except ImportError:
            checks.append((pkg, "missing", False))

    # Optional packages
    for pkg in ["tiktoken", "weaviate", "docling", "unstructured"]:
        try:
            __import__(pkg)
            checks.append((pkg, "installed", True))
        except ImportError:
            checks.append((pkg, "not installed", None))

    # API keys
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
        if os.getenv(key):
            checks.append((key, "set", True))
        else:
            checks.append((key, "not set", None))

    # Print results
    print()
    all_ok = True
    for name, status, ok in checks:
        if ok is True:
            icon = "+"
        elif ok is False:
            icon = "x"
            all_ok = False
        else:
            icon = "o"
        print(f"  [{icon}] {name}: {status}")

    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
