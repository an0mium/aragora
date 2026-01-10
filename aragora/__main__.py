#!/usr/bin/env python3
"""
Enable running Aragora commands via: python -m aragora

Usage:
    python -m aragora doctor    # Run health checks
    python -m aragora [args]    # Run main CLI
"""

import sys


def main():
    """Route to appropriate command."""
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        # Remove 'doctor' from args and run doctor command
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from aragora.cli.doctor import main as doctor_main
        sys.exit(doctor_main())
    else:
        # Fall through to main CLI
        from aragora.cli.main import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
