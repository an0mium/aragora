#!/usr/bin/env python3
"""
Provision Uptime Kuma status page with Aragora monitors.

This script auto-configures monitors via Uptime Kuma's socket API after
initial deployment. Run this after the status page container is running.

Usage:
    python scripts/provision_status_page.py --url http://localhost:3001
    python scripts/provision_status_page.py --url http://localhost:3001 --api-url https://api.aragora.ai

Requirements:
    pip install python-socketio websocket-client

SOC 2 Control: A1-01 - Public availability information
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Monitor:
    """Monitor configuration."""

    name: str
    type: str
    url: str
    interval: int = 60
    timeout: int = 10
    retries: int = 3
    keyword: Optional[str] = None
    method: str = "GET"
    accepted_codes: List[str] = None
    group: str = "Core Services"
    public: bool = True
    description: str = ""

    def __post_init__(self):
        if self.accepted_codes is None:
            self.accepted_codes = ["200"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Uptime Kuma API format."""
        data = {
            "name": self.name,
            "type": "http" if self.type == "http" else "keyword",
            "url": self.url,
            "interval": self.interval,
            "timeout": self.timeout,
            "maxretries": self.retries,
            "method": self.method,
            "accepted_statuscodes": self.accepted_codes,
            "description": self.description,
        }
        if self.keyword:
            data["keyword"] = self.keyword
        return data


def get_default_monitors(api_url: str) -> List[Monitor]:
    """Get default Aragora monitors."""
    return [
        Monitor(
            name="API - Health",
            type="http",
            url=f"{api_url}/api/health",
            interval=60,
            timeout=10,
            description="Primary health check endpoint",
            group="Core Services",
        ),
        Monitor(
            name="API - Detailed Health",
            type="keyword",
            url=f"{api_url}/api/health/detailed",
            interval=120,
            timeout=15,
            keyword="healthy",
            description="Comprehensive service health check",
            group="Core Services",
        ),
        Monitor(
            name="WebSocket Server",
            type="http",
            url=f"{api_url}/api/ws/stats",
            interval=60,
            timeout=10,
            description="Real-time communication endpoint",
            group="Core Services",
        ),
        Monitor(
            name="Authentication Service",
            type="http",
            url=f"{api_url}/api/auth/providers",
            interval=120,
            timeout=10,
            description="Auth endpoint availability",
            group="Core Services",
        ),
        Monitor(
            name="Database Health",
            type="keyword",
            url=f"{api_url}/api/health/stores",
            interval=120,
            timeout=15,
            keyword="healthy",
            description="Database connectivity check",
            group="Infrastructure",
            public=False,
        ),
        Monitor(
            name="Circuit Breakers",
            type="keyword",
            url=f"{api_url}/api/health/circuits",
            interval=120,
            timeout=10,
            keyword="healthy",
            description="Circuit breaker status",
            group="Infrastructure",
            public=False,
        ),
        Monitor(
            name="Knowledge Mound",
            type="keyword",
            url=f"{api_url}/api/health/knowledge-mound",
            interval=300,
            timeout=15,
            keyword="healthy",
            description="Knowledge Mound subsystem",
            group="Features",
            public=False,
        ),
        Monitor(
            name="Encryption Service",
            type="keyword",
            url=f"{api_url}/api/health/encryption",
            interval=300,
            timeout=10,
            keyword="healthy",
            description="Encryption service status",
            group="Security",
            public=False,
        ),
    ]


def provision_with_socketio(
    uptime_url: str,
    username: str,
    password: str,
    monitors: List[Monitor],
) -> bool:
    """Provision monitors using Uptime Kuma's socket.io API."""
    try:
        import socketio
    except ImportError:
        logger.error(
            "python-socketio not installed. Run: pip install python-socketio websocket-client"
        )
        return False

    sio = socketio.Client()
    success = False

    @sio.event
    def connect():
        logger.info("Connected to Uptime Kuma")

    @sio.event
    def disconnect():
        logger.info("Disconnected from Uptime Kuma")

    try:
        # Connect
        sio.connect(uptime_url, transports=["websocket"])

        # Login
        logger.info(f"Logging in as {username}...")
        response = sio.call(
            "login",
            {
                "username": username,
                "password": password,
                "token": "",
            },
        )

        if not response.get("ok"):
            logger.error(f"Login failed: {response.get('msg', 'Unknown error')}")
            return False

        logger.info("Login successful")

        # Get existing monitors
        existing = sio.call("getMonitorList")
        existing_names = {m["name"] for m in existing.values()} if existing else set()

        # Add monitors
        added = 0
        skipped = 0

        for monitor in monitors:
            if monitor.name in existing_names:
                logger.info(f"  Skipping '{monitor.name}' (already exists)")
                skipped += 1
                continue

            logger.info(f"  Adding '{monitor.name}'...")
            result = sio.call("add", monitor.to_dict())

            if result.get("ok"):
                added += 1
            else:
                logger.warning(f"    Failed: {result.get('msg', 'Unknown error')}")

        logger.info(f"\nProvisioning complete: {added} added, {skipped} skipped")
        success = True

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
    finally:
        sio.disconnect()

    return success


def provision_with_api(
    uptime_url: str,
    api_key: str,
    monitors: List[Monitor],
) -> bool:
    """Provision monitors using Uptime Kuma's REST API (if available)."""
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return False

    headers = {"Authorization": f"Bearer {api_key}"}

    # Get existing monitors
    try:
        response = requests.get(f"{uptime_url}/api/monitor", headers=headers)
        if response.status_code == 200:
            existing = response.json()
            existing_names = {m["name"] for m in existing}
        else:
            logger.warning("Could not fetch existing monitors")
            existing_names = set()
    except Exception as e:
        logger.warning(f"Could not fetch existing monitors: {e}")
        existing_names = set()

    added = 0
    skipped = 0

    for monitor in monitors:
        if monitor.name in existing_names:
            logger.info(f"  Skipping '{monitor.name}' (already exists)")
            skipped += 1
            continue

        logger.info(f"  Adding '{monitor.name}'...")
        try:
            response = requests.post(
                f"{uptime_url}/api/monitor",
                headers=headers,
                json=monitor.to_dict(),
            )
            if response.status_code in (200, 201):
                added += 1
            else:
                logger.warning(f"    Failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.info(f"\nProvisioning complete: {added} added, {skipped} skipped")
    return True


def export_monitors_json(monitors: List[Monitor], output_path: str) -> None:
    """Export monitors to JSON file for manual import."""
    data = {
        "_comment": "Uptime Kuma monitor configuration for Aragora",
        "_version": "1.0.0",
        "_import_instructions": "Import via Settings > Backup > Restore (partial import)",
        "monitors": [m.to_dict() for m in monitors],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {len(monitors)} monitors to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Provision Uptime Kuma with Aragora monitors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive setup (will prompt for credentials)
    python scripts/provision_status_page.py --url http://localhost:3001

    # With environment variables
    UPTIME_KUMA_USERNAME=admin UPTIME_KUMA_PASSWORD=secret \\
        python scripts/provision_status_page.py --url http://localhost:3001

    # Export monitors to JSON for manual import
    python scripts/provision_status_page.py --export monitors.json

    # Custom API URL
    python scripts/provision_status_page.py --url http://localhost:3001 \\
        --api-url https://api.example.com
        """,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:3001",
        help="Uptime Kuma URL (default: http://localhost:3001)",
    )
    parser.add_argument(
        "--api-url",
        default="https://api.aragora.ai",
        help="Aragora API URL to monitor (default: https://api.aragora.ai)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("UPTIME_KUMA_USERNAME"),
        help="Uptime Kuma username (or UPTIME_KUMA_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("UPTIME_KUMA_PASSWORD"),
        help="Uptime Kuma password (or UPTIME_KUMA_PASSWORD env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("UPTIME_KUMA_API_KEY"),
        help="Uptime Kuma API key (or UPTIME_KUMA_API_KEY env var)",
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export monitors to JSON file instead of provisioning",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Get monitors
    monitors = get_default_monitors(args.api_url)

    # Export mode
    if args.export:
        export_monitors_json(monitors, args.export)
        return 0

    # Dry run
    if args.dry_run:
        logger.info("Dry run - would configure these monitors:\n")
        for m in monitors:
            logger.info(f"  - {m.name} ({m.type})")
            logger.info(f"    URL: {m.url}")
            logger.info(f"    Interval: {m.interval}s, Timeout: {m.timeout}s")
            if m.keyword:
                logger.info(f"    Keyword: {m.keyword}")
            logger.info("")
        return 0

    # Get credentials
    username = args.username
    password = args.password

    if not username:
        try:
            username = input("Uptime Kuma username: ")
        except (EOFError, KeyboardInterrupt):
            logger.info("\nAborted")
            return 1

    if not password:
        try:
            import getpass

            password = getpass.getpass("Uptime Kuma password: ")
        except (EOFError, KeyboardInterrupt):
            logger.info("\nAborted")
            return 1

    # Provision
    logger.info(f"\nProvisioning {len(monitors)} monitors to {args.url}...")
    logger.info(f"API URL: {args.api_url}\n")

    if args.api_key:
        success = provision_with_api(args.url, args.api_key, monitors)
    else:
        success = provision_with_socketio(args.url, username, password, monitors)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
