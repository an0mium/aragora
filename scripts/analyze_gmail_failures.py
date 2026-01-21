#!/usr/bin/env python3
"""
Gmail Inbox Analyzer for Aragora Failures

This script:
1. Connects to your Gmail account(s) via OAuth
2. Searches for Aragora-related failure emails
3. Categorizes and summarizes the failures
4. Optionally exports to JSON for further analysis

Usage:
    # First, set up Google OAuth credentials:
    export GMAIL_CLIENT_ID="your-client-id.apps.googleusercontent.com"
    export GMAIL_CLIENT_SECRET="your-client-secret"

    # Run the analyzer
    python scripts/analyze_gmail_failures.py

    # Or analyze specific accounts
    python scripts/analyze_gmail_failures.py --accounts email1@gmail.com email2@gmail.com
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import webbrowser
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# Add aragora to path if running from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports to avoid issues if dependencies missing
GmailConnector = None
EmailMessage = None


def lazy_imports():
    """Import Aragora modules lazily."""
    global GmailConnector, EmailMessage

    try:
        from aragora.connectors.enterprise.communication.gmail import GmailConnector
        from aragora.connectors.enterprise.communication.models import EmailMessage

        return True
    except ImportError as e:
        print(f"Error importing Aragora modules: {e}")
        print("Make sure you're running from the aragora directory.")
        return False


@dataclass
class FailureEmail:
    """Represents a failure-related email."""

    id: str
    thread_id: str
    subject: str
    from_address: str
    date: datetime
    snippet: str
    failure_type: str  # error, test_failure, ci_failure, alert, exception
    severity: str  # critical, high, medium, low
    keywords_found: list[str]
    body_preview: str


@dataclass
class FailureSummary:
    """Summary of failure emails."""

    total_emails: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    by_sender: dict[str, int]
    date_range: tuple[str, str]
    top_subjects: list[tuple[str, int]]
    emails: list[FailureEmail]


# Failure detection patterns
FAILURE_PATTERNS = {
    "error": [
        r"error",
        r"failed",
        r"failure",
        r"exception",
        r"crash",
        r"broken",
    ],
    "test_failure": [
        r"test.*fail",
        r"pytest.*fail",
        r"assertion.*error",
        r"tests?\s+failed",
        r"\d+\s+failed",
    ],
    "ci_failure": [
        r"build\s+fail",
        r"ci\s+fail",
        r"pipeline\s+fail",
        r"github\s+actions",
        r"workflow.*fail",
        r"deployment.*fail",
    ],
    "alert": [
        r"alert",
        r"warning",
        r"critical",
        r"urgent",
        r"down",
        r"outage",
    ],
    "exception": [
        r"traceback",
        r"stacktrace",
        r"unhandled\s+exception",
        r"runtime\s+error",
        r"type\s*error",
        r"value\s*error",
        r"key\s*error",
        r"attribute\s*error",
    ],
}

# Severity patterns
SEVERITY_PATTERNS = {
    "critical": [r"critical", r"fatal", r"emergency", r"down", r"outage"],
    "high": [r"error", r"failed", r"exception", r"crash"],
    "medium": [r"warning", r"alert", r"issue"],
    "low": [r"notice", r"info", r"minor"],
}

# Aragora-specific patterns
ARAGORA_PATTERNS = [
    r"aragora",
    r"debate",
    r"consensus",
    r"agent",
    r"nomic",
    r"elo",
    r"belief",
    r"knowledge\s*mound",
    r"rlm",
    r"orchestrat",
]


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handler for OAuth callback."""

    auth_code: Optional[str] = None

    def do_GET(self):
        """Handle GET request with OAuth callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <head><title>Gmail Connected</title></head>
                <body style="font-family: monospace; background: #1a1a2e; color: #00ff00; padding: 40px;">
                    <h1>Gmail Connected Successfully!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        else:
            error = params.get("error", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body>OAuth Error: {error}</body></html>".encode())

    def log_message(self, format, *args):
        """Suppress HTTP logging."""
        pass


def find_available_port(start_port: int = 8766, max_attempts: int = 20) -> int:
    """Find an available port starting from start_port."""
    import socket

    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise OSError(f"No available ports found in range {start_port}-{start_port + max_attempts}")


async def get_oauth_token(connector: Any, account_name: str) -> bool:
    """Get OAuth token via browser flow."""
    port = find_available_port()
    print(f"Using port {port} for OAuth callback...")
    redirect_uri = f"http://localhost:{port}/callback"

    # Get authorization URL
    auth_url = connector.get_oauth_url(
        redirect_uri=redirect_uri,
        state=account_name,
    )

    print(f"\n{'='*60}")
    print(f"Connecting Gmail account: {account_name}")
    print(f"{'='*60}")
    print("\nOpening browser for Google authorization...")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")

    # Start local server for callback
    OAuthCallbackHandler.auth_code = None
    server = HTTPServer(("localhost", port), OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    # Open browser
    webbrowser.open(auth_url)

    # Wait for callback
    print("Waiting for authorization (timeout: 2 minutes)...")
    while OAuthCallbackHandler.auth_code is None:
        server.handle_request()
        if OAuthCallbackHandler.auth_code:
            break

    if not OAuthCallbackHandler.auth_code:
        print("Authorization timed out or failed.")
        return False

    # Exchange code for tokens
    print("Exchanging authorization code for tokens...")
    success = await connector.authenticate(
        code=OAuthCallbackHandler.auth_code,
        redirect_uri=redirect_uri,
    )

    if success:
        print(f"Successfully connected to {account_name}!")
        return True
    else:
        print(f"Failed to authenticate {account_name}")
        return False


def classify_failure(subject: str, body: str) -> tuple[str, str, list[str]]:
    """Classify a failure email by type and severity."""
    text = f"{subject} {body}".lower()

    # Find failure type
    failure_type = "unknown"
    keywords_found = []

    for ftype, patterns in FAILURE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                failure_type = ftype
                keywords_found.append(pattern)
                break
        if failure_type != "unknown":
            break

    # Find severity
    severity = "medium"
    for sev, patterns in SEVERITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                severity = sev
                break
        if severity != "medium":
            break

    return failure_type, severity, keywords_found


def is_aragora_related(subject: str, body: str) -> bool:
    """Check if email is Aragora-related."""
    text = f"{subject} {body}".lower()
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in ARAGORA_PATTERNS)


async def search_failure_emails(
    connector: Any,
    days_back: int = 30,
    max_results: int = 500,
) -> list[FailureEmail]:
    """Search for failure-related emails."""
    # Build Gmail search query for failures
    failure_terms = [
        "failed",
        "error",
        "failure",
        "exception",
        "crash",
        "alert",
        "traceback",
    ]

    # Also search for Aragora-specific terms
    aragora_terms = [
        "aragora",
        "debate",
        "nomic",
        "consensus",
    ]

    # Combine into Gmail query
    # Search for (failure terms) AND (aragora terms)
    failure_query = " OR ".join(failure_terms)
    aragora_query = " OR ".join(aragora_terms)

    after_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    query = f"({aragora_query}) ({failure_query}) after:{after_date}"

    print(f"\nSearching with query: {query}")
    print(f"Looking back {days_back} days, max {max_results} results...")

    failures = []
    page_token = None
    total_fetched = 0

    while total_fetched < max_results:
        batch_size = min(100, max_results - total_fetched)
        messages, page_token = await connector.list_messages(
            query=query,
            max_results=batch_size,
            page_token=page_token,
        )

        if not messages:
            break

        for msg_summary in messages:
            # Get full message
            try:
                msg = await connector.get_message(msg_summary["id"])

                # Check if Aragora-related
                body = msg.body_text or msg.snippet or ""
                if not is_aragora_related(msg.subject, body):
                    continue

                # Classify failure
                failure_type, severity, keywords = classify_failure(msg.subject, body)

                failures.append(
                    FailureEmail(
                        id=msg.id,
                        thread_id=msg.thread_id,
                        subject=msg.subject,
                        from_address=msg.from_address,
                        date=msg.date,
                        snippet=msg.snippet[:200] if msg.snippet else "",
                        failure_type=failure_type,
                        severity=severity,
                        keywords_found=keywords,
                        body_preview=body[:500] if body else "",
                    )
                )
            except Exception as e:
                print(f"  Error fetching message {msg_summary.get('id', 'unknown')}: {e}")

        total_fetched += len(messages)
        print(f"  Fetched {total_fetched} messages, found {len(failures)} failures...")

        if not page_token:
            break

    return failures


def summarize_failures(failures: list[FailureEmail]) -> FailureSummary:
    """Create summary of failure emails."""
    if not failures:
        return FailureSummary(
            total_emails=0,
            by_type={},
            by_severity={},
            by_sender={},
            date_range=("", ""),
            top_subjects=[],
            emails=[],
        )

    by_type: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    by_sender: dict[str, int] = defaultdict(int)
    subjects: dict[str, int] = defaultdict(int)

    dates = []
    for f in failures:
        by_type[f.failure_type] += 1
        by_severity[f.severity] += 1
        by_sender[f.from_address] += 1
        # Normalize subject (remove Re:, Fwd:, etc.)
        subj = re.sub(r"^(re:|fwd:|fw:)\s*", "", f.subject.lower(), flags=re.IGNORECASE)
        subjects[subj] += 1
        if f.date:
            dates.append(f.date)

    date_range = ("", "")
    if dates:
        dates.sort()
        date_range = (
            dates[0].isoformat() if dates[0] else "",
            dates[-1].isoformat() if dates[-1] else "",
        )

    top_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)[:10]

    return FailureSummary(
        total_emails=len(failures),
        by_type=dict(by_type),
        by_severity=dict(by_severity),
        by_sender=dict(by_sender),
        date_range=date_range,
        top_subjects=top_subjects,
        emails=failures,
    )


def print_summary(summary: FailureSummary, account: str):
    """Print failure summary to console."""
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS: {account}")
    print(f"{'='*60}")

    if summary.total_emails == 0:
        print("\nNo Aragora-related failure emails found!")
        return

    print(f"\nTotal failure emails: {summary.total_emails}")
    print(f"Date range: {summary.date_range[0]} to {summary.date_range[1]}")

    print("\n--- By Failure Type ---")
    for ftype, count in sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ftype}: {count}")

    print("\n--- By Severity ---")
    for sev, count in sorted(summary.by_severity.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sev}: {count}")

    print("\n--- Top Senders ---")
    for sender, count in sorted(summary.by_sender.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {sender}: {count}")

    print("\n--- Top Failure Subjects ---")
    for subj, count in summary.top_subjects[:5]:
        print(f"  [{count}x] {subj[:60]}...")

    print("\n--- Recent Critical/High Failures ---")
    critical = [f for f in summary.emails if f.severity in ("critical", "high")]
    for f in sorted(critical, key=lambda x: x.date or datetime.min, reverse=True)[:5]:
        date_str = f.date.strftime("%Y-%m-%d") if f.date else "unknown"
        print(f"  [{date_str}] [{f.severity.upper()}] {f.subject[:50]}...")
        print(f"           From: {f.from_address}")


def export_results(
    summaries: dict[str, FailureSummary],
    output_file: str,
):
    """Export results to JSON file."""
    export_data = {}
    for account, summary in summaries.items():
        export_data[account] = {
            "total_emails": summary.total_emails,
            "by_type": summary.by_type,
            "by_severity": summary.by_severity,
            "by_sender": summary.by_sender,
            "date_range": summary.date_range,
            "top_subjects": summary.top_subjects,
            "emails": [
                {
                    "id": e.id,
                    "subject": e.subject,
                    "from": e.from_address,
                    "date": e.date.isoformat() if e.date else None,
                    "failure_type": e.failure_type,
                    "severity": e.severity,
                    "keywords": e.keywords_found,
                    "snippet": e.snippet,
                }
                for e in summary.emails
            ],
        }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"\nResults exported to: {output_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Gmail inboxes for Aragora failure emails")
    parser.add_argument(
        "--accounts",
        nargs="+",
        help="Email accounts to analyze (will prompt for OAuth)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days to look back (default: 30)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum emails to fetch per account (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gmail_failures.json",
        help="Output JSON file (default: gmail_failures.json)",
    )
    parser.add_argument(
        "--skip-oauth",
        action="store_true",
        help="Skip OAuth (use saved tokens)",
    )

    args = parser.parse_args()

    # Try to get credentials from Aragora's Secrets Manager first (AWS)
    client_id = None
    client_secret = None

    try:
        from aragora.config.secrets import get_secret

        client_id = get_secret("GMAIL_CLIENT_ID") or get_secret("GOOGLE_CLIENT_ID")
        client_secret = get_secret("GMAIL_CLIENT_SECRET") or get_secret("GOOGLE_CLIENT_SECRET")

        if client_id and client_secret:
            print("Loaded Gmail OAuth credentials from AWS Secrets Manager")
    except ImportError:
        pass

    # Fall back to environment variables
    if not client_id:
        client_id = os.environ.get("GMAIL_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID")
    if not client_secret:
        client_secret = os.environ.get("GMAIL_CLIENT_SECRET") or os.environ.get(
            "GOOGLE_CLIENT_SECRET"
        )

    if not client_id or not client_secret:
        print("ERROR: Gmail OAuth credentials not configured!")
        print("\nCredentials not found in:")
        print("  1. AWS Secrets Manager (GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET)")
        print("  2. Environment variables")
        print("\nTo fix, ensure your AWS secret 'aragora/production' contains:")
        print(
            '  {"GMAIL_CLIENT_ID": "xxx.apps.googleusercontent.com", "GMAIL_CLIENT_SECRET": "xxx"}'
        )
        print("\nOr set environment variables:")
        print("  export GMAIL_CLIENT_ID='your-client-id.apps.googleusercontent.com'")
        print("  export GMAIL_CLIENT_SECRET='your-client-secret'")
        return 1

    # Import Aragora modules
    if not lazy_imports():
        return 1

    # Default accounts if not specified
    accounts = args.accounts or ["Account 1", "Account 2"]

    print(f"\n{'='*60}")
    print("ARAGORA GMAIL FAILURE ANALYZER")
    print(f"{'='*60}")
    print(f"\nAnalyzing {len(accounts)} account(s)")
    print(f"Looking back: {args.days} days")
    print(f"Max results per account: {args.max_results}")

    summaries = {}

    for account in accounts:
        connector = GmailConnector()

        # OAuth flow
        if not args.skip_oauth:
            success = await get_oauth_token(connector, account)
            if not success:
                print(f"Skipping {account} due to OAuth failure")
                continue

        # Search for failures
        print(f"\nSearching {account} for Aragora failure emails...")
        failures = await search_failure_emails(
            connector,
            days_back=args.days,
            max_results=args.max_results,
        )

        # Summarize
        summary = summarize_failures(failures)
        summaries[account] = summary

        # Print summary
        print_summary(summary, account)

    # Export results
    if summaries:
        export_results(summaries, args.output)

    # Final summary
    total_failures = sum(s.total_emails for s in summaries.values())
    print(f"\n{'='*60}")
    print(
        f"TOTAL: Found {total_failures} Aragora failure emails across {len(summaries)} account(s)"
    )
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
