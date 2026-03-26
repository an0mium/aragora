"""CLI command for inbox triage: the trust wedge entry point.

Usage::

    aragora triage run --batch 5
    aragora triage run --batch 5 --auto-approve
    aragora triage run --dry-run
    aragora triage auth
    aragora triage status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings
from enum import Enum

logger = logging.getLogger(__name__)


def _load_local_dotenv() -> None:
    """Best-effort dotenv loading for local founder runs."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        load_dotenv("/etc/aragora/.env")
    except Exception:
        return


def _get_secret_fallback(name: str) -> str:
    """Resolve a secret via Aragora's secret loader when available."""
    try:
        from aragora.config.secrets import get_secret

        return str(get_secret(name) or "")
    except Exception:
        return ""


def _resolve_gmail_oauth_credentials() -> tuple[str, str]:
    """Resolve Gmail OAuth client credentials from env, dotenv, or secrets."""
    _load_local_dotenv()

    client_id_candidates = (
        "GMAIL_CLIENT_ID",
        "GOOGLE_GMAIL_CLIENT_ID",
        "GOOGLE_CLIENT_ID",
    )
    client_secret_candidates = (
        "GMAIL_CLIENT_SECRET",
        "GOOGLE_GMAIL_CLIENT_SECRET",
        "GOOGLE_CLIENT_SECRET",
    )

    client_id = ""
    for name in client_id_candidates:
        value = os.environ.get(name)
        if value:
            client_id = value
            break
    if not client_id:
        for name in client_id_candidates:
            value = _get_secret_fallback(name)
            if value:
                client_id = value
                os.environ.setdefault("GMAIL_CLIENT_ID", value)
                break

    client_secret = ""
    for name in client_secret_candidates:
        value = os.environ.get(name)
        if value:
            client_secret = value
            break
    if not client_secret:
        for name in client_secret_candidates:
            value = _get_secret_fallback(name)
            if value:
                client_secret = value
                os.environ.setdefault("GMAIL_CLIENT_SECRET", value)
                break

    return client_id, client_secret


def _action_value(action: object) -> str:
    if isinstance(action, Enum):
        return str(action.value)
    return str(action)


def add_triage_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'triage' subcommand."""
    parser = subparsers.add_parser(
        "triage",
        help="Inbox triage via adversarial debate with receipt-gated actions",
        description=(
            "Run the inbox trust wedge: fetch unread Gmail, debate triage\n"
            "actions adversarially, persist signed receipts, and execute\n"
            "approved actions (archive/star/label/ignore).\n\n"
            "Commands:\n"
            "  run     Fetch and triage unread emails\n"
            "  auth    Authenticate with Gmail via OAuth\n"
            "  status  Show triage session status\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="triage_command")

    run_p = sub.add_parser("run", help="Fetch and triage unread emails")
    run_p.add_argument(
        "--batch",
        type=int,
        default=5,
        help="Number of unread messages to fetch (default: 5)",
    )
    run_p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve safe actions (archive/star/ignore) when confidence >= 0.85",
    )
    run_p.add_argument(
        "--provider",
        default="gmail",
        choices=["gmail"],
        help="Email provider (default: gmail)",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview triage decisions without executing Gmail actions",
    )
    run_p.add_argument(
        "--page-token",
        default=None,
        help="Continue from a Gmail nextPageToken returned by a prior dry-run",
    )

    sub.add_parser("auth", help="Authenticate with Gmail via OAuth")
    sub.add_parser("status", help="Show triage session status")

    parser.set_defaults(func=cmd_triage)


def cmd_triage(args: argparse.Namespace) -> None:
    """Dispatch triage subcommands."""
    command = getattr(args, "triage_command", None)
    if command == "run":
        batch = getattr(args, "batch", 5)
        auto_approve = getattr(args, "auto_approve", False)
        dry_run = getattr(args, "dry_run", False)
        page_token = getattr(args, "page_token", None)
        asyncio.run(
            _run_triage(
                batch_size=batch,
                auto_approve=auto_approve,
                dry_run=dry_run,
                page_token=page_token,
            )
        )
    elif command == "auth":
        asyncio.run(_run_gmail_auth())
    elif command == "status":
        _show_status()
    else:
        print("Usage: aragora triage {run,auth,status}")
        sys.exit(1)


async def _initialize_triage_storage() -> None:
    """Initialize event-loop-bound PostgreSQL infrastructure for CLI triage.

    This mirrors server startup: create the shared pool inside the active event
    loop so stores used by debate hooks do not fall back to SQLite purely
    because triage is running from the CLI.
    """
    try:
        from aragora.server.startup.database import init_postgres_pool

        await init_postgres_pool()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI initialization
        logger.debug("Triage storage initialization skipped: %s", exc)


async def _shutdown_triage_storage() -> None:
    """Best-effort shutdown for triage-owned database resources."""
    try:
        from aragora.server.startup.database import close_postgres_pool

        await close_postgres_pool()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage shared-pool shutdown skipped: %s", exc)

    try:
        from aragora.server.http_client_pool import close_http_pool

        await close_http_pool()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage HTTP client pool shutdown skipped: %s", exc)

    try:
        from aragora.agents.api_agents.common import close_shared_connector

        await close_shared_connector()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage API connector shutdown skipped: %s", exc)

    try:
        from aragora.storage.connection_factory import close_all_pools

        await close_all_pools()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage connection-factory shutdown skipped: %s", exc)

    try:
        from aragora.events.dispatcher import shutdown_dispatcher

        shutdown_dispatcher(wait=True)
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage dispatcher shutdown skipped: %s", exc)

    try:
        from aragora.storage.webhook_config_store import reset_webhook_config_store

        reset_webhook_config_store()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage webhook config reset skipped: %s", exc)

    try:
        from aragora.inbox.trust_wedge import (
            reset_inbox_trust_wedge_service,
            reset_inbox_trust_wedge_store,
        )

        reset_inbox_trust_wedge_service()
        reset_inbox_trust_wedge_store()
    except Exception as exc:  # noqa: BLE001 - best-effort CLI shutdown
        logger.debug("Triage trust wedge reset skipped: %s", exc)

    # Give async transports a brief chance to finish their close callbacks
    # before the triage event loop shuts down.
    await asyncio.sleep(0.05)


async def _run_triage(
    batch_size: int,
    auto_approve: bool,
    dry_run: bool = False,
    page_token: str | None = None,
) -> None:
    """Run the inbox triage pipeline."""
    try:
        from aragora.inbox.triage_runner import InboxTriageRunner
        from aragora.inbox.triage_diagnostics import TriageRunDiagnostics
    except ImportError:
        print("Error: inbox triage module not available", file=sys.stderr)
        sys.exit(1)

    profile = os.getenv("ARAGORA_TRIAGE_PROFILE", "staged_v1")
    verbose = logging.getLogger().isEnabledFor(logging.DEBUG)
    diagnostics = TriageRunDiagnostics(
        profile=profile,
        batch_size=batch_size,
        auto_approve=auto_approve and not dry_run,
        dry_run=dry_run,
        verbose=verbose,
        diagnostics_dir=os.getenv("ARAGORA_TRIAGE_DIAGNOSTICS_DIR"),
    )
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
        module=r"multiprocessing\.resource_tracker",
    )

    with diagnostics.activate(), diagnostics.capture_logging():
        decisions = []
        try:
            await _initialize_triage_storage()

            gmail = _get_gmail_connector()
            if gmail is None:
                print(
                    "Error: Gmail not configured. Run 'aragora triage auth' first,\n"
                    "or set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET environment variables.",
                    file=sys.stderr,
                )
                sys.exit(1)

            from aragora.inbox.trust_wedge import get_inbox_trust_wedge_service

            wedge_service = get_inbox_trust_wedge_service()
            runner = InboxTriageRunner(
                gmail_connector=gmail,
                wedge_service=wedge_service,
                diagnostics=diagnostics,
                profile=profile,
            )

            if dry_run:
                print(
                    f"[DRY RUN] Fetching up to {batch_size} unread messages "
                    "(no actions will be executed)..."
                )
            else:
                print(f"Fetching up to {batch_size} unread messages...")

            decisions = await runner.run_triage(
                batch_size=batch_size,
                auto_approve=auto_approve and not dry_run,
                page_token=page_token,
            )
            meta = diagnostics.finalize(decisions)

            if not decisions:
                print("No messages to triage.")
                _print_run_footer(decisions, meta, diagnostics)
                return

            if dry_run:
                print("\n[DRY RUN] Proposed triage decisions (no actions executed):")
                _print_decisions(decisions)
                _print_run_footer(
                    decisions,
                    meta,
                    diagnostics,
                    next_page_token=getattr(runner, "next_page_token", None),
                )
                return

            _print_decisions(decisions)
            _print_run_footer(
                decisions,
                meta,
                diagnostics,
                next_page_token=getattr(runner, "next_page_token", None),
            )

            if not auto_approve:
                try:
                    from aragora.inbox.cli_review import CLIReviewLoop

                    loop = CLIReviewLoop(review_fn=wedge_service.review_receipt)
                    loop.review_batch(decisions)
                except ImportError:
                    return
        finally:
            await _shutdown_triage_storage()


async def _run_gmail_auth() -> None:
    """Run interactive Gmail OAuth flow from CLI."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path
    from urllib.parse import parse_qs, urlparse

    client_id, client_secret = _resolve_gmail_oauth_credentials()

    if not client_id or not client_secret:
        print(
            "Error: Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET environment\n"
            "variables before running 'aragora triage auth'.\n"
            "\n"
            "Get these from: https://console.cloud.google.com/apis/credentials",
            file=sys.stderr,
        )
        sys.exit(1)

    redirect_uri = "http://localhost:8089/callback"
    auth_code_holder: dict[str, str] = {}

    class _OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            code = qs.get("code", [None])[0]
            if code:
                auth_code_holder["code"] = code
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authenticated! You can close this tab.</h2></body></html>"
                )
            else:
                self.send_response(400)
                self.end_headers()
                error = qs.get("error", ["unknown"])[0]
                auth_code_holder["error"] = error
                self.wfile.write(f"OAuth error: {error}".encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    try:
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        connector = GmailConnector()
    except ImportError:
        print("Error: GmailConnector not available", file=sys.stderr)
        sys.exit(1)

    auth_url = connector.get_oauth_url(redirect_uri)
    print("Opening browser for Gmail authorization...")
    print(f"\nIf the browser doesn't open, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8089), _OAuthCallbackHandler)
    print("Waiting for authorization callback on localhost:8089...")
    server.handle_request()
    server.server_close()

    if "error" in auth_code_holder:
        print(f"OAuth failed: {auth_code_holder['error']}", file=sys.stderr)
        sys.exit(1)

    code = auth_code_holder.get("code")
    if not code:
        print("No authorization code received.", file=sys.stderr)
        sys.exit(1)

    success = await connector.authenticate(code=code, redirect_uri=redirect_uri)
    if not success:
        print("Failed to exchange authorization code for tokens.", file=sys.stderr)
        sys.exit(1)

    refresh_token = getattr(connector, "_refresh_token", None)
    if refresh_token:
        token_path = Path.home() / ".aragora" / "gmail_refresh_token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(refresh_token)
        token_path.chmod(0o600)
        print("\nGmail authenticated successfully!")
        print(f"Refresh token saved to: {token_path}")
        print("\nYou can now run: aragora triage run --dry-run")
    else:
        print("\nAuthenticated but no refresh token received.")
        print("Try re-running 'aragora triage auth'.")


def _get_gmail_connector():
    """Build and return an authenticated GmailConnector, or None."""
    from pathlib import Path

    client_id, client_secret = _resolve_gmail_oauth_credentials()
    if not (client_id and client_secret):
        return None

    try:
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        connector = GmailConnector()

        refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN", "").strip()
        if not refresh_token:
            token_file = Path.home() / ".aragora" / "gmail_refresh_token"
            if token_file.exists():
                refresh_token = token_file.read_text().strip()

        if refresh_token:
            connector._refresh_token = refresh_token

        return connector
    except ImportError:
        logger.warning("GmailConnector not available")
        return None


def _print_decisions(decisions: list) -> None:
    """Print triage decisions as a summary table."""
    print(f"\n{'─' * 60}")
    print(f"{'Action':<10} {'Confidence':>10}  {'Status':<10} {'Subject'}")
    print(f"{'─' * 60}")

    for d in decisions:
        action = _action_value(getattr(d, "final_action", "?"))
        confidence = getattr(d, "confidence", 0.0)
        blocked = bool(getattr(d, "blocked_by_policy", False))
        status = "blocked" if blocked else str(getattr(d, "receipt_state", "created"))
        intent = getattr(d, "intent", None)
        subject = "(unknown)"
        if intent and hasattr(intent, "_subject"):
            subject = intent._subject
        elif intent:
            subject = getattr(intent, "message_id", "?")

        bar = "█" * int(confidence * 10)
        print(f"{action:<10} {confidence:>8.1%} {bar:<10}  {status:<10} {subject[:29]}")

    print(f"{'─' * 60}")
    print(f"Total: {len(decisions)} decisions")

    from aragora.inbox.trust_wedge import ReceiptState

    approved = sum(
        1 for d in decisions if getattr(d, "receipt_state", None) == ReceiptState.APPROVED.value
    )
    executed = sum(
        1 for d in decisions if getattr(d, "receipt_state", None) == ReceiptState.EXECUTED.value
    )
    if approved or executed:
        print(f"  Approved: {approved}  Executed: {executed}")


def _print_run_footer(
    decisions: list,
    meta: dict[str, object],
    diagnostics: object,
    *,
    next_page_token: str | None = None,
) -> None:
    """Print a compact diagnostics-aware run footer."""
    print(
        "Run summary: "
        f"processed={len(decisions)} "
        f"fast={meta.get('fast_tier_count', 0)} "
        f"escalated={meta.get('escalated_count', 0)} "
        f"blocked={meta.get('blocked_count', 0)} "
        f"suppressed={meta.get('suppressed_diagnostics_count', 0)}"
    )
    if getattr(diagnostics, "has_degraded_or_blocking", lambda: False)():
        print(f"Diagnostics: {meta.get('artifact_dir')}")
    if next_page_token:
        print(f"Next page token: {next_page_token}")


def _show_status() -> None:
    """Show triage configuration status."""
    print("Inbox Triage Status")
    print(f"{'─' * 40}")

    client_id, client_secret = _resolve_gmail_oauth_credentials()
    has_gmail = bool(client_id and client_secret)
    print(f"  Gmail configured:     {'yes' if has_gmail else 'NO'}")

    from pathlib import Path

    key_path = Path.home() / ".aragora" / "signing.key"
    print(f"  Durable signing key:  {'yes' if key_path.exists() else 'NO'}")

    token_path = Path.home() / ".aragora" / "gmail_refresh_token"
    print(f"  Gmail refresh token:  {'yes' if token_path.exists() else 'NO'}")

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    print(f"  OpenRouter fallback:  {'yes' if has_openrouter else 'NO'}")

    providers = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY",
    }
    for name, var in providers.items():
        status = "yes" if os.environ.get(var) else "no"
        print(f"  {name + ' key:':<22}{status}")
