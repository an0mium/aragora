"""
Gmail Pub/Sub watch and push notification management.

Provides setup and management of Gmail push notifications via
Google Cloud Pub/Sub, including automatic watch renewal.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, Protocol, TYPE_CHECKING

from ..models import (
    EmailMessage,
    GmailSyncState,
    GmailWebhookPayload,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class GmailBaseMethods(Protocol):
    """Protocol defining expected methods from base classes for type checking."""

    user_id: str
    exclude_labels: set[str]
    _gmail_state: GmailSyncState | None
    _watch_task: Optional["asyncio.Task[None]"]
    _watch_running: bool

    async def _get_access_token(self) -> str: ...
    async def _api_request(
        self, endpoint: str, method: str = "GET", **kwargs: Any
    ) -> dict[str, Any]: ...
    @asynccontextmanager
    def _get_client(self) -> AsyncIterator["httpx.AsyncClient"]: ...
    def check_circuit_breaker(self) -> bool: ...
    def get_circuit_breaker_status(self) -> dict[str, Any]: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    async def get_history(
        self, start_history_id: str, page_token: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None, str | None]: ...
    async def get_message(self, message_id: str) -> EmailMessage: ...


class GmailWatchMixin(GmailBaseMethods):
    """Mixin providing Pub/Sub watch and push notification operations."""

    async def setup_watch(
        self,
        topic_name: str,
        label_ids: Optional[list[str]] = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Set up Gmail push notifications via Google Cloud Pub/Sub.

        This enables real-time notifications when new emails arrive,
        eliminating the need for polling.

        Args:
            topic_name: Pub/Sub topic name (e.g., "gmail-notifications")
            label_ids: Labels to watch (default: ["INBOX"])
            project_id: Google Cloud project ID (reads from env if not provided)

        Returns:
            Dict with watch status, history_id, and expiration

        Note:
            - Requires Gmail API scope and Pub/Sub topic access
            - Watch expires after ~7 days, use start_watch_renewal() for auto-renewal
            - Topic must grant Gmail service account publish permission
        """
        import os

        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            raise ValueError("project_id required for Pub/Sub watch")

        full_topic = f"projects/{project_id}/topics/{topic_name}"
        watch_labels = label_ids or ["INBOX"]

        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/watch",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "topicName": full_topic,
                        "labelIds": watch_labels,
                        "labelFilterBehavior": "INCLUDE",
                    },
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to setup watch: {error.get('message', response.text)}"
                    )

                self.record_success()
                data = response.json()

                # Update state
                history_id = str(data.get("historyId", ""))
                expiration_ms = data.get("expiration")
                expiration = None
                if expiration_ms:
                    expiration = datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc)

                # Initialize or update gmail state
                if not self._gmail_state:  # type: ignore[has-type]
                    self._gmail_state = GmailSyncState(
                        user_id=self.user_id,
                        history_id=history_id,
                    )
                else:
                    self._gmail_state.history_id = history_id

                self._gmail_state.watch_expiration = expiration
                self._gmail_state.watch_resource_id = "active"

                logger.info(f"[Gmail] Watch set up successfully, expires at {expiration}")

                return {
                    "success": True,
                    "history_id": history_id,
                    "expiration": expiration.isoformat() if expiration else None,
                    "topic": full_topic,
                    "labels": watch_labels,
                }

        except Exception as e:
            if not isinstance(e, (RuntimeError, ConnectionError)):
                self.record_failure()
            logger.error(f"[Gmail] Watch setup failed: {e}")
            raise

    async def stop_watch(self) -> dict[str, Any]:
        """
        Stop Gmail push notifications.

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        # Cancel renewal task if running
        if self._watch_task and not self._watch_task.done():  # type: ignore[has-type]
            self._watch_running = False
            self._watch_task.cancel()  # type: ignore[has-type]
            try:
                await self._watch_task  # type: ignore[has-type]
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/stop",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code == 204:
                    self.record_success()

                    # Clear watch state
                    if self._gmail_state:
                        self._gmail_state.watch_resource_id = None
                        self._gmail_state.watch_expiration = None

                    logger.info("[Gmail] Watch stopped successfully")
                    return {"success": True}
                else:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    logger.warning(
                        f"[Gmail] Stop watch returned {response.status_code}: "
                        f"{error.get('message', response.text)}"
                    )
                    return {
                        "success": False,
                        "error": error.get("message", "Unknown error"),
                    }

        except Exception as e:
            self.record_failure()
            logger.error(f"[Gmail] Failed to stop watch: {e}")
            raise

    async def handle_pubsub_notification(
        self,
        payload: dict[str, Any],
    ) -> list[EmailMessage]:
        """
        Handle incoming Pub/Sub webhook notification.

        Parses the notification, fetches new messages via History API,
        and returns the list of new emails.

        Args:
            payload: Raw webhook payload from Pub/Sub

        Returns:
            List of new EmailMessage objects
        """
        webhook = GmailWebhookPayload.from_pubsub(payload)

        # Validate this is for us
        if self._gmail_state and webhook.email_address:
            if (
                self._gmail_state.email_address
                and webhook.email_address != self._gmail_state.email_address
            ):
                logger.warning(
                    f"[Gmail] Webhook for {webhook.email_address} "
                    f"but expecting {self._gmail_state.email_address}"
                )
                return []

        logger.info(f"[Gmail] Pub/Sub notification received: historyId={webhook.history_id}")

        # Use History API to get changes
        if not self._gmail_state or not self._gmail_state.history_id:
            logger.warning("[Gmail] No history ID available, cannot process webhook")
            return []

        try:
            new_messages: list[EmailMessage] = []
            page_token = None
            new_history_id = self._gmail_state.history_id

            while True:
                history, page_token, history_id = await self.get_history(
                    self._gmail_state.history_id,
                    page_token=page_token,
                )

                if not history and not page_token:
                    if not history_id:
                        logger.warning("[Gmail] History ID expired during webhook handling")
                        break
                    break

                # Extract new message IDs
                new_message_ids: set[str] = set()
                for record in history:
                    for msg_added in record.get("messagesAdded", []):
                        msg_data = msg_added.get("message", {})
                        msg_id = msg_data.get("id")
                        labels = msg_data.get("labelIds", [])

                        # Skip excluded labels
                        if self.exclude_labels and any(
                            lbl in self.exclude_labels for lbl in labels
                        ):
                            continue

                        if msg_id:
                            new_message_ids.add(msg_id)

                # Fetch full messages
                for msg_id in new_message_ids:
                    try:
                        msg = await self.get_message(msg_id)
                        new_messages.append(msg)
                    except Exception as e:
                        logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

                if history_id:
                    new_history_id = history_id

                if not page_token:
                    break

            # Update history ID
            self._gmail_state.history_id = new_history_id
            self._gmail_state.last_sync = datetime.now(timezone.utc)
            self._gmail_state.indexed_messages += len(new_messages)

            logger.info(f"[Gmail] Webhook processed: {len(new_messages)} new messages")
            return new_messages

        except Exception as e:
            if self._gmail_state:
                self._gmail_state.sync_errors += 1
                self._gmail_state.last_error = str(e)
            logger.error(f"[Gmail] Webhook processing failed: {e}")
            raise

    async def start_watch_renewal(
        self,
        topic_name: str,
        renewal_hours: int = 144,  # 6 days (watch expires after ~7 days)
        project_id: str | None = None,
    ) -> None:
        """
        Start background task to auto-renew watch before expiration.

        Args:
            topic_name: Pub/Sub topic name
            renewal_hours: Hours between renewals (default: 144 = 6 days)
            project_id: Google Cloud project ID
        """
        if self._watch_task and not self._watch_task.done():
            logger.warning("[Gmail] Watch renewal already running")
            return

        self._watch_running = True
        self._watch_task = asyncio.create_task(  # type: ignore[assignment]
            self._watch_renewal_loop(topic_name, renewal_hours, project_id)
        )
        logger.info(f"[Gmail] Watch renewal started (every {renewal_hours} hours)")

    async def _watch_renewal_loop(
        self,
        topic_name: str,
        renewal_hours: int,
        project_id: str | None,
    ) -> None:
        """Background loop to renew watch before expiration."""
        renewal_seconds = renewal_hours * 3600

        while self._watch_running:
            try:
                await asyncio.sleep(renewal_seconds)

                if not self._watch_running:
                    break

                logger.info("[Gmail] Renewing watch...")
                await self.setup_watch(
                    topic_name=topic_name,
                    project_id=project_id,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Gmail] Watch renewal failed: {e}")
                # Retry in 1 minute on failure
                await asyncio.sleep(60)
