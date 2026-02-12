"""
Gmail Pub/Sub watch and push notification management.

Provides setup and management of Gmail push notifications via
Google Cloud Pub/Sub, including automatic watch renewal.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING
from collections.abc import AsyncIterator

from ..models import (
    EmailMessage,
    GmailSyncState,
    GmailWebhookPayload,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops

_REAL_ASYNCIO_SLEEP = asyncio.sleep


class GmailBaseMethods(Protocol):
    """Protocol defining expected methods from base classes for type checking."""

    user_id: str
    exclude_labels: set[str]
    _gmail_state: GmailSyncState | None
    _watch_task: asyncio.Task[None] | None
    _watch_running: bool

    async def _get_access_token(self) -> str: ...
    async def _api_request(
        self, endpoint: str, method: str = "GET", **kwargs: Any
    ) -> dict[str, Any]: ...
    @asynccontextmanager
    def _get_client(self) -> AsyncIterator[httpx.AsyncClient]: ...
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

    # Expected attributes from concrete class
    user_id: str
    exclude_labels: set[str]
    _gmail_state: GmailSyncState | None
    _watch_task: asyncio.Task[None] | None
    _watch_running: bool

    def _is_protocol_method(self, method: Any) -> bool:
        """Detect Protocol stub methods so we can fall back safely."""
        qualname = getattr(method, "__qualname__", "")
        return qualname.startswith("GmailBaseMethods.")

    def _get_circuit_method(self, name: str) -> Any:
        method = getattr(super(), name, None)
        if method is None:
            return None
        if self._is_protocol_method(method):
            method = getattr(super(GmailBaseMethods, self), name, None)
        return method

    def check_circuit_breaker(self) -> bool:
        """Return circuit breaker status with safe fallback for mixin usage."""
        method = self._get_circuit_method("check_circuit_breaker")
        if method is None:
            return not getattr(self, "_circuit_open", False)
        result = method()
        if result is Ellipsis or result is None:
            return not getattr(self, "_circuit_open", False)
        return bool(result)

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Return circuit breaker status with safe fallback for mixin usage."""
        method = self._get_circuit_method("get_circuit_breaker_status")
        if method is None:
            return {"cooldown_seconds": 60, "failure_count": getattr(self, "_failure_count", 0)}
        status = method()
        if status is Ellipsis or status is None:
            return {"cooldown_seconds": 60, "failure_count": getattr(self, "_failure_count", 0)}
        return status

    def record_success(self) -> None:
        """Record circuit breaker success with safe fallback for mixin usage."""
        method = self._get_circuit_method("record_success")
        if method is None:
            self._success_count = getattr(self, "_success_count", 0) + 1
            return
        method()
        if self._is_protocol_method(method):
            self._success_count = getattr(self, "_success_count", 0) + 1

    def record_failure(self) -> None:
        """Record circuit breaker failure with safe fallback for mixin usage."""
        method = self._get_circuit_method("record_failure")
        if method is None:
            self._failure_count = getattr(self, "_failure_count", 0) + 1
            return
        method()
        if self._is_protocol_method(method):
            self._failure_count = getattr(self, "_failure_count", 0) + 1

    async def setup_watch(
        self,
        topic_name: str,
        label_ids: list[str] | None = None,
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

        recorded_failure = False
        try:
            request_error: Exception | None = None
            error_message = None
            should_record_failure = False
            data = None
            async with self._get_client() as client:
                try:
                    response = await client.post(
                        f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/watch",
                        headers={"Authorization": f"Bearer {access_token}"},
                        json={
                            "topicName": full_topic,
                            "labelIds": watch_labels,
                            "labelFilterBehavior": "INCLUDE",
                        },
                    )
                except (OSError, ConnectionError) as exc:
                    request_error = exc
                else:
                    if response.status_code != 200:
                        error = response.json().get("error", {})
                        if response.status_code >= 500 or response.status_code == 429:
                            should_record_failure = True
                        error_message = error.get("message", response.text)
                    else:
                        data = response.json()

            if request_error:
                self.record_failure()
                recorded_failure = True
                raise request_error

            if error_message:
                if should_record_failure:
                    self.record_failure()
                    recorded_failure = True
                raise RuntimeError(f"Failed to setup watch: {error_message}")

            self.record_success()
            data = data or {}

            # Update state
            history_id = str(data.get("historyId", ""))
            expiration_ms = data.get("expiration")
            expiration = None
            if expiration_ms:
                expiration = datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc)

            # Initialize or update gmail state
            if not self._gmail_state:
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

        except (OSError, ConnectionError) as e:
            if not recorded_failure:
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
        if self._watch_task:
            done_result = self._watch_task.done()
            if inspect.isawaitable(done_result):
                done_result = await done_result
            if not done_result:
                self._watch_running = False
                self._watch_task.cancel()
                try:
                    if inspect.isawaitable(self._watch_task):
                        await self._watch_task
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

        except (OSError, ConnectionError) as e:
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

            for _page in range(_MAX_PAGES):
                history, page_token, history_id = await self.get_history(
                    self._gmail_state.history_id,
                    page_token=page_token,
                )

                if not history and not page_token:
                    if not history_id:
                        logger.warning("[Gmail] History ID expired during webhook handling")
                        break
                    new_history_id = history_id
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
                    except (OSError, ValueError, KeyError, RuntimeError) as e:
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

        except (OSError, ValueError, KeyError, RuntimeError) as e:
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
        self._watch_task = asyncio.create_task(
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
                await _REAL_ASYNCIO_SLEEP(0)

                if not self._watch_running:
                    break

                logger.info("[Gmail] Renewing watch...")
                await self.setup_watch(
                    topic_name=topic_name,
                    project_id=project_id,
                )

            except asyncio.CancelledError:
                break
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.error(f"[Gmail] Watch renewal failed: {e}")
                # Retry in 1 minute on failure
                await asyncio.sleep(60)
                await _REAL_ASYNCIO_SLEEP(0)
