"""
Microsoft Teams messaging operations mixin.

Provides send, update, delete, typing indicator, and response
methods for the TeamsConnector.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from aragora.connectors.exceptions import (
    ConnectorNetworkError,
    ConnectorTimeoutError,
)

from aragora.connectors.chat.models import (
    BotCommand,
    SendMessageResponse,
    UserInteraction,
)

import aragora.connectors.chat.teams._constants as _tc

try:
    import httpx
except ImportError:
    pass

logger = logging.getLogger(__name__)


class _TeamsConnectorProtocol(Protocol):
    """Protocol for methods expected by TeamsMessagingMixin from the main connector."""

    def _check_circuit_breaker(self) -> tuple[bool, str | None]: ...

    async def _get_access_token(self) -> str: ...

    async def _http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = ...,
        json: dict[str, Any] | None = ...,
        data: Any | None = ...,
        content: bytes | None = ...,
        files: dict[str, Any] | None = ...,
        max_retries: int = ...,
        base_delay: float = ...,
        timeout: float | None = ...,
        return_raw: bool = ...,
        operation: str = ...,
    ) -> tuple[bool, dict[str, Any] | bytes | None, str | None]: ...

    def _record_failure(self, error: Exception | None = ...) -> None: ...

    # Methods from the mixin itself that are called internally
    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict[str, Any]] | None = ...,
        thread_id: str | None = ...,
        service_url: str | None = ...,
        conversation_id: str | None = ...,
        **kwargs: Any,
    ) -> SendMessageResponse: ...

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: list[dict[str, Any]] | None = ...,
        service_url: str | None = ...,
        **kwargs: Any,
    ) -> SendMessageResponse: ...

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: list[dict[str, Any]] | None = ...,
    ) -> SendMessageResponse: ...


class TeamsMessagingMixin:
    """Mixin providing messaging operations for TeamsConnector.

    Expected from the main connector class (e.g., TeamsConnector):
    - _check_circuit_breaker(): Check circuit breaker state
    - _get_access_token(): Get OAuth access token
    - _http_request(): Make HTTP request with retry/circuit breaker
    - _record_failure(): Record failure for circuit breaker
    """

    # Attribute declarations for methods expected from the main connector class
    # These are provided by HTTPResilienceMixin and TeamsConnector
    _check_circuit_breaker: Any
    _get_access_token: Any
    _http_request: Any
    _record_failure: Any

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        service_url: str | None = None,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send message to Teams channel.

        Includes circuit breaker protection for fault tolerance.
        """
        if not _tc.HTTPX_AVAILABLE:
            return SendMessageResponse(
                success=False,
                error="httpx not available",
            )

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            token = await self._get_access_token()
            base_url = service_url or _tc.BOT_FRAMEWORK_API_BASE
            conv_id = conversation_id or channel_id

            # Build activity payload
            activity: dict[str, Any] = {
                "type": "message",
                "text": text,
            }

            # Add Adaptive Card if blocks provided
            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Handle threaded reply
            if thread_id:
                activity["replyToId"] = thread_id

            # Use shared HTTP helper with retry and circuit breaker
            success, data, error = await self._http_request(
                method="POST",
                url=f"{base_url}/v3/conversations/{conv_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="send_message",
            )

            if success and data and isinstance(data, dict):
                return SendMessageResponse(
                    success=True,
                    message_id=data.get("id"),
                    channel_id=conv_id,
                )
            else:
                return SendMessageResponse(
                    success=False,
                    error=error or "Unknown error",
                )

        except httpx.TimeoutException as e:
            classified = _tc._classify_teams_error(f"Timeout: {e}")
            logger.error("Teams send_message timeout: %s", e)
            self._record_failure(classified)
            raise ConnectorTimeoutError("Request timed out", connector_name="teams") from e
        except httpx.ConnectError as e:
            classified = _tc._classify_teams_error(f"Connection error: {e}")
            logger.error("Teams send_message connection error: %s", e)
            self._record_failure(classified)
            raise ConnectorNetworkError("Connection failed", connector_name="teams") from e
        except (
            httpx.HTTPError,
            RuntimeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            classified = _tc._classify_teams_error(str(e))
            logger.error("Teams send_message error: %s", e)
            self._record_failure(classified)
            return SendMessageResponse(
                success=False,
                error="Message send failed",
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        service_url: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update an existing Teams message.

        Includes circuit breaker protection for fault tolerance.
        """
        if not _tc.HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            token = await self._get_access_token()
            base_url = service_url or _tc.BOT_FRAMEWORK_API_BASE

            activity: dict[str, Any] = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Use shared HTTP helper with retry and circuit breaker
            success, _, error = await self._http_request(
                method="PUT",
                url=f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="update_message",
            )

            if success:
                return SendMessageResponse(
                    success=True,
                    message_id=message_id,
                    channel_id=channel_id,
                )
            else:
                return SendMessageResponse(
                    success=False,
                    error=error or "Unknown error",
                )

        except httpx.TimeoutException as e:
            classified = _tc._classify_teams_error(f"Timeout: {e}")
            logger.error("Teams update_message timeout: %s", e)
            self._record_failure(classified)
            raise ConnectorTimeoutError("Request timed out", connector_name="teams") from e
        except httpx.ConnectError as e:
            classified = _tc._classify_teams_error(f"Connection error: {e}")
            logger.error("Teams update_message connection error: %s", e)
            self._record_failure(classified)
            raise ConnectorNetworkError("Connection failed", connector_name="teams") from e
        except (
            httpx.HTTPError,
            RuntimeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            classified = _tc._classify_teams_error(str(e))
            logger.error("Teams update_message error: %s", e)
            self._record_failure(classified)
            return SendMessageResponse(success=False, error="Message update failed")

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        service_url: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Teams message.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not _tc.HTTPX_AVAILABLE:
            return False

        try:
            token = await self._get_access_token()
            base_url = service_url or _tc.BOT_FRAMEWORK_API_BASE

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="DELETE",
                url=f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                headers={"Authorization": f"Bearer {token}"},
                operation="delete_message",
            )

            if not success:
                logger.warning("Teams delete_message failed: %s", error)

            return success

        except (
            httpx.HTTPError,
            httpx.TimeoutException,
            httpx.ConnectError,
            RuntimeError,
            OSError,
        ) as e:
            logger.error("Teams delete_message error: %s", e)
            return False

    async def send_typing_indicator(
        self,
        channel_id: str,
        service_url: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Send typing indicator to a Teams conversation.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not _tc.HTTPX_AVAILABLE:
            return False

        try:
            token = await self._get_access_token()
            base_url = service_url or _tc.BOT_FRAMEWORK_API_BASE

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="POST",
                url=f"{base_url}/v3/conversations/{channel_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"type": "typing"},
                operation="send_typing_indicator",
            )

            if not success:
                logger.debug("Teams typing indicator failed: %s", error)

            return success

        except (
            httpx.HTTPError,
            httpx.TimeoutException,
            httpx.ConnectError,
            RuntimeError,
            OSError,
        ) as e:
            logger.debug("Teams typing indicator error: %s", e)
            return False

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams command (mention or direct message)."""
        if command.response_url:
            # Use response URL for async response
            return await self._send_to_response_url(
                command.response_url,
                text,
                blocks,
            )

        if command.channel:
            return await self.send_message(
                command.channel.id,
                text,
                blocks,
                service_url=command.metadata.get("service_url"),
                **kwargs,
            )

        return SendMessageResponse(
            success=False,
            error="No channel or response URL available",
        )

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams Adaptive Card action."""
        if interaction.response_url:
            return await self._send_to_response_url(
                interaction.response_url,
                text,
                blocks,
            )

        if interaction.channel and interaction.message_id and replace_original:
            return await self.update_message(
                interaction.channel.id,
                interaction.message_id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        if interaction.channel:
            return await self.send_message(
                interaction.channel.id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        return SendMessageResponse(success=False, error="No response target available")

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> SendMessageResponse:
        """
        Send response to a Bot Framework response URL.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not _tc.HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            token = await self._get_access_token()

            activity: dict[str, Any] = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="POST",
                url=response_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="send_to_response_url",
            )

            if success:
                return SendMessageResponse(success=True)
            else:
                return SendMessageResponse(success=False, error=error or "Unknown error")

        except (
            httpx.HTTPError,
            httpx.TimeoutException,
            httpx.ConnectError,
            RuntimeError,
            OSError,
        ) as e:
            logger.error("Teams response URL error: %s", e)
            return SendMessageResponse(success=False, error="Response URL delivery failed")
