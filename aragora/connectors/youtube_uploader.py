"""
YouTube upload connector for publishing debate videos.

Uses YouTube Data API v3 with OAuth 2.0 for uploading videos
and managing video metadata.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx

from aragora.resilience import CircuitBreaker
from aragora.connectors.exceptions import (
    ConnectorError,
    ConnectorAuthError,
    ConnectorQuotaError,
    ConnectorAPIError,
)

logger = logging.getLogger(__name__)


class YouTubeError(ConnectorError):
    """Base exception for YouTube connector errors."""

    def __init__(self, message: str = "YouTube API operation failed", **kwargs):
        super().__init__(message, connector_name="youtube", **kwargs)


class YouTubeAuthError(YouTubeError, ConnectorAuthError):
    """Authentication/authorization failed."""

    def __init__(self, message: str = "YouTube authentication failed. Check OAuth credentials."):
        super().__init__(message)


class YouTubeQuotaError(YouTubeError, ConnectorQuotaError):
    """API quota exceeded."""

    def __init__(self, message: str = "YouTube API quota exceeded", remaining: int = 0, reset_hours: int = 24):
        super().__init__(f"{message}. Remaining: {remaining} units. Resets in ~{reset_hours}h", quota_reset=reset_hours * 3600)
        self.remaining = remaining
        self.reset_hours = reset_hours


class YouTubeUploadError(YouTubeError):
    """Video upload failed."""

    def __init__(self, message: str = "YouTube video upload failed", video_path: str = ""):
        full_message = f"{message}: {video_path}" if video_path else message
        super().__init__(full_message, is_retryable=True)
        self.video_path = video_path


class YouTubeAPIError(YouTubeError, ConnectorAPIError):
    """General API error."""

    def __init__(self, message: str = "YouTube API request failed", status_code: Optional[int] = None):
        full_message = f"{message} (HTTP {status_code})" if status_code else message
        # Let ConnectorAPIError determine retryability based on status code
        is_retryable = status_code is not None and 500 <= status_code < 600
        super().__init__(full_message, is_retryable=is_retryable)
        self.status_code = status_code


# YouTube API limits
MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5000
MAX_TAGS_LENGTH = 500  # Total characters for all tags


@dataclass
class YouTubeVideoMetadata:
    """Metadata for YouTube video upload."""

    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    category_id: str = "28"  # Science & Technology
    privacy_status: str = "public"  # public, unlisted, private
    made_for_kids: bool = False
    language: str = "en"

    def __post_init__(self):
        # Enforce limits
        if len(self.title) > MAX_TITLE_LENGTH:
            self.title = self.title[: MAX_TITLE_LENGTH - 3] + "..."

        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            self.description = self.description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

        # Truncate tags if total length exceeds limit
        total_len = sum(len(tag) for tag in self.tags)
        while total_len > MAX_TAGS_LENGTH and self.tags:
            self.tags.pop()
            total_len = sum(len(tag) for tag in self.tags)

    def to_api_body(self) -> dict:
        """Convert to YouTube API request body."""
        return {
            "snippet": {
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "categoryId": self.category_id,
                "defaultLanguage": self.language,
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": self.made_for_kids,
            },
        }


@dataclass
class UploadResult:
    """Result of a YouTube upload."""

    video_id: str
    title: str
    url: str
    success: bool = True
    error: Optional[str] = None
    upload_status: str = "complete"  # complete, processing, failed


class YouTubeRateLimiter:
    """Rate limiter for YouTube API quota management."""

    def __init__(self, daily_quota: int = 10000):
        """
        Initialize rate limiter.

        Args:
            daily_quota: Daily API quota units (default 10,000)
        """
        self.daily_quota = daily_quota
        self.used_quota = 0
        self.reset_time: Optional[float] = None

    def _check_reset(self) -> None:
        """Reset quota if a day has passed."""
        now = time.time()
        if self.reset_time is None or now > self.reset_time:
            # Reset at midnight Pacific Time (YouTube's reset time)
            self.used_quota = 0
            # Set next reset to ~24 hours from now
            self.reset_time = now + 86400

    def can_upload(self) -> bool:
        """
        Check if we have quota for an upload.

        Upload costs 1600 units.
        """
        self._check_reset()
        return self.used_quota + 1600 <= self.daily_quota

    def record_upload(self) -> None:
        """Record an upload (1600 units)."""
        self._check_reset()
        self.used_quota += 1600

    def record_api_call(self, units: int = 1) -> None:
        """Record a generic API call."""
        self._check_reset()
        self.used_quota += units

    @property
    def remaining_quota(self) -> int:
        """Get remaining quota units."""
        self._check_reset()
        return max(0, self.daily_quota - self.used_quota)


class YouTubeUploaderConnector:
    """
    Upload videos to YouTube using the Data API v3.

    Supports:
    - Video uploads with metadata
    - OAuth 2.0 authentication with refresh tokens
    - Resumable uploads for large files
    - Quota management

    Environment variables:
    - YOUTUBE_CLIENT_ID: Google OAuth client ID
    - YOUTUBE_CLIENT_SECRET: Google OAuth client secret
    - YOUTUBE_REFRESH_TOKEN: OAuth refresh token (for offline access)
    """

    # YouTube API endpoints
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
    API_URL = "https://www.googleapis.com/youtube/v3"

    # OAuth scopes
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ):
        self.client_id = client_id or os.environ.get("YOUTUBE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.environ.get("YOUTUBE_REFRESH_TOKEN", "")

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

        self.rate_limiter = YouTubeRateLimiter()
        # Use 5 minute cooldown for YouTube API (matches original recovery_timeout)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)

        # Log warning if credentials incomplete
        if not all([self.client_id, self.client_secret, self.refresh_token]):
            missing = []
            if not self.client_id:
                missing.append("YOUTUBE_CLIENT_ID")
            if not self.client_secret:
                missing.append("YOUTUBE_CLIENT_SECRET")
            if not self.refresh_token:
                missing.append("YOUTUBE_REFRESH_TOKEN")
            logger.warning(f"YouTube credentials incomplete. Missing: {', '.join(missing)}. Uploads will fail.")

    @property
    def is_configured(self) -> bool:
        """Check if YouTube credentials are configured."""
        return all([self.client_id, self.client_secret, self.refresh_token])

    def get_auth_url(self, redirect_uri: str, state: str = "") -> str:
        """
        Get OAuth authorization URL for user consent.

        Args:
            redirect_uri: OAuth callback URL
            state: State parameter for CSRF protection

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state

        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from OAuth callback
            redirect_uri: Same redirect_uri used in auth request

        Returns:
            Token response dict containing access_token and refresh_token

        Raises:
            YouTubeAuthError: If token exchange fails
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.TOKEN_URL,
                    data={
                        "code": code,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    # Store refresh token
                    if "refresh_token" in data:
                        self.refresh_token = data["refresh_token"]
                    return data
                else:
                    # Don't log full response which may contain sensitive data
                    logger.error(f"Token exchange failed: HTTP {response.status_code}")
                    raise YouTubeAuthError(
                        f"Token exchange failed with status {response.status_code}"
                    )

        except YouTubeAuthError:
            raise
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            raise YouTubeAuthError(f"Token exchange failed: {e}") from e

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
            return False

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.TOKEN_URL,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data["access_token"]
                    # Token expires in ~1 hour, refresh 5 min early
                    self._token_expiry = time.time() + data.get("expires_in", 3600) - 300
                    return True
                else:
                    # Don't log full response which may contain sensitive data
                    logger.error(f"Token refresh failed: HTTP {response.status_code}")
                    return False

        except httpx.TimeoutException as e:
            logger.error(f"Token refresh timeout: {e}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Token refresh network error: {e}")
            return False
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Token refresh parse error: {e}")
            return False

    async def _get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        if self._access_token and self._token_expiry and time.time() < self._token_expiry:
            return self._access_token

        if await self._refresh_access_token():
            return self._access_token

        return None

    async def upload(
        self,
        video_path: Path,
        metadata: YouTubeVideoMetadata,
    ) -> UploadResult:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to video file
            metadata: Video metadata

        Returns:
            UploadResult with video details or error
        """
        if not self.is_configured:
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error="YouTube API credentials not configured",
            )

        if not self.circuit_breaker.can_proceed():
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error="Circuit breaker open - too many recent failures",
            )

        if not self.rate_limiter.can_upload():
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error=f"Daily quota exceeded. Remaining: {self.rate_limiter.remaining_quota}",
            )

        if not video_path.exists():
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error=f"Video file not found: {video_path}",
            )

        access_token = await self._get_access_token()
        if not access_token:
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error="Failed to obtain access token",
            )

        try:
            import httpx

            # Start resumable upload
            file_size = video_path.stat().st_size

            # Initialize upload
            init_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/*",
            }

            init_url = f"{self.UPLOAD_URL}?uploadType=resumable&part=snippet,status"

            async with httpx.AsyncClient(timeout=60.0) as client:
                # Initialize resumable upload
                init_response = await client.post(
                    init_url,
                    headers=init_headers,
                    json=metadata.to_api_body(),
                )

                if init_response.status_code != 200:
                    self.circuit_breaker.record_failure()
                    return UploadResult(
                        video_id="",
                        title=metadata.title,
                        url="",
                        success=False,
                        error=f"Upload init failed: {init_response.status_code}",
                    )

                upload_url = init_response.headers.get("Location")
                if not upload_url:
                    return UploadResult(
                        video_id="",
                        title=metadata.title,
                        url="",
                        success=False,
                        error="No upload URL in response",
                    )

                # Upload the file
                video_data = video_path.read_bytes()
                upload_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "video/*",
                    "Content-Length": str(file_size),
                }

                upload_response = await client.put(
                    upload_url,
                    headers=upload_headers,
                    content=video_data,
                    timeout=600.0,  # 10 min for large files
                )

                if upload_response.status_code in (200, 201):
                    data = upload_response.json()
                    video_id = data["id"]

                    self.circuit_breaker.record_success()
                    self.rate_limiter.record_upload()

                    logger.info(f"Uploaded video: {video_id}")
                    return UploadResult(
                        video_id=video_id,
                        title=metadata.title,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        success=True,
                        upload_status=data.get("status", {}).get("uploadStatus", "complete"),
                    )
                else:
                    self.circuit_breaker.record_failure()
                    return UploadResult(
                        video_id="",
                        title=metadata.title,
                        url="",
                        success=False,
                        error=f"Upload failed: {upload_response.status_code}",
                    )

        except httpx.TimeoutException as e:
            logger.error(f"YouTube upload timeout: {e}")
            self.circuit_breaker.record_failure()
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error=f"Upload timeout: {e}",
            )
        except httpx.RequestError as e:
            logger.error(f"YouTube upload network error: {e}")
            self.circuit_breaker.record_failure()
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error=f"Network error: {e}",
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f"YouTube upload error: {e}")
            self.circuit_breaker.record_failure()
            return UploadResult(
                video_id="",
                title=metadata.title,
                url="",
                success=False,
                error=f"Upload error: {e}",
            )

    async def get_video_status(self, video_id: str) -> dict:
        """
        Get status of an uploaded video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video status dict with status and processingDetails

        Raises:
            YouTubeAuthError: If authentication fails
            YouTubeAPIError: If API call fails or video not found
        """
        access_token = await self._get_access_token()
        if not access_token:
            raise YouTubeAuthError("Failed to obtain access token")

        try:
            import httpx

            url = f"{self.API_URL}/videos"
            params = {
                "part": "status,processingDetails",
                "id": video_id,
            }
            headers = {"Authorization": f"Bearer {access_token}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    self.rate_limiter.record_api_call(1)
                    data = response.json()
                    items = data.get("items", [])
                    if items:
                        return items[0]
                    raise YouTubeAPIError(f"Video not found: {video_id}")
                else:
                    raise YouTubeAPIError(
                        f"API request failed", status_code=response.status_code
                    )

        except (YouTubeAuthError, YouTubeAPIError):
            raise
        except Exception as e:
            logger.error(f"Failed to get video status: {e}")
            raise YouTubeAPIError(f"Failed to get video status: {e}") from e


def create_video_metadata_from_debate(
    task: str,
    agents: list[str],
    consensus_reached: bool,
    debate_id: str,
    duration_seconds: Optional[int] = None,
) -> YouTubeVideoMetadata:
    """
    Create YouTube video metadata from debate information.

    Args:
        task: Debate topic
        agents: List of participating agents
        consensus_reached: Whether consensus was reached
        debate_id: Unique debate identifier
        duration_seconds: Optional video duration

    Returns:
        YouTubeVideoMetadata ready for upload
    """
    # Create title
    title = f"AI Debate: {task}"
    if len(title) > MAX_TITLE_LENGTH:
        title = f"AI Debate: {task[:MAX_TITLE_LENGTH - 15]}..."

    # Create description
    agent_list = "\n".join(f"- {agent}" for agent in agents)
    result = "Consensus reached" if consensus_reached else "No consensus"

    description = f"""AI agents debate: {task}

Participants:
{agent_list}

Result: {result}

---
Generated by Aragora - AI red team decision stress-test framework
Debate ID: {debate_id}

#AIDebate #Aragora #MultiAgent
"""

    # Create tags
    tags = ["AI", "debate", "artificial intelligence", "multi-agent", "Aragora"]
    for agent in agents[:5]:
        tags.append(agent.replace("-", " "))

    return YouTubeVideoMetadata(
        title=title,
        description=description,
        tags=tags,
        category_id="28",  # Science & Technology
        privacy_status="public",
    )
