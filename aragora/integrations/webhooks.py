"""
Outbound webhook dispatcher for aragora events.

Non-blocking, bounded-queue implementation that sends events to external endpoints
without affecting debate loop performance.
"""

import atexit
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import queue
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from aragora.resilience import CircuitOpenError, get_circuit_breaker

logger = logging.getLogger(__name__)


def _safe_log(level: int, msg: str) -> None:
    """Log a message safely, ignoring errors during interpreter shutdown.

    During atexit or interpreter shutdown, the logging module may be
    partially torn down, causing ValueError or other exceptions. The
    stream handlers may also have closed streams even if handlers exist.
    """
    try:
        # Check if logging is still functional
        if not logging.root.handlers and not logger.handlers:
            return

        # Check all handlers in the chain (root + logger-specific)
        all_handlers = list(logging.root.handlers) + list(logger.handlers)
        for handler in all_handlers:
            if hasattr(handler, "stream"):
                stream = handler.stream
                if stream is None:
                    return
                # Check if stream is closed
                if hasattr(stream, "closed") and stream.closed:
                    return
                # Additional check for sys.stdout/stderr being replaced
                if hasattr(stream, "fileno"):
                    try:
                        stream.fileno()
                    except (ValueError, OSError):
                        return  # Stream file descriptor is invalid

        logger.log(level, msg)
    except (ValueError, RuntimeError, AttributeError, OSError, TypeError):
        # Logging system is shutting down - silently ignore
        pass


# Default event types that webhooks receive (low-frequency, high-value)
DEFAULT_EVENT_TYPES = frozenset(
    {
        "debate_start",
        "debate_end",
        "consensus",
        "cycle_start",
        "cycle_end",
        "verification_result",
        "error",
        "proposal_accepted",
        "proposal_rejected",
        # Gauntlet events
        "gauntlet_start",
        "gauntlet_complete",
        "gauntlet_finding",
        "gauntlet_verdict",
        "decision_receipt_generated",
    }
)


class AragoraJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for aragora events.

    Handles:
    - set → sorted list (deterministic)
    - datetime → ISO 8601 string
    - Objects with to_dict() → their dict representation
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, frozenset):
            return sorted(list(obj))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""

    name: str
    url: str
    secret: str = ""
    event_types: Set[str] = field(default_factory=lambda: set(DEFAULT_EVENT_TYPES))
    loop_ids: Optional[Set[str]] = None  # None = all loops
    timeout_s: float = 10.0
    max_retries: int = 3
    backoff_base_s: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        """Create config from dictionary with validation and normalization.

        Raises ValueError if required fields are missing.
        Normalizes event_types/loop_ids to sets without mutating input.
        """
        # Work on a copy to avoid mutating input
        safe_data = data.copy()

        # Validate required fields
        if "name" not in safe_data or not safe_data["name"]:
            raise ValueError("WebhookConfig requires 'name' field")
        if "url" not in safe_data or not safe_data["url"]:
            raise ValueError("WebhookConfig requires 'url' field")

        # Normalize event_types to set
        event_types = safe_data.get("event_types")
        if event_types is None:
            event_types = set(DEFAULT_EVENT_TYPES)
        elif isinstance(event_types, (list, tuple)):
            event_types = set(event_types)
        elif isinstance(event_types, set):
            pass  # Already a set
        else:
            logger.warning(f"Invalid event_types type {type(event_types)}, using defaults")
            event_types = set(DEFAULT_EVENT_TYPES)

        # Normalize loop_ids to set or None
        loop_ids = safe_data.get("loop_ids")
        if loop_ids is None:
            pass  # None means all loops
        elif isinstance(loop_ids, (list, tuple)):
            loop_ids = set(loop_ids)
        elif isinstance(loop_ids, set):
            pass  # Already a set
        else:
            logger.warning(f"Invalid loop_ids type {type(loop_ids)}, using None (all loops)")
            loop_ids = None

        return cls(
            name=str(safe_data["name"]),
            url=str(safe_data["url"]),
            secret=str(safe_data.get("secret", "")),
            event_types=event_types,
            loop_ids=loop_ids,
            timeout_s=float(safe_data.get("timeout_s", 10.0)),
            max_retries=int(safe_data.get("max_retries", 3)),
            backoff_base_s=float(safe_data.get("backoff_base_s", 1.0)),
        )


def sign_payload(secret: str, body: bytes) -> str:
    """Generate HMAC-SHA256 signature for webhook payload.

    Returns empty string if no secret provided or on error.
    """
    if not secret:
        return ""
    try:
        signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return f"sha256={signature}"
    except Exception as e:
        logger.debug(f"Failed to sign payload: {e}")
        return ""


def load_webhook_configs() -> List[WebhookConfig]:
    """Load webhook configs from environment.

    Checks ARAGORA_WEBHOOKS (inline JSON) or ARAGORA_WEBHOOKS_CONFIG (file path).
    Returns empty list if neither is set or parsing fails.
    Invalid individual configs are skipped with warnings.
    """
    configs: List[WebhookConfig] = []

    # Try inline JSON first
    inline = os.environ.get("ARAGORA_WEBHOOKS", "").strip()
    if inline:
        try:
            configs_data = json.loads(inline)
            if not isinstance(configs_data, list):
                logger.warning("ARAGORA_WEBHOOKS must be a JSON array")
                return []
            for i, item in enumerate(configs_data):
                try:
                    configs.append(WebhookConfig.from_dict(item))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid webhook config at index {i}: {e}")
            return configs
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ARAGORA_WEBHOOKS: {e}")
            return []

    # Try file path
    config_path = os.environ.get("ARAGORA_WEBHOOKS_CONFIG", "").strip()
    if config_path:
        if not os.path.exists(config_path):
            logger.warning(f"Webhook config file not found: {config_path}")
            return []
        try:
            with open(config_path, "r") as f:
                configs_data = json.load(f)
            if not isinstance(configs_data, list):
                logger.warning(f"Webhook config file must contain a JSON array: {config_path}")
                return []
            for i, item in enumerate(configs_data):
                try:
                    configs.append(WebhookConfig.from_dict(item))
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Skipping invalid webhook config at index {i} in {config_path}: {e}"
                    )
            return configs
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load webhook config from {config_path}: {e}")
            return []

    return []


class WebhookDispatcher:
    """Thread-safe webhook dispatcher with bounded queue.

    Events are enqueued non-blocking and delivered by a background daemon thread.
    """

    DEFAULT_QUEUE_SIZE = 1000
    DROP_LOG_INTERVAL_S = 10.0

    def __init__(
        self,
        configs: List[WebhookConfig],
        queue_max_size: Optional[int] = None,
        allow_localhost: Optional[bool] = None,
    ):
        self.configs = configs

        # Allow localhost for testing (via constructor or env var)
        if allow_localhost is None:
            self._allow_localhost = os.environ.get(
                "ARAGORA_WEBHOOK_ALLOW_LOCALHOST", ""
            ).lower() in ("1", "true", "yes")
        else:
            self._allow_localhost = allow_localhost

        # Allow queue size override via env var
        if queue_max_size is None:
            env_size = os.environ.get("ARAGORA_WEBHOOK_QUEUE_SIZE", "").strip()
            if env_size.isdigit():
                queue_max_size = min(int(env_size), 100000)  # Cap at 100k
            else:
                queue_max_size = self.DEFAULT_QUEUE_SIZE

        self._queue: queue.Queue = queue.Queue(maxsize=queue_max_size)
        self._shutdown_event = threading.Event()  # Thread-safe shutdown signal
        self._started = False  # Tracks if we've ever started
        self._worker: Optional[threading.Thread] = None

        # Thread pool for parallel webhook delivery (prevents one slow webhook from blocking others)
        # Max workers = number of webhook configs, capped at 10 to prevent resource exhaustion
        max_workers = min(len(configs), 10) if configs else 4
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="webhook-deliver"
        )

        # Semaphore to limit pending deliveries and provide backpressure
        # Without this, executor queue can grow unbounded under sustained load
        self._delivery_semaphore = threading.Semaphore(max_workers * 2)

        # Thread-safe stats
        self._stats_lock = threading.Lock()
        self._drop_count = 0
        self._delivery_count = 0
        self._failure_count = 0
        self._last_drop_log = 0.0

    def start(self) -> None:
        """Start the background worker thread."""
        if self._started and not self._shutdown_event.is_set():
            return  # Already running
        self._shutdown_event.clear()  # Reset shutdown signal
        self._started = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="aragora-webhook-dispatcher"
        )
        self._worker.start()

        # Register shutdown hook for graceful cleanup
        atexit.register(self.stop)

        logger.info(f"Webhook dispatcher started with {len(self.configs)} endpoint(s)")

    @property
    def is_running(self) -> bool:
        """Check if the dispatcher is currently running."""
        return self._started and not self._shutdown_event.is_set()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker (for graceful shutdown).

        Uses threading.Event for thread-safe shutdown signaling, avoiding
        race conditions between setting the flag and checking worker state.
        """
        self._shutdown_event.set()  # Thread-safe signal to stop
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

        # Shutdown executor, wait for pending deliveries to complete
        self._executor.shutdown(wait=True, cancel_futures=False)

        with self._stats_lock:
            # Use safe logging to avoid errors during interpreter shutdown
            _safe_log(
                logging.INFO,
                f"Webhook dispatcher stopped. "
                f"Delivered: {self._delivery_count}, "
                f"Failed: {self._failure_count}, "
                f"Dropped: {self._drop_count}",
            )

    def enqueue(self, event_dict: Dict[str, Any]) -> bool:
        """Non-blocking enqueue of an event for delivery.

        Returns True if enqueued, False if dropped (no matching config or queue full).
        """
        if self._shutdown_event.is_set():
            return False

        event_type = event_dict.get("type", "")
        loop_id = event_dict.get("loop_id", "")

        # Pre-filter: only enqueue if at least one config wants this event
        if not any(self._matches_config(cfg, event_type, loop_id) for cfg in self.configs):
            return False

        try:
            self._queue.put_nowait(event_dict)
            return True
        except queue.Full:
            with self._stats_lock:
                self._drop_count += 1
                now = time.time()
                if now - self._last_drop_log > self.DROP_LOG_INTERVAL_S:
                    logger.warning(f"Webhook queue full, dropped {self._drop_count} events total")
                    self._last_drop_log = now
            return False

    def _matches_config(self, cfg: WebhookConfig, event_type: str, loop_id: str) -> bool:
        """Check if an event matches a webhook config's filters."""
        if event_type not in cfg.event_types:
            return False
        if cfg.loop_ids is not None and loop_id not in cfg.loop_ids:
            return False
        return True

    # Cloud metadata endpoints to block (SSRF targets)
    BLOCKED_METADATA_IPS = frozenset(
        [
            # AWS
            "169.254.169.254",
            "fd00:ec2::254",
            # GCP
            "metadata.google.internal",
            # Azure
            "169.254.169.254",  # Same as AWS
            # DigitalOcean
            "169.254.169.254",  # Same as AWS
        ]
    )

    BLOCKED_METADATA_HOSTNAMES = frozenset(
        [
            "metadata.google.internal",
            "metadata.goog",
            "169.254.169.254",
            "instance-data",
        ]
    )

    def _validate_webhook_url(self, url: str) -> tuple[bool, str]:
        """Validate webhook URL is not pointing to internal services (SSRF protection).

        Blocks:
        - Non HTTP/HTTPS schemes
        - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
        - Loopback addresses (127.x, ::1)
        - Link-local addresses (169.254.x, fe80::)
        - Reserved addresses
        - Cloud metadata endpoints (AWS, GCP, Azure, etc.)

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return False, "Invalid URL format"

        # Only allow HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            return False, f"Only HTTP/HTTPS allowed, got: {parsed.scheme}"

        if not parsed.hostname:
            return False, "URL must have a hostname"

        hostname_lower = parsed.hostname.lower()

        # Block known metadata hostnames
        if hostname_lower in self.BLOCKED_METADATA_HOSTNAMES:
            return False, f"Blocked metadata hostname: {parsed.hostname}"

        # Block hostnames ending with internal suffixes
        blocked_suffixes = (".internal", ".local", ".localhost", ".lan")
        if any(hostname_lower.endswith(suffix) for suffix in blocked_suffixes):
            return False, f"Internal hostname not allowed: {parsed.hostname}"

        # Skip IP validation only for actual localhost (for testing)
        # Don't bypass ALL validation - only skip for specific localhost identifiers
        if self._allow_localhost and hostname_lower in ("localhost", "127.0.0.1", "::1"):
            return True, ""

        # Try to resolve hostname and check all returned IPs (IPv4 and IPv6)
        try:
            # Use getaddrinfo for both IPv4 and IPv6 resolution
            addr_info = socket.getaddrinfo(
                parsed.hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
            for family, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    ip_obj = ipaddress.ip_address(ip_str)
                except ValueError:
                    continue

                if ip_obj.is_private:
                    return False, f"Private IP not allowed: {ip_str}"
                if ip_obj.is_loopback:
                    return False, f"Loopback IP not allowed: {ip_str}"
                if ip_obj.is_link_local:
                    return False, f"Link-local IP not allowed: {ip_str}"
                if ip_obj.is_reserved:
                    return False, f"Reserved IP not allowed: {ip_str}"
                if ip_obj.is_multicast:
                    return False, f"Multicast IP not allowed: {ip_str}"
                if ip_obj.is_unspecified:
                    return False, f"Unspecified IP not allowed: {ip_str}"

                # Block cloud metadata endpoints
                if ip_str in self.BLOCKED_METADATA_IPS:
                    return False, f"Cloud metadata endpoint not allowed: {ip_str}"

        except socket.gaierror:
            # DNS resolution failed - let the request fail naturally
            pass
        except OSError:
            # Other socket errors - allow and let request handle it
            pass

        return True, ""

    def _worker_loop(self) -> None:
        """Background thread that delivers events to webhooks."""
        while not self._shutdown_event.is_set():
            try:
                event_dict = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            event_type = event_dict.get("type", "")
            loop_id = event_dict.get("loop_id", "")

            for cfg in self.configs:
                if self._matches_config(cfg, event_type, loop_id):
                    # Apply backpressure: skip if too many pending deliveries
                    if not self._delivery_semaphore.acquire(blocking=False):
                        with self._stats_lock:
                            self._drop_count += 1
                        logger.warning(
                            f"webhook_backpressure_drop url={cfg.url[:50]}... "
                            "executor queue full"
                        )
                        continue

                    # Submit to thread pool for parallel delivery
                    # This prevents one slow webhook from blocking others
                    self._executor.submit(self._deliver_and_track, cfg, event_dict)

    def _deliver_and_track(self, cfg: "WebhookConfig", event_dict: Dict[str, Any]) -> None:
        """Deliver webhook and update stats (runs in thread pool)."""
        try:
            success = self._deliver(cfg, event_dict)
            with self._stats_lock:
                if success:
                    self._delivery_count += 1
                else:
                    self._failure_count += 1
        finally:
            # Release semaphore to allow more deliveries
            self._delivery_semaphore.release()

    def _deliver(self, cfg: WebhookConfig, event_dict: Dict[str, Any]) -> bool:
        """Attempt delivery to a single webhook with retries and circuit breaker.

        Circuit breaker policy:
        - After 5 consecutive failures, circuit opens for 120 seconds
        - During open state, deliveries are skipped (fail-fast)
        - After cooldown, circuit enters half-open state for trial delivery

        Retry policy:
        - 2xx: Success, no retry
        - 429 (Too Many Requests): Retry with backoff, honor Retry-After if present
        - 5xx (Server Error): Retry with backoff
        - 4xx (other): No retry (client error, likely permanent)
        - Network errors: Retry with backoff

        Returns True on success, False on final failure.
        """
        # Circuit breaker check: fail-fast if endpoint is consistently failing
        circuit = get_circuit_breaker(
            f"webhook:{cfg.name}",
            failure_threshold=5,
            cooldown_seconds=120.0,
        )
        if not circuit.can_proceed():
            logger.debug(
                f"Webhook {cfg.name} circuit open, skipping delivery "
                f"for {event_dict.get('type')}"
            )
            return False

        # SSRF protection: validate URL before making request
        is_valid, error_msg = self._validate_webhook_url(cfg.url)
        if not is_valid:
            logger.warning(f"Webhook {cfg.name} blocked (SSRF): {error_msg}")
            return False

        # Use custom encoder for proper set/datetime handling
        body = json.dumps(event_dict, cls=AragoraJSONEncoder).encode("utf-8")
        signature = sign_payload(cfg.secret, body)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Aragora-Webhook/1.0",
            "X-Aragora-Event-Type": str(event_dict.get("type", "")),
            "X-Aragora-Loop-Id": str(event_dict.get("loop_id", "")),
            "X-Aragora-Event-Id": f"{event_dict.get('timestamp', int(time.time()))}-{hashlib.sha256(body).hexdigest()[:8]}",
        }
        if signature:
            headers["X-Aragora-Signature"] = signature

        for attempt in range(cfg.max_retries):
            try:
                req = urllib.request.Request(cfg.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    if 200 <= resp.status < 300:
                        circuit.record_success()
                        logger.debug(f"Webhook {cfg.name} delivered: {event_dict.get('type')}")
                        return True
                    # Non-2xx but not HTTPError means unexpected success-like response
                    logger.warning(f"Webhook {cfg.name} got unexpected status {resp.status}")

            except urllib.error.HTTPError as e:
                # Determine if we should retry
                if e.code == 429 or (500 <= e.code < 600):
                    # Retriable: 429 (rate limit) or 5xx (server error)
                    retry_after = e.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        backoff = min(float(retry_after), 60.0)  # Cap at 60s
                    else:
                        backoff = cfg.backoff_base_s * (2**attempt) + random.uniform(0, 0.5)

                    if attempt < cfg.max_retries - 1:
                        logger.debug(
                            f"Webhook {cfg.name} attempt {attempt+1} got {e.code}, "
                            f"retrying in {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                        continue
                else:
                    # Non-retriable 4xx (400, 401, 403, 404, etc.)
                    # Don't record as circuit failure - this is likely a config issue
                    logger.warning(
                        f"Webhook {cfg.name} failed with non-retriable error {e.code} "
                        f"for {event_dict.get('type')}"
                    )
                    return False

            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # Network errors: retry
                if attempt < cfg.max_retries - 1:
                    backoff = cfg.backoff_base_s * (2**attempt) + random.uniform(0, 0.5)
                    logger.debug(
                        f"Webhook {cfg.name} attempt {attempt+1} failed: {e}, "
                        f"retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue

        # All retries exhausted - record circuit failure
        circuit.record_failure()
        logger.warning(
            f"Webhook {cfg.name} delivery failed after {cfg.max_retries} attempts "
            f"for {event_dict.get('type')}"
        )
        return False

    @property
    def stats(self) -> Dict[str, Any]:
        """Return delivery statistics (thread-safe)."""
        with self._stats_lock:
            return {
                "queue_size": self._queue.qsize(),
                "delivered": self._delivery_count,
                "failed": self._failure_count,
                "dropped": self._drop_count,
            }

    def get_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all configured webhooks.

        Returns:
            Dict mapping webhook names to their circuit breaker status.
        """
        status: Dict[str, Dict[str, Any]] = {}
        for cfg in self.configs:
            circuit = get_circuit_breaker(
                f"webhook:{cfg.name}",
                failure_threshold=5,
                cooldown_seconds=120.0,
            )
            status[cfg.name] = {
                "status": circuit.get_status(),
                "failures": circuit.failures,
                "url": cfg.url[:50] + "..." if len(cfg.url) > 50 else cfg.url,
            }
        return status


# Module-level singleton
_dispatcher: Optional[WebhookDispatcher] = None


def get_dispatcher() -> Optional[WebhookDispatcher]:
    """Get the global webhook dispatcher (may be None if not configured)."""
    return _dispatcher


def init_dispatcher(configs: Optional[List[WebhookConfig]] = None) -> Optional[WebhookDispatcher]:
    """Initialize the global webhook dispatcher.

    If configs is None, loads from environment. Returns None if no configs.
    """
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher

    if configs is None:
        configs = load_webhook_configs()

    if not configs:
        logger.debug("No webhook configs found, dispatcher not started")
        return None

    _dispatcher = WebhookDispatcher(configs)
    _dispatcher.start()
    return _dispatcher


def shutdown_dispatcher() -> None:
    """Shutdown the global dispatcher gracefully."""
    global _dispatcher
    if _dispatcher:
        _dispatcher.stop()
        _dispatcher = None
