"""
Anomaly Detection for Security Monitoring.

Provides real-time anomaly detection for security events including:
- Authentication pattern analysis (failed logins, unusual times)
- Behavioral baseline learning and deviation detection
- Rate-based attack detection (brute force, DDoS)
- Geographic and IP anomalies
- User behavior profiling

SOC 2 Compliance: CC4.1, CC4.2 (Monitoring)

Usage:
    from aragora.security.anomaly_detection import (
        AnomalyDetector,
        get_anomaly_detector,
        check_auth_anomaly,
        check_rate_anomaly,
    )

    # Initialize detector with storage
    detector = AnomalyDetector(storage_path="anomaly_data.db")

    # Check for authentication anomaly
    anomaly = await detector.check_auth_event(
        user_id="user_123",
        ip_address="192.168.1.1",
        success=False,
    )
    if anomaly.is_anomalous:
        await alert_security_team(anomaly)
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Anomaly Types and Severity
# =============================================================================


class AnomalyType(Enum):
    """Types of security anomalies."""

    # Authentication anomalies
    AUTH_BRUTE_FORCE = "auth.brute_force"
    AUTH_CREDENTIAL_STUFFING = "auth.credential_stuffing"
    AUTH_IMPOSSIBLE_TRAVEL = "auth.impossible_travel"
    AUTH_UNUSUAL_TIME = "auth.unusual_time"
    AUTH_NEW_DEVICE = "auth.new_device"
    AUTH_FAILED_SPIKE = "auth.failed_spike"

    # Rate anomalies
    RATE_API_SPIKE = "rate.api_spike"
    RATE_REQUEST_FLOOD = "rate.request_flood"
    RATE_DATA_EXFILTRATION = "rate.data_exfiltration"

    # Behavioral anomalies
    BEHAVIOR_UNUSUAL_RESOURCE = "behavior.unusual_resource"
    BEHAVIOR_PRIVILEGE_ESCALATION = "behavior.privilege_escalation"
    BEHAVIOR_UNUSUAL_PATTERN = "behavior.unusual_pattern"

    # Network anomalies
    NETWORK_TOR_EXIT = "network.tor_exit"
    NETWORK_KNOWN_BAD_IP = "network.known_bad_ip"
    NETWORK_UNUSUAL_COUNTRY = "network.unusual_country"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Result of an anomaly check."""

    is_anomalous: bool
    anomaly_type: Optional[AnomalyType] = None
    severity: AnomalySeverity = AnomalySeverity.LOW
    confidence: float = 0.0
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomalous": self.is_anomalous,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "details": self.details,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection."""

    # Authentication thresholds
    failed_login_threshold: int = 5  # Failed logins before alert
    failed_login_window_minutes: int = 15
    brute_force_threshold: int = 10  # Attempts in window
    credential_stuffing_threshold: int = 3  # Different users from same IP

    # Rate thresholds
    api_spike_multiplier: float = 3.0  # X times normal rate
    request_flood_per_minute: int = 1000
    data_exfil_threshold_mb: float = 100.0  # MB in short period

    # Behavioral thresholds
    baseline_learning_days: int = 7
    unusual_time_std_deviations: float = 2.0

    # Network thresholds
    impossible_travel_speed_kmh: float = 1000.0  # Speed suggesting VPN/proxy

    # Storage
    storage_path: Optional[str] = None
    retention_days: int = 90


# =============================================================================
# Baseline Learning
# =============================================================================


@dataclass
class UserBaseline:
    """Learned baseline for a user's behavior."""

    user_id: str
    typical_login_hours: List[int] = field(default_factory=list)
    typical_ips: Set[str] = field(default_factory=set)
    typical_user_agents: Set[str] = field(default_factory=set)
    typical_resources: Set[str] = field(default_factory=set)
    avg_requests_per_hour: float = 0.0
    std_requests_per_hour: float = 0.0
    last_known_locations: List[Tuple[str, str, datetime]] = field(default_factory=list)
    learning_samples: int = 0
    last_updated: Optional[datetime] = None

    def is_mature(self, min_samples: int = 50) -> bool:
        """Check if baseline has enough samples for reliable detection."""
        return self.learning_samples >= min_samples


# =============================================================================
# Storage Layer
# =============================================================================


class AnomalyStorage:
    """SQLite-backed storage for anomaly detection data."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage.

        Args:
            db_path: Path to SQLite database. Uses in-memory if not provided.
        """
        self._db_path = db_path or ":memory:"
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS auth_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                success INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                country TEXT,
                city TEXT
            );

            CREATE TABLE IF NOT EXISTS api_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                ip_address TEXT,
                endpoint TEXT,
                method TEXT,
                timestamp TEXT NOT NULL,
                response_size_bytes INTEGER
            );

            CREATE TABLE IF NOT EXISTS user_baselines (
                user_id TEXT PRIMARY KEY,
                typical_login_hours_json TEXT,
                typical_ips_json TEXT,
                typical_user_agents_json TEXT,
                typical_resources_json TEXT,
                avg_requests_per_hour REAL,
                std_requests_per_hour REAL,
                learning_samples INTEGER DEFAULT 0,
                last_updated TEXT
            );

            CREATE TABLE IF NOT EXISTS detected_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT,
                ip_address TEXT,
                confidence REAL,
                description TEXT,
                details_json TEXT,
                timestamp TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT,
                resolved_by TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_auth_events_user_ts ON auth_events(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_auth_events_ip_ts ON auth_events(ip_address, timestamp);
            CREATE INDEX IF NOT EXISTS idx_api_requests_user_ts ON api_requests(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_api_requests_ip_ts ON api_requests(ip_address, timestamp);
            CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON detected_anomalies(timestamp);
            """
        )
        conn.commit()

    def record_auth_event(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
    ) -> None:
        """Record an authentication event."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO auth_events (user_id, ip_address, user_agent, success, timestamp, country, city)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                ip_address,
                user_agent,
                1 if success else 0,
                datetime.now(timezone.utc).isoformat(),
                country,
                city,
            ),
        )
        conn.commit()

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        response_size_bytes: int = 0,
    ) -> None:
        """Record an API request."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO api_requests (user_id, ip_address, endpoint, method, timestamp, response_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                ip_address,
                endpoint,
                method,
                datetime.now(timezone.utc).isoformat(),
                response_size_bytes,
            ),
        )
        conn.commit()

    def get_failed_logins_in_window(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        window_minutes: int = 15,
    ) -> int:
        """Get count of failed logins in time window."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

        if user_id:
            result = conn.execute(
                "SELECT COUNT(*) FROM auth_events WHERE user_id = ? AND success = 0 AND timestamp > ?",
                (user_id, cutoff),
            ).fetchone()
        elif ip_address:
            result = conn.execute(
                "SELECT COUNT(*) FROM auth_events WHERE ip_address = ? AND success = 0 AND timestamp > ?",
                (ip_address, cutoff),
            ).fetchone()
        else:
            return 0

        return result[0] if result else 0

    def get_distinct_failed_users_from_ip(self, ip_address: str, window_minutes: int = 15) -> int:
        """Get count of distinct users with failed logins from an IP."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

        result = conn.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM auth_events
            WHERE ip_address = ? AND success = 0 AND timestamp > ?
            """,
            (ip_address, cutoff),
        ).fetchone()

        return result[0] if result else 0

    def get_requests_in_window(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        window_minutes: int = 60,
    ) -> int:
        """Get request count in time window."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

        if user_id:
            result = conn.execute(
                "SELECT COUNT(*) FROM api_requests WHERE user_id = ? AND timestamp > ?",
                (user_id, cutoff),
            ).fetchone()
        elif ip_address:
            result = conn.execute(
                "SELECT COUNT(*) FROM api_requests WHERE ip_address = ? AND timestamp > ?",
                (ip_address, cutoff),
            ).fetchone()
        else:
            return 0

        return result[0] if result else 0

    def get_data_transferred_in_window(
        self,
        user_id: str,
        window_minutes: int = 60,
    ) -> int:
        """Get total data transferred in bytes."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

        result = conn.execute(
            """
            SELECT COALESCE(SUM(response_size_bytes), 0)
            FROM api_requests
            WHERE user_id = ? AND timestamp > ?
            """,
            (user_id, cutoff),
        ).fetchone()

        return result[0] if result else 0

    def get_user_login_hours(self, user_id: str, days: int = 30) -> List[int]:
        """Get login hours for a user over the past N days."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = conn.execute(
            """
            SELECT timestamp FROM auth_events
            WHERE user_id = ? AND success = 1 AND timestamp > ?
            """,
            (user_id, cutoff),
        ).fetchall()

        hours = []
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                hours.append(ts.hour)
            except (ValueError, TypeError):
                pass

        return hours

    def save_baseline(self, baseline: UserBaseline) -> None:
        """Save or update a user baseline."""
        import json

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO user_baselines (
                user_id, typical_login_hours_json, typical_ips_json,
                typical_user_agents_json, typical_resources_json,
                avg_requests_per_hour, std_requests_per_hour,
                learning_samples, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                baseline.user_id,
                json.dumps(baseline.typical_login_hours),
                json.dumps(list(baseline.typical_ips)),
                json.dumps(list(baseline.typical_user_agents)),
                json.dumps(list(baseline.typical_resources)),
                baseline.avg_requests_per_hour,
                baseline.std_requests_per_hour,
                baseline.learning_samples,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    def get_baseline(self, user_id: str) -> Optional[UserBaseline]:
        """Get a user's baseline."""
        import json

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM user_baselines WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if not row:
            return None

        return UserBaseline(
            user_id=row["user_id"],
            typical_login_hours=json.loads(row["typical_login_hours_json"] or "[]"),
            typical_ips=set(json.loads(row["typical_ips_json"] or "[]")),
            typical_user_agents=set(json.loads(row["typical_user_agents_json"] or "[]")),
            typical_resources=set(json.loads(row["typical_resources_json"] or "[]")),
            avg_requests_per_hour=row["avg_requests_per_hour"] or 0.0,
            std_requests_per_hour=row["std_requests_per_hour"] or 0.0,
            learning_samples=row["learning_samples"] or 0,
            last_updated=datetime.fromisoformat(row["last_updated"])
            if row["last_updated"]
            else None,
        )

    def record_anomaly(
        self, result: AnomalyResult, user_id: Optional[str] = None, ip_address: Optional[str] = None
    ) -> int:
        """Record a detected anomaly."""
        import json

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO detected_anomalies (
                anomaly_type, severity, user_id, ip_address,
                confidence, description, details_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.anomaly_type.value if result.anomaly_type else "unknown",
                result.severity.value,
                user_id,
                ip_address,
                result.confidence,
                result.description,
                json.dumps(result.details),
                result.timestamp,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_recent_anomalies(
        self,
        hours: int = 24,
        severity: Optional[AnomalySeverity] = None,
        unresolved_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get recent anomalies."""
        import json

        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        query = "SELECT * FROM detected_anomalies WHERE timestamp > ?"
        params: List[Any] = [cutoff]

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if unresolved_only:
            query += " AND resolved = 0"

        query += " ORDER BY timestamp DESC"

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "anomaly_type": row["anomaly_type"],
                "severity": row["severity"],
                "user_id": row["user_id"],
                "ip_address": row["ip_address"],
                "confidence": row["confidence"],
                "description": row["description"],
                "details": json.loads(row["details_json"] or "{}"),
                "timestamp": row["timestamp"],
                "resolved": bool(row["resolved"]),
            }
            for row in rows
        ]

    def cleanup_old_data(self, retention_days: int = 90) -> int:
        """Clean up data older than retention period."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

        deleted = 0
        for table in ["auth_events", "api_requests", "detected_anomalies"]:
            result = conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
            deleted += result.rowcount

        conn.commit()
        return deleted


# =============================================================================
# Anomaly Detector
# =============================================================================


class AnomalyDetector:
    """Main anomaly detection engine."""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        """Initialize detector.

        Args:
            config: Detection configuration
        """
        self.config = config or AnomalyDetectorConfig()
        self._storage = AnomalyStorage(self.config.storage_path)
        self._baselines: Dict[str, UserBaseline] = {}
        self._lock = threading.Lock()

        # In-memory rate tracking (for high-frequency checks)
        self._request_counts: Dict[str, List[datetime]] = defaultdict(list)
        self._last_cleanup = datetime.now(timezone.utc)

    # =========================================================================
    # Authentication Anomaly Detection
    # =========================================================================

    async def check_auth_event(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
    ) -> AnomalyResult:
        """Check an authentication event for anomalies.

        Args:
            user_id: User ID
            success: Whether login succeeded
            ip_address: Client IP address
            user_agent: Client user agent
            country: Detected country (from IP geolocation)
            city: Detected city

        Returns:
            Anomaly result with detection details
        """
        # Record event for future baseline learning
        self._storage.record_auth_event(
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            country=country,
            city=city,
        )

        # Only check for anomalies on failed logins (most attacks) or suspicious success
        if not success:
            # Check for brute force on user
            user_failures = self._storage.get_failed_logins_in_window(
                user_id=user_id,
                window_minutes=self.config.failed_login_window_minutes,
            )

            if user_failures >= self.config.brute_force_threshold:
                result = AnomalyResult(
                    is_anomalous=True,
                    anomaly_type=AnomalyType.AUTH_BRUTE_FORCE,
                    severity=AnomalySeverity.HIGH,
                    confidence=min(
                        0.5 + (user_failures - self.config.brute_force_threshold) * 0.1, 0.99
                    ),
                    description=f"Brute force attack detected: {user_failures} failed logins for user {user_id}",
                    details={
                        "failed_count": user_failures,
                        "window_minutes": self.config.failed_login_window_minutes,
                        "threshold": self.config.brute_force_threshold,
                    },
                    recommendations=[
                        "Consider temporarily locking the account",
                        "Require MFA for next successful login",
                        "Review IP addresses attempting access",
                    ],
                )
                self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                return result

            # Check for credential stuffing (multiple users from same IP)
            if ip_address:
                distinct_users = self._storage.get_distinct_failed_users_from_ip(
                    ip_address=ip_address,
                    window_minutes=self.config.failed_login_window_minutes,
                )

                if distinct_users >= self.config.credential_stuffing_threshold:
                    result = AnomalyResult(
                        is_anomalous=True,
                        anomaly_type=AnomalyType.AUTH_CREDENTIAL_STUFFING,
                        severity=AnomalySeverity.CRITICAL,
                        confidence=min(0.6 + distinct_users * 0.1, 0.99),
                        description=f"Credential stuffing attack: {distinct_users} users targeted from {ip_address}",
                        details={
                            "distinct_users": distinct_users,
                            "ip_address": ip_address,
                            "window_minutes": self.config.failed_login_window_minutes,
                        },
                        recommendations=[
                            "Block IP address immediately",
                            "Notify affected users",
                            "Review all accounts for compromise",
                        ],
                    )
                    self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                    return result

            # Check for failed login spike
            if user_failures >= self.config.failed_login_threshold:
                result = AnomalyResult(
                    is_anomalous=True,
                    anomaly_type=AnomalyType.AUTH_FAILED_SPIKE,
                    severity=AnomalySeverity.MEDIUM,
                    confidence=0.7,
                    description=f"Failed login spike: {user_failures} failures for {user_id}",
                    details={
                        "failed_count": user_failures,
                        "threshold": self.config.failed_login_threshold,
                    },
                    recommendations=[
                        "Monitor for continued attempts",
                        "Consider requiring password reset",
                    ],
                )
                self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                return result

        # On successful login, check behavioral anomalies
        if success:
            baseline = self._get_or_create_baseline(user_id)

            # Check for unusual login time
            if baseline.is_mature():
                current_hour = datetime.now(timezone.utc).hour
                if baseline.typical_login_hours:
                    mean_hour = statistics.mean(baseline.typical_login_hours)
                    std_hour = (
                        statistics.stdev(baseline.typical_login_hours)
                        if len(baseline.typical_login_hours) > 1
                        else 4.0
                    )

                    if std_hour > 0:
                        z_score = abs(current_hour - mean_hour) / std_hour
                        if z_score > self.config.unusual_time_std_deviations:
                            result = AnomalyResult(
                                is_anomalous=True,
                                anomaly_type=AnomalyType.AUTH_UNUSUAL_TIME,
                                severity=AnomalySeverity.LOW,
                                confidence=min(0.5 + z_score * 0.1, 0.9),
                                description=f"Login at unusual time ({current_hour}:00) for user {user_id}",
                                details={
                                    "current_hour": current_hour,
                                    "typical_mean_hour": round(mean_hour, 1),
                                    "z_score": round(z_score, 2),
                                },
                                recommendations=[
                                    "Verify this was the actual user",
                                    "Consider additional verification step",
                                ],
                            )
                            self._storage.record_anomaly(
                                result, user_id=user_id, ip_address=ip_address
                            )
                            return result

            # Check for new device/IP
            if ip_address and baseline.typical_ips and ip_address not in baseline.typical_ips:
                result = AnomalyResult(
                    is_anomalous=True,
                    anomaly_type=AnomalyType.AUTH_NEW_DEVICE,
                    severity=AnomalySeverity.LOW,
                    confidence=0.6,
                    description=f"Login from new IP address {ip_address} for user {user_id}",
                    details={
                        "new_ip": ip_address,
                        "known_ips": list(baseline.typical_ips)[:5],
                    },
                    recommendations=[
                        "Send notification to user about new device",
                        "Consider MFA challenge",
                    ],
                )
                self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                return result

            # Update baseline with successful login
            self._update_baseline(user_id, ip_address, user_agent)

        return AnomalyResult(is_anomalous=False)

    # =========================================================================
    # Rate Anomaly Detection
    # =========================================================================

    async def check_rate_anomaly(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: str = "GET",
        response_size_bytes: int = 0,
    ) -> AnomalyResult:
        """Check for rate-based anomalies.

        Args:
            user_id: User ID (if authenticated)
            ip_address: Client IP
            endpoint: API endpoint accessed
            method: HTTP method
            response_size_bytes: Response size in bytes

        Returns:
            Anomaly result
        """
        # Record request
        self._storage.record_api_request(
            endpoint=endpoint or "",
            method=method,
            user_id=user_id,
            ip_address=ip_address,
            response_size_bytes=response_size_bytes,
        )

        # Track in-memory for high-frequency detection
        key = f"{user_id or 'anon'}:{ip_address or 'unknown'}"
        now = datetime.now(timezone.utc)

        with self._lock:
            # Cleanup old entries periodically
            if (now - self._last_cleanup).seconds > 60:
                self._cleanup_rate_tracking()
                self._last_cleanup = now

            self._request_counts[key].append(now)

            # Count requests in last minute
            minute_ago = now - timedelta(minutes=1)
            recent_requests = [t for t in self._request_counts[key] if t > minute_ago]
            self._request_counts[key] = recent_requests  # Cleanup old

        request_count = len(recent_requests)

        # Check for request flood
        if request_count > self.config.request_flood_per_minute:
            result = AnomalyResult(
                is_anomalous=True,
                anomaly_type=AnomalyType.RATE_REQUEST_FLOOD,
                severity=AnomalySeverity.HIGH,
                confidence=min(
                    0.7 + (request_count / self.config.request_flood_per_minute - 1) * 0.1, 0.99
                ),
                description=f"Request flood detected: {request_count} requests/min from {ip_address or user_id}",
                details={
                    "request_count": request_count,
                    "threshold": self.config.request_flood_per_minute,
                    "ip_address": ip_address,
                    "user_id": user_id,
                },
                recommendations=[
                    "Apply rate limiting immediately",
                    "Consider blocking IP temporarily",
                    "Investigate for DDoS attack",
                ],
            )
            self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
            return result

        # Check for API spike vs baseline (for authenticated users)
        if user_id:
            baseline = self._get_or_create_baseline(user_id)
            if baseline.is_mature() and baseline.avg_requests_per_hour > 0:
                hourly_rate = request_count * 60  # Extrapolate to hourly

                if hourly_rate > baseline.avg_requests_per_hour * self.config.api_spike_multiplier:
                    result = AnomalyResult(
                        is_anomalous=True,
                        anomaly_type=AnomalyType.RATE_API_SPIKE,
                        severity=AnomalySeverity.MEDIUM,
                        confidence=0.75,
                        description=f"API usage spike for {user_id}: {hourly_rate}/hr vs typical {baseline.avg_requests_per_hour:.0f}/hr",
                        details={
                            "current_rate": hourly_rate,
                            "baseline_rate": baseline.avg_requests_per_hour,
                            "multiplier": round(hourly_rate / baseline.avg_requests_per_hour, 1),
                        },
                        recommendations=[
                            "Review recent activity for automation",
                            "Check for API key compromise",
                        ],
                    )
                    self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                    return result

            # Check for data exfiltration
            data_transferred = self._storage.get_data_transferred_in_window(
                user_id=user_id,
                window_minutes=60,
            )
            data_mb = data_transferred / (1024 * 1024)

            if data_mb > self.config.data_exfil_threshold_mb:
                result = AnomalyResult(
                    is_anomalous=True,
                    anomaly_type=AnomalyType.RATE_DATA_EXFILTRATION,
                    severity=AnomalySeverity.CRITICAL,
                    confidence=0.85,
                    description=f"Potential data exfiltration: {data_mb:.1f}MB transferred by {user_id}",
                    details={
                        "data_transferred_mb": round(data_mb, 2),
                        "threshold_mb": self.config.data_exfil_threshold_mb,
                    },
                    recommendations=[
                        "Immediately investigate user activity",
                        "Consider suspending account",
                        "Review what data was accessed",
                    ],
                )
                self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                return result

        return AnomalyResult(is_anomalous=False)

    # =========================================================================
    # Behavioral Anomaly Detection
    # =========================================================================

    async def check_behavioral_anomaly(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> AnomalyResult:
        """Check for behavioral anomalies.

        Args:
            user_id: User ID
            resource_type: Type of resource accessed
            resource_id: Resource ID
            action: Action performed

        Returns:
            Anomaly result
        """
        baseline = self._get_or_create_baseline(user_id)

        # Check for unusual resource access
        resource_key = f"{resource_type}:{action}"
        if baseline.is_mature() and baseline.typical_resources:
            if resource_key not in baseline.typical_resources:
                result = AnomalyResult(
                    is_anomalous=True,
                    anomaly_type=AnomalyType.BEHAVIOR_UNUSUAL_RESOURCE,
                    severity=AnomalySeverity.LOW,
                    confidence=0.6,
                    description=f"Unusual resource access: {user_id} accessing {resource_type} (action: {action})",
                    details={
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "action": action,
                        "typical_resources": list(baseline.typical_resources)[:10],
                    },
                    recommendations=[
                        "Verify this is expected behavior",
                        "Update user permissions if legitimate",
                    ],
                )
                self._storage.record_anomaly(result, user_id=user_id)
                return result

        # Update baseline
        baseline.typical_resources.add(resource_key)
        baseline.learning_samples += 1
        self._storage.save_baseline(baseline)

        return AnomalyResult(is_anomalous=False)

    # =========================================================================
    # Network Anomaly Detection
    # =========================================================================

    async def check_network_anomaly(
        self,
        ip_address: str,
        user_id: Optional[str] = None,
        is_tor_exit: bool = False,
        is_known_bad_ip: bool = False,
        country: Optional[str] = None,
    ) -> AnomalyResult:
        """Check for network-based anomalies.

        Args:
            ip_address: Client IP address
            user_id: User ID if authenticated
            is_tor_exit: Whether IP is a Tor exit node
            is_known_bad_ip: Whether IP is on a blocklist
            country: Detected country

        Returns:
            Anomaly result
        """
        # Check Tor exit
        if is_tor_exit:
            result = AnomalyResult(
                is_anomalous=True,
                anomaly_type=AnomalyType.NETWORK_TOR_EXIT,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.95,
                description=f"Connection from Tor exit node: {ip_address}",
                details={"ip_address": ip_address, "is_tor_exit": True},
                recommendations=[
                    "Require additional verification",
                    "Consider blocking Tor exits if policy requires",
                ],
            )
            self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
            return result

        # Check known bad IP
        if is_known_bad_ip:
            result = AnomalyResult(
                is_anomalous=True,
                anomaly_type=AnomalyType.NETWORK_KNOWN_BAD_IP,
                severity=AnomalySeverity.HIGH,
                confidence=0.9,
                description=f"Connection from known malicious IP: {ip_address}",
                details={"ip_address": ip_address, "is_blocklisted": True},
                recommendations=[
                    "Block this IP immediately",
                    "Investigate any successful connections",
                ],
            )
            self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
            return result

        # Check unusual country for authenticated user
        if user_id and country:
            baseline = self._get_or_create_baseline(user_id)
            if baseline.last_known_locations:
                known_countries = {loc[0] for loc in baseline.last_known_locations}
                if country not in known_countries:
                    result = AnomalyResult(
                        is_anomalous=True,
                        anomaly_type=AnomalyType.NETWORK_UNUSUAL_COUNTRY,
                        severity=AnomalySeverity.MEDIUM,
                        confidence=0.7,
                        description=f"Login from unusual country ({country}) for user {user_id}",
                        details={
                            "country": country,
                            "known_countries": list(known_countries),
                            "ip_address": ip_address,
                        },
                        recommendations=[
                            "Verify user's travel status",
                            "Consider step-up authentication",
                        ],
                    )
                    self._storage.record_anomaly(result, user_id=user_id, ip_address=ip_address)
                    return result

        return AnomalyResult(is_anomalous=False)

    # =========================================================================
    # Baseline Management
    # =========================================================================

    def _get_or_create_baseline(self, user_id: str) -> UserBaseline:
        """Get or create a user baseline."""
        with self._lock:
            if user_id in self._baselines:
                return self._baselines[user_id]

            # Try loading from storage
            baseline = self._storage.get_baseline(user_id)
            if baseline:
                self._baselines[user_id] = baseline
                return baseline

            # Create new baseline
            baseline = UserBaseline(user_id=user_id)
            self._baselines[user_id] = baseline
            return baseline

    def _update_baseline(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Update a user's baseline with new data."""
        baseline = self._get_or_create_baseline(user_id)

        # Update login hours
        current_hour = datetime.now(timezone.utc).hour
        baseline.typical_login_hours.append(current_hour)
        if len(baseline.typical_login_hours) > 100:
            baseline.typical_login_hours = baseline.typical_login_hours[-100:]

        # Update known IPs
        if ip_address:
            baseline.typical_ips.add(ip_address)
            if len(baseline.typical_ips) > 20:
                # Keep only most recent
                baseline.typical_ips = set(list(baseline.typical_ips)[-20:])

        # Update known user agents
        if user_agent:
            baseline.typical_user_agents.add(user_agent)
            if len(baseline.typical_user_agents) > 10:
                baseline.typical_user_agents = set(list(baseline.typical_user_agents)[-10:])

        # Update request rate baseline (from storage)
        hourly_counts = []
        for i in range(24):
            count = self._storage.get_requests_in_window(user_id=user_id, window_minutes=60)
            hourly_counts.append(count)

        if hourly_counts:
            baseline.avg_requests_per_hour = statistics.mean(hourly_counts)
            if len(hourly_counts) > 1:
                baseline.std_requests_per_hour = statistics.stdev(hourly_counts)

        baseline.learning_samples += 1
        baseline.last_updated = datetime.now(timezone.utc)

        # Persist
        self._storage.save_baseline(baseline)

    def _cleanup_rate_tracking(self) -> None:
        """Clean up old rate tracking data."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        for key in list(self._request_counts.keys()):
            self._request_counts[key] = [t for t in self._request_counts[key] if t > cutoff]
            if not self._request_counts[key]:
                del self._request_counts[key]

    # =========================================================================
    # Administration
    # =========================================================================

    def get_recent_anomalies(
        self,
        hours: int = 24,
        severity: Optional[AnomalySeverity] = None,
        unresolved_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get recent detected anomalies."""
        return self._storage.get_recent_anomalies(
            hours=hours,
            severity=severity,
            unresolved_only=unresolved_only,
        )

    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        recent = self._storage.get_recent_anomalies(hours=24)
        by_type: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)

        for anomaly in recent:
            by_type[anomaly["anomaly_type"]] += 1
            by_severity[anomaly["severity"]] += 1

        return {
            "total_24h": len(recent),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "unresolved": len([a for a in recent if not a["resolved"]]),
        }

    def cleanup(self) -> int:
        """Clean up old data according to retention policy."""
        return self._storage.cleanup_old_data(self.config.retention_days)


# =============================================================================
# Global Instance & Convenience Functions
# =============================================================================

_detector: Optional[AnomalyDetector] = None
_detector_lock = threading.Lock()


def get_anomaly_detector(config: Optional[AnomalyDetectorConfig] = None) -> AnomalyDetector:
    """Get or create the global anomaly detector."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = AnomalyDetector(config)
        return _detector


async def check_auth_anomaly(
    user_id: str,
    success: bool,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    country: Optional[str] = None,
) -> AnomalyResult:
    """Convenience function to check for auth anomalies."""
    return await get_anomaly_detector().check_auth_event(
        user_id=user_id,
        success=success,
        ip_address=ip_address,
        user_agent=user_agent,
        country=country,
    )


async def check_rate_anomaly(
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    endpoint: Optional[str] = None,
    response_size_bytes: int = 0,
) -> AnomalyResult:
    """Convenience function to check for rate anomalies."""
    return await get_anomaly_detector().check_rate_anomaly(
        user_id=user_id,
        ip_address=ip_address,
        endpoint=endpoint,
        response_size_bytes=response_size_bytes,
    )


__all__ = [
    # Types
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyResult",
    # Configuration
    "AnomalyDetectorConfig",
    # Core
    "AnomalyDetector",
    "get_anomaly_detector",
    # Convenience functions
    "check_auth_anomaly",
    "check_rate_anomaly",
]
