"""
Repositories for Phase 2 features: Inbox, Codebase Security, PR Review.

Provides SQLite-backed persistence for:
- Email prioritization results and inbox state
- Security scan results and vulnerability findings
- Code metrics analysis results
- PR review history and comments

Usage:
    from aragora.persistence.repositories.phase2 import (
        InboxRepository,
        SecurityScanRepository,
        PRReviewRepository,
    )

    # Initialize repositories
    inbox_repo = InboxRepository()
    security_repo = SecurityScanRepository()
    pr_repo = PRReviewRepository()

    # Store and retrieve data
    inbox_repo.save_prioritization(user_id, email_id, priority_result)
    scans = security_repo.get_recent_scans(repo_id, limit=10)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.persistence.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


# =============================================================================
# Inbox Repository
# =============================================================================


class InboxRepository(BaseRepository[Dict[str, Any]]):
    """
    Repository for inbox prioritization and state management.

    Stores:
    - Email prioritization results (priority, confidence, reasoning)
    - User actions on emails (archive, snooze, etc.)
    - Sender statistics and VIP lists
    """

    def __init__(self, db_path: str | Path = "inbox.db") -> None:
        super().__init__(db_path)

    def _get_schema(self) -> str:
        """Return the SQL schema for inbox tables."""
        return """
        -- Email prioritization results
        CREATE TABLE IF NOT EXISTS email_priorities (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            email_id TEXT NOT NULL,
            priority TEXT NOT NULL,
            confidence REAL NOT NULL,
            reasoning TEXT,
            tier_used TEXT,
            scores TEXT,  -- JSON blob of score breakdown
            suggested_labels TEXT,  -- JSON array
            auto_archive INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, email_id)
        );

        CREATE INDEX IF NOT EXISTS idx_priorities_user
            ON email_priorities(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_priorities_priority
            ON email_priorities(user_id, priority);

        -- User actions on emails (for learning)
        CREATE TABLE IF NOT EXISTS email_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            email_id TEXT NOT NULL,
            action TEXT NOT NULL,
            params TEXT,  -- JSON blob of action parameters
            predicted_priority TEXT,
            actual_feedback TEXT,  -- User correction if any
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_actions_user
            ON email_actions(user_id, created_at DESC);

        -- Sender statistics
        CREATE TABLE IF NOT EXISTS sender_stats (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            sender_email TEXT NOT NULL,
            display_name TEXT,
            is_vip INTEGER DEFAULT 0,
            is_internal INTEGER DEFAULT 0,
            total_emails INTEGER DEFAULT 0,
            response_rate REAL DEFAULT 0.0,
            avg_response_time_hours REAL,
            last_interaction TEXT,
            metadata TEXT,  -- JSON blob
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, sender_email)
        );

        CREATE INDEX IF NOT EXISTS idx_sender_user
            ON sender_stats(user_id);
        """

    def _to_entity(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        result = dict(row)
        # Parse JSON fields
        if "scores" in result and result["scores"]:
            result["scores"] = json.loads(result["scores"])
        if "suggested_labels" in result and result["suggested_labels"]:
            result["suggested_labels"] = json.loads(result["suggested_labels"])
        if "params" in result and result["params"]:
            result["params"] = json.loads(result["params"])
        if "metadata" in result and result["metadata"]:
            result["metadata"] = json.loads(result["metadata"])
        return result

    def _from_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an entity to database row format."""
        result = dict(entity)
        # Serialize JSON fields
        if "scores" in result and result["scores"]:
            result["scores"] = json.dumps(result["scores"])
        if "suggested_labels" in result and result["suggested_labels"]:
            result["suggested_labels"] = json.dumps(result["suggested_labels"])
        if "params" in result and result["params"]:
            result["params"] = json.dumps(result["params"])
        if "metadata" in result and result["metadata"]:
            result["metadata"] = json.dumps(result["metadata"])
        return result

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript(self._get_schema())

    def save_prioritization(
        self,
        user_id: str,
        email_id: str,
        priority: str,
        confidence: float,
        reasoning: Optional[str] = None,
        tier_used: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None,
        suggested_labels: Optional[List[str]] = None,
        auto_archive: bool = False,
    ) -> str:
        """Save email prioritization result."""
        import uuid

        record_id = f"pri_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO email_priorities
                (id, user_id, email_id, priority, confidence, reasoning,
                 tier_used, scores, suggested_labels, auto_archive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    user_id,
                    email_id,
                    priority,
                    confidence,
                    reasoning,
                    tier_used,
                    json.dumps(scores) if scores else None,
                    json.dumps(suggested_labels) if suggested_labels else None,
                    1 if auto_archive else 0,
                    now,
                    now,
                ),
            )
            return record_id

    def get_prioritization(self, user_id: str, email_id: str) -> Optional[Dict[str, Any]]:
        """Get prioritization for a specific email."""
        with self._connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM email_priorities WHERE user_id = ? AND email_id = ?",
                (user_id, email_id),
            ).fetchone()
            return self._to_entity(row) if row else None

    def get_recent_priorities(
        self,
        user_id: str,
        limit: int = 50,
        priority_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent prioritizations for a user."""
        with self._connection(readonly=True) as conn:
            if priority_filter:
                rows = conn.execute(
                    """
                    SELECT * FROM email_priorities
                    WHERE user_id = ? AND priority = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, priority_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM email_priorities
                    WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            return [self._to_entity(row) for row in rows]

    def record_action(
        self,
        user_id: str,
        email_id: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        predicted_priority: Optional[str] = None,
        actual_feedback: Optional[str] = None,
    ) -> None:
        """Record a user action on an email."""
        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO email_actions
                (user_id, email_id, action, params, predicted_priority, actual_feedback, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email_id,
                    action,
                    json.dumps(params) if params else None,
                    predicted_priority,
                    actual_feedback,
                    now,
                ),
            )


# =============================================================================
# Security Scan Repository
# =============================================================================


class SecurityScanRepository(BaseRepository[Dict[str, Any]]):
    """
    Repository for security scan results and vulnerability findings.

    Stores:
    - Dependency scan results
    - Vulnerability findings (CVE matches)
    - Secrets detection results
    """

    def __init__(self, db_path: str | Path = "security_scans.db") -> None:
        super().__init__(db_path)

    def _get_schema(self) -> str:
        """Return the SQL schema for security scan tables."""
        return """
        -- Security scan results
        CREATE TABLE IF NOT EXISTS security_scans (
            id TEXT PRIMARY KEY,
            repository TEXT NOT NULL,
            branch TEXT,
            commit_sha TEXT,
            scan_type TEXT NOT NULL,  -- dependency, secrets, full
            status TEXT NOT NULL,  -- running, completed, failed
            started_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT,
            summary TEXT,  -- JSON blob with counts
            metadata TEXT  -- JSON blob
        );

        CREATE INDEX IF NOT EXISTS idx_scans_repo
            ON security_scans(repository, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_scans_status
            ON security_scans(status);

        -- Vulnerability findings
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            cve_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            severity TEXT NOT NULL,
            cvss_score REAL,
            package_name TEXT,
            package_ecosystem TEXT,
            vulnerable_versions TEXT,  -- JSON array
            patched_versions TEXT,  -- JSON array
            file_path TEXT,
            line_number INTEGER,
            fix_available INTEGER DEFAULT 0,
            recommended_version TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES security_scans(id)
        );

        CREATE INDEX IF NOT EXISTS idx_vulns_scan
            ON vulnerabilities(scan_id);
        CREATE INDEX IF NOT EXISTS idx_vulns_severity
            ON vulnerabilities(severity);
        CREATE INDEX IF NOT EXISTS idx_vulns_cve
            ON vulnerabilities(cve_id);

        -- Secrets findings
        CREATE TABLE IF NOT EXISTS secrets_findings (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            secret_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            matched_text TEXT NOT NULL,  -- Redacted
            context_line TEXT,
            severity TEXT NOT NULL,
            confidence REAL,
            is_in_history INTEGER DEFAULT 0,
            commit_sha TEXT,
            remediation TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES security_scans(id)
        );

        CREATE INDEX IF NOT EXISTS idx_secrets_scan
            ON secrets_findings(scan_id);
        CREATE INDEX IF NOT EXISTS idx_secrets_type
            ON secrets_findings(secret_type);

        -- Code metrics results
        CREATE TABLE IF NOT EXISTS metrics_results (
            id TEXT PRIMARY KEY,
            repository TEXT NOT NULL,
            scan_id TEXT,
            file_path TEXT,
            total_lines INTEGER,
            code_lines INTEGER,
            comment_lines INTEGER,
            blank_lines INTEGER,
            complexity REAL,
            cognitive_complexity REAL,
            maintainability_index REAL,
            function_metrics TEXT,  -- JSON blob
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_repo
            ON metrics_results(repository, created_at DESC);
        """

    def _to_entity(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        result = dict(row)
        # Parse JSON fields
        for field in [
            "summary",
            "metadata",
            "vulnerable_versions",
            "patched_versions",
            "function_metrics",
        ]:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return result

    def _from_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an entity to database row format."""
        result = dict(entity)
        for field in [
            "summary",
            "metadata",
            "vulnerable_versions",
            "patched_versions",
            "function_metrics",
        ]:
            if field in result and result[field] and not isinstance(result[field], str):
                result[field] = json.dumps(result[field])
        return result

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript(self._get_schema())

    def save_scan(
        self,
        scan_id: str,
        repository: str,
        scan_type: str,
        status: str,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
        error: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save or update a scan result."""
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO security_scans
                (id, repository, branch, commit_sha, scan_type, status,
                 started_at, completed_at, error, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    repository,
                    branch,
                    commit_sha,
                    scan_type,
                    status,
                    now,
                    now if status in ("completed", "failed") else None,
                    error,
                    json.dumps(summary) if summary else None,
                ),
            )
            return scan_id

    def get_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scan by ID."""
        with self._connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM security_scans WHERE id = ?",
                (scan_id,),
            ).fetchone()
            return self._to_entity(row) if row else None

    def get_recent_scans(
        self,
        repository: str,
        limit: int = 10,
        scan_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent scans for a repository."""
        with self._connection(readonly=True) as conn:
            if scan_type:
                rows = conn.execute(
                    """
                    SELECT * FROM security_scans
                    WHERE repository = ? AND scan_type = ?
                    ORDER BY started_at DESC LIMIT ?
                    """,
                    (repository, scan_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM security_scans
                    WHERE repository = ?
                    ORDER BY started_at DESC LIMIT ?
                    """,
                    (repository, limit),
                ).fetchall()
            return [self._to_entity(row) for row in rows]

    def save_vulnerability(
        self,
        vuln_id: str,
        scan_id: str,
        title: str,
        severity: str,
        cve_id: Optional[str] = None,
        description: Optional[str] = None,
        cvss_score: Optional[float] = None,
        package_name: Optional[str] = None,
        package_ecosystem: Optional[str] = None,
        vulnerable_versions: Optional[List[str]] = None,
        patched_versions: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        fix_available: bool = False,
        recommended_version: Optional[str] = None,
    ) -> str:
        """Save a vulnerability finding."""
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vulnerabilities
                (id, scan_id, cve_id, title, description, severity, cvss_score,
                 package_name, package_ecosystem, vulnerable_versions, patched_versions,
                 file_path, line_number, fix_available, recommended_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vuln_id,
                    scan_id,
                    cve_id,
                    title,
                    description,
                    severity,
                    cvss_score,
                    package_name,
                    package_ecosystem,
                    json.dumps(vulnerable_versions) if vulnerable_versions else None,
                    json.dumps(patched_versions) if patched_versions else None,
                    file_path,
                    line_number,
                    1 if fix_available else 0,
                    recommended_version,
                    now,
                ),
            )
            return vuln_id

    def get_vulnerabilities(
        self,
        scan_id: str,
        severity_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get vulnerabilities for a scan."""
        with self._connection(readonly=True) as conn:
            if severity_filter:
                rows = conn.execute(
                    "SELECT * FROM vulnerabilities WHERE scan_id = ? AND severity = ?",
                    (scan_id, severity_filter),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM vulnerabilities WHERE scan_id = ?",
                    (scan_id,),
                ).fetchall()
            return [self._to_entity(row) for row in rows]


# =============================================================================
# PR Review Repository
# =============================================================================


class PRReviewRepository(BaseRepository[Dict[str, Any]]):
    """
    Repository for PR review results and comments.

    Stores:
    - Automated review results
    - Review comments and suggestions
    - Review history per PR
    """

    def __init__(self, db_path: str | Path = "pr_reviews.db") -> None:
        super().__init__(db_path)

    def _get_schema(self) -> str:
        """Return the SQL schema for PR review tables."""
        return """
        -- PR review results
        CREATE TABLE IF NOT EXISTS pr_reviews (
            id TEXT PRIMARY KEY,
            repository TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            status TEXT NOT NULL,  -- pending, in_progress, completed, failed
            verdict TEXT,  -- APPROVE, REQUEST_CHANGES, COMMENT
            summary TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT,
            metrics TEXT,  -- JSON blob
            debate_id TEXT,  -- Reference to Arena debate if used
            created_by TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_repo_pr
            ON pr_reviews(repository, pr_number);
        CREATE INDEX IF NOT EXISTS idx_reviews_status
            ON pr_reviews(status);

        -- Review comments
        CREATE TABLE IF NOT EXISTS review_comments (
            id TEXT PRIMARY KEY,
            review_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            body TEXT NOT NULL,
            suggestion TEXT,
            severity TEXT NOT NULL,  -- info, warning, error
            category TEXT,  -- quality, security, performance, etc.
            side TEXT DEFAULT 'RIGHT',
            submitted INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (review_id) REFERENCES pr_reviews(id)
        );

        CREATE INDEX IF NOT EXISTS idx_comments_review
            ON review_comments(review_id);
        CREATE INDEX IF NOT EXISTS idx_comments_file
            ON review_comments(file_path);
        """

    def _to_entity(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        result = dict(row)
        if "metrics" in result and result["metrics"]:
            try:
                result["metrics"] = json.loads(result["metrics"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def _from_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an entity to database row format."""
        result = dict(entity)
        if "metrics" in result and result["metrics"] and not isinstance(result["metrics"], str):
            result["metrics"] = json.dumps(result["metrics"])
        return result

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript(self._get_schema())

    def save_review(
        self,
        review_id: str,
        repository: str,
        pr_number: int,
        status: str,
        verdict: Optional[str] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        debate_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> str:
        """Save or update a PR review."""
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pr_reviews
                (id, repository, pr_number, status, verdict, summary,
                 started_at, completed_at, error, metrics, debate_id, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    repository,
                    pr_number,
                    status,
                    verdict,
                    summary,
                    now,
                    now if status in ("completed", "failed") else None,
                    error,
                    json.dumps(metrics) if metrics else None,
                    debate_id,
                    created_by,
                ),
            )
            return review_id

    def get_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific review by ID."""
        with self._connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM pr_reviews WHERE id = ?",
                (review_id,),
            ).fetchone()
            return self._to_entity(row) if row else None

    def get_pr_reviews(
        self,
        repository: str,
        pr_number: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get reviews for a specific PR."""
        with self._connection(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT * FROM pr_reviews
                WHERE repository = ? AND pr_number = ?
                ORDER BY started_at DESC LIMIT ?
                """,
                (repository, pr_number, limit),
            ).fetchall()
            return [self._to_entity(row) for row in rows]

    def save_comment(
        self,
        comment_id: str,
        review_id: str,
        file_path: str,
        body: str,
        severity: str,
        line_number: Optional[int] = None,
        suggestion: Optional[str] = None,
        category: Optional[str] = None,
        side: str = "RIGHT",
    ) -> str:
        """Save a review comment."""
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO review_comments
                (id, review_id, file_path, line_number, body, suggestion,
                 severity, category, side, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comment_id,
                    review_id,
                    file_path,
                    line_number,
                    body,
                    suggestion,
                    severity,
                    category,
                    side,
                    now,
                ),
            )
            return comment_id

    def get_comments(self, review_id: str) -> List[Dict[str, Any]]:
        """Get comments for a review."""
        with self._connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT * FROM review_comments WHERE review_id = ?",
                (review_id,),
            ).fetchall()
            return [self._to_entity(row) for row in rows]


# =============================================================================
# Repository Factory
# =============================================================================


def get_inbox_repository() -> InboxRepository:
    """Get the inbox repository singleton."""
    return InboxRepository()


def get_security_scan_repository() -> SecurityScanRepository:
    """Get the security scan repository singleton."""
    return SecurityScanRepository()


def get_pr_review_repository() -> PRReviewRepository:
    """Get the PR review repository singleton."""
    return PRReviewRepository()
