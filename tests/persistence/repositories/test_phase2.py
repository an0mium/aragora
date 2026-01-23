"""
Tests for Phase 2 repositories.

Tests:
- InboxRepository
- SecurityScanRepository
- PRReviewRepository
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from aragora.persistence.repositories.phase2 import (
    InboxRepository,
    SecurityScanRepository,
    PRReviewRepository,
)


class TestInboxRepository:
    """Tests for InboxRepository."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test_inbox.db"
        return InboxRepository(db_path=db_path)

    def test_save_and_get_prioritization(self, repo):
        """Test saving and retrieving email prioritization."""
        record_id = repo.save_prioritization(
            user_id="user1",
            email_id="email_123",
            priority="high",
            confidence=0.85,
            reasoning="Important sender",
            tier_used="tier_1_rules",
            scores={"sender": 0.9, "urgency": 0.8},
            suggested_labels=["important", "reply"],
            auto_archive=False,
        )

        assert record_id.startswith("pri_")

        # Retrieve it
        result = repo.get_prioritization("user1", "email_123")

        assert result is not None
        assert result["priority"] == "high"
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Important sender"
        assert result["scores"]["sender"] == 0.9
        assert "important" in result["suggested_labels"]

    def test_get_prioritization_not_found(self, repo):
        """Test getting non-existent prioritization."""
        result = repo.get_prioritization("user1", "nonexistent")
        assert result is None

    def test_get_recent_priorities(self, repo):
        """Test getting recent priorities."""
        # Save multiple
        for i in range(5):
            repo.save_prioritization(
                user_id="user1",
                email_id=f"email_{i}",
                priority="high" if i % 2 == 0 else "low",
                confidence=0.8,
            )

        # Get all
        results = repo.get_recent_priorities("user1", limit=10)
        assert len(results) == 5

        # Get with filter
        high_results = repo.get_recent_priorities("user1", limit=10, priority_filter="high")
        assert all(r["priority"] == "high" for r in high_results)

    def test_record_action(self, repo):
        """Test recording user actions."""
        repo.record_action(
            user_id="user1",
            email_id="email_123",
            action="archive",
            params={"reason": "cleanup"},
            predicted_priority="low",
            actual_feedback="correct",
        )

        # Verify by checking the database directly
        with repo._connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM email_actions WHERE email_id = ?", ("email_123",)
            ).fetchone()

        assert row is not None
        assert row["action"] == "archive"
        assert row["predicted_priority"] == "low"

    def test_upsert_prioritization(self, repo):
        """Test that prioritization is updated on conflict."""
        repo.save_prioritization(
            user_id="user1",
            email_id="email_123",
            priority="low",
            confidence=0.5,
        )

        # Update
        repo.save_prioritization(
            user_id="user1",
            email_id="email_123",
            priority="high",
            confidence=0.9,
        )

        result = repo.get_prioritization("user1", "email_123")
        assert result["priority"] == "high"
        assert result["confidence"] == 0.9


class TestSecurityScanRepository:
    """Tests for SecurityScanRepository."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test_security.db"
        return SecurityScanRepository(db_path=db_path)

    def test_save_and_get_scan(self, repo):
        """Test saving and retrieving a scan."""
        scan_id = repo.save_scan(
            scan_id="scan_123",
            repository="owner/repo",
            scan_type="dependency",
            status="completed",
            branch="main",
            commit_sha="abc123",
            summary={"vulnerabilities": 5, "critical": 1},
        )

        assert scan_id == "scan_123"

        result = repo.get_scan("scan_123")

        assert result is not None
        assert result["repository"] == "owner/repo"
        assert result["scan_type"] == "dependency"
        assert result["status"] == "completed"
        assert result["summary"]["vulnerabilities"] == 5

    def test_get_scan_not_found(self, repo):
        """Test getting non-existent scan."""
        result = repo.get_scan("nonexistent")
        assert result is None

    def test_get_recent_scans(self, repo):
        """Test getting recent scans for a repository."""
        for i in range(5):
            repo.save_scan(
                scan_id=f"scan_{i}",
                repository="owner/repo",
                scan_type="dependency" if i % 2 == 0 else "secrets",
                status="completed",
            )

        # Get all
        results = repo.get_recent_scans("owner/repo", limit=10)
        assert len(results) == 5

        # Get by type
        dep_scans = repo.get_recent_scans("owner/repo", limit=10, scan_type="dependency")
        assert all(r["scan_type"] == "dependency" for r in dep_scans)

    def test_save_and_get_vulnerability(self, repo):
        """Test saving and retrieving vulnerabilities."""
        # Create scan first
        repo.save_scan(
            scan_id="scan_1",
            repository="owner/repo",
            scan_type="dependency",
            status="completed",
        )

        # Add vulnerabilities
        repo.save_vulnerability(
            vuln_id="vuln_1",
            scan_id="scan_1",
            title="SQL Injection",
            severity="critical",
            cve_id="CVE-2024-1234",
            cvss_score=9.8,
            package_name="vulnerable-lib",
            vulnerable_versions=["<1.0.0"],
            patched_versions=["1.0.1"],
            fix_available=True,
            recommended_version="1.0.1",
        )

        # Retrieve
        vulns = repo.get_vulnerabilities("scan_1")

        assert len(vulns) == 1
        vuln = vulns[0]
        assert vuln["cve_id"] == "CVE-2024-1234"
        assert vuln["severity"] == "critical"
        assert vuln["cvss_score"] == 9.8
        assert "<1.0.0" in vuln["vulnerable_versions"]

    def test_get_vulnerabilities_by_severity(self, repo):
        """Test filtering vulnerabilities by severity."""
        repo.save_scan(
            scan_id="scan_1",
            repository="owner/repo",
            scan_type="dependency",
            status="completed",
        )

        severities = ["critical", "high", "medium", "low"]
        for i, sev in enumerate(severities):
            repo.save_vulnerability(
                vuln_id=f"vuln_{i}",
                scan_id="scan_1",
                title=f"Vuln {i}",
                severity=sev,
            )

        critical = repo.get_vulnerabilities("scan_1", severity_filter="critical")
        assert len(critical) == 1
        assert critical[0]["severity"] == "critical"


class TestPRReviewRepository:
    """Tests for PRReviewRepository."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test_pr_reviews.db"
        return PRReviewRepository(db_path=db_path)

    def test_save_and_get_review(self, repo):
        """Test saving and retrieving a review."""
        review_id = repo.save_review(
            review_id="review_123",
            repository="owner/repo",
            pr_number=42,
            status="completed",
            verdict="APPROVE",
            summary="LGTM!",
            metrics={"files_reviewed": 5, "comments": 3},
        )

        assert review_id == "review_123"

        result = repo.get_review("review_123")

        assert result is not None
        assert result["repository"] == "owner/repo"
        assert result["pr_number"] == 42
        assert result["status"] == "completed"
        assert result["verdict"] == "APPROVE"
        assert result["metrics"]["files_reviewed"] == 5

    def test_get_review_not_found(self, repo):
        """Test getting non-existent review."""
        result = repo.get_review("nonexistent")
        assert result is None

    def test_get_pr_reviews(self, repo):
        """Test getting reviews for a specific PR."""
        for i in range(3):
            repo.save_review(
                review_id=f"review_{i}",
                repository="owner/repo",
                pr_number=42,
                status="completed",
                verdict="APPROVE",
            )

        results = repo.get_pr_reviews("owner/repo", 42)

        assert len(results) == 3
        assert all(r["pr_number"] == 42 for r in results)

    def test_save_and_get_comments(self, repo):
        """Test saving and retrieving review comments."""
        # Create review first
        repo.save_review(
            review_id="review_1",
            repository="owner/repo",
            pr_number=42,
            status="completed",
        )

        # Add comments
        repo.save_comment(
            comment_id="comment_1",
            review_id="review_1",
            file_path="src/main.py",
            body="Consider refactoring this function.",
            severity="warning",
            line_number=10,
            category="quality",
        )

        repo.save_comment(
            comment_id="comment_2",
            review_id="review_1",
            file_path="src/auth.py",
            body="Security issue detected.",
            severity="error",
            line_number=25,
            category="security",
            suggestion="Use parameterized queries.",
        )

        # Retrieve
        comments = repo.get_comments("review_1")

        assert len(comments) == 2

        quality_comment = next(c for c in comments if c["category"] == "quality")
        assert quality_comment["file_path"] == "src/main.py"
        assert quality_comment["severity"] == "warning"

        security_comment = next(c for c in comments if c["category"] == "security")
        assert security_comment["suggestion"] == "Use parameterized queries."

    def test_review_with_debate_id(self, repo):
        """Test storing review with debate reference."""
        repo.save_review(
            review_id="review_1",
            repository="owner/repo",
            pr_number=42,
            status="completed",
            debate_id="debate_xyz",
        )

        result = repo.get_review("review_1")
        assert result["debate_id"] == "debate_xyz"


class TestRepositoryEdgeCases:
    """Test edge cases and error handling."""

    def test_inbox_json_fields_empty(self, tmp_path):
        """Test handling of empty JSON fields."""
        repo = InboxRepository(db_path=tmp_path / "test.db")

        repo.save_prioritization(
            user_id="user1",
            email_id="email_1",
            priority="medium",
            confidence=0.5,
            scores=None,
            suggested_labels=None,
        )

        result = repo.get_prioritization("user1", "email_1")
        assert result["scores"] is None
        assert result["suggested_labels"] is None

    def test_security_scan_update(self, tmp_path):
        """Test updating a scan record."""
        repo = SecurityScanRepository(db_path=tmp_path / "test.db")

        repo.save_scan(
            scan_id="scan_1",
            repository="owner/repo",
            scan_type="dependency",
            status="running",
        )

        # Update status
        repo.save_scan(
            scan_id="scan_1",
            repository="owner/repo",
            scan_type="dependency",
            status="completed",
            summary={"total": 100, "vulnerable": 5},
        )

        result = repo.get_scan("scan_1")
        assert result["status"] == "completed"
        assert result["summary"]["total"] == 100

    def test_pr_review_created_by(self, tmp_path):
        """Test tracking who created a review."""
        repo = PRReviewRepository(db_path=tmp_path / "test.db")

        repo.save_review(
            review_id="review_1",
            repository="owner/repo",
            pr_number=42,
            status="completed",
            created_by="bot_user",
        )

        result = repo.get_review("review_1")
        assert result["created_by"] == "bot_user"
