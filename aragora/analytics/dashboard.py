"""
Analytics Dashboard Backend.

Provides aggregated metrics and trend analysis for audit findings,
agent performance, and compliance tracking.

Features:
- Finding trends over time
- Severity and category distribution
- Mean time to remediation (MTTR)
- Agent accuracy and agreement rates
- Cost analysis per audit
- Compliance scorecards

Usage:
    from aragora.analytics.dashboard import (
        AnalyticsDashboard,
        get_analytics_dashboard,
        TimeRange,
    )

    dashboard = get_analytics_dashboard()

    # Get finding trends
    trends = await dashboard.get_finding_trends(
        workspace_id="ws-123",
        time_range=TimeRange.LAST_30_DAYS,
    )

    # Get compliance scorecard
    scorecard = await dashboard.get_compliance_scorecard(
        workspace_id="ws-123",
        frameworks=["SOC2", "GDPR"],
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
import threading

logger = logging.getLogger(__name__)


class TimeRange(str, Enum):
    """Predefined time ranges for analytics."""

    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    LAST_365_DAYS = "365d"
    ALL_TIME = "all"

    def to_timedelta(self) -> Optional[timedelta]:
        """Convert to timedelta."""
        mapping = {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "365d": timedelta(days=365),
            "all": None,
        }
        return mapping.get(self.value)


class Granularity(str, Enum):
    """Time granularity for trend data."""

    HOURLY = "hour"
    DAILY = "day"
    WEEKLY = "week"
    MONTHLY = "month"


@dataclass
class FindingTrend:
    """Finding count over a time period."""

    timestamp: datetime
    total: int
    by_severity: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total": self.total,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "by_status": self.by_status,
        }


@dataclass
class RemediationMetrics:
    """Metrics about finding remediation."""

    total_resolved: int
    total_open: int
    mttr_hours: float  # Mean Time To Remediation
    mttr_by_severity: dict[str, float] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    accepted_risk_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_resolved": self.total_resolved,
            "total_open": self.total_open,
            "mttr_hours": round(self.mttr_hours, 2),
            "mttr_by_severity": {
                k: round(v, 2) for k, v in self.mttr_by_severity.items()
            },
            "false_positive_rate": round(self.false_positive_rate, 4),
            "accepted_risk_rate": round(self.accepted_risk_rate, 4),
        }


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    agent_id: str
    agent_name: str
    total_findings: int
    agreement_rate: float  # Rate of agreement with consensus
    precision: float  # True positives / (True positives + False positives)
    finding_distribution: dict[str, int] = field(default_factory=dict)
    avg_response_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_findings": self.total_findings,
            "agreement_rate": round(self.agreement_rate, 4),
            "precision": round(self.precision, 4),
            "finding_distribution": self.finding_distribution,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
        }


@dataclass
class AuditCostMetrics:
    """Cost metrics for audits."""

    total_audits: int
    total_cost_usd: float
    avg_cost_per_audit: float
    cost_by_type: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_audits": self.total_audits,
            "total_cost_usd": round(self.total_cost_usd, 2),
            "avg_cost_per_audit": round(self.avg_cost_per_audit, 2),
            "cost_by_type": {k: round(v, 2) for k, v in self.cost_by_type.items()},
            "token_usage": self.token_usage,
        }


@dataclass
class ComplianceScore:
    """Compliance score for a framework."""

    framework: str
    score: float  # 0.0 to 1.0
    passing_controls: int
    failing_controls: int
    not_applicable: int
    critical_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "score": round(self.score, 4),
            "passing_controls": self.passing_controls,
            "failing_controls": self.failing_controls,
            "not_applicable": self.not_applicable,
            "critical_gaps": self.critical_gaps,
        }


@dataclass
class RiskHeatmapCell:
    """A cell in the risk heatmap."""

    category: str
    severity: str
    count: int
    trend: str  # "up", "down", "stable"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "severity": self.severity,
            "count": self.count,
            "trend": self.trend,
        }


@dataclass
class DashboardSummary:
    """Overall dashboard summary."""

    workspace_id: str
    time_range: TimeRange
    generated_at: datetime

    # Key metrics
    total_findings: int
    open_findings: int
    critical_findings: int
    resolved_last_period: int

    # Trends
    finding_trend: str  # "up", "down", "stable"
    trend_percentage: float

    # Top issues
    top_categories: list[tuple[str, int]] = field(default_factory=list)
    recent_critical: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "time_range": self.time_range.value,
            "generated_at": self.generated_at.isoformat(),
            "total_findings": self.total_findings,
            "open_findings": self.open_findings,
            "critical_findings": self.critical_findings,
            "resolved_last_period": self.resolved_last_period,
            "finding_trend": self.finding_trend,
            "trend_percentage": round(self.trend_percentage, 2),
            "top_categories": [{"category": c, "count": n} for c, n in self.top_categories],
            "recent_critical": self.recent_critical,
        }


class AnalyticsDashboard:
    """
    Analytics dashboard for audit findings and performance metrics.

    Provides aggregated views and trend analysis for enterprise
    compliance monitoring and reporting.
    """

    def __init__(self):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Cache a value."""
        self._cache[key] = (value, datetime.now(timezone.utc))

    async def get_summary(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
    ) -> DashboardSummary:
        """
        Get dashboard summary with key metrics.

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis

        Returns:
            Dashboard summary with key metrics
        """
        cache_key = f"summary:{workspace_id}:{time_range.value}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Get findings from audit system
        findings = await self._get_findings(workspace_id, time_range)

        # Calculate metrics
        total = len(findings)
        open_findings = sum(1 for f in findings if f.get("status") == "open")
        critical = sum(1 for f in findings if f.get("severity") == "critical")

        # Calculate trend
        delta = time_range.to_timedelta()
        if delta:
            prev_findings = await self._get_findings(
                workspace_id,
                time_range,
                offset=delta,
            )
            prev_total = len(prev_findings)
            if prev_total > 0:
                trend_pct = ((total - prev_total) / prev_total) * 100
            else:
                trend_pct = 100 if total > 0 else 0

            if trend_pct > 5:
                trend = "up"
            elif trend_pct < -5:
                trend = "down"
            else:
                trend = "stable"
        else:
            trend = "stable"
            trend_pct = 0

        # Resolved in period
        resolved = sum(
            1 for f in findings
            if f.get("status") == "resolved" and f.get("resolved_at")
        )

        # Top categories
        category_counts: dict[str, int] = {}
        for f in findings:
            cat = f.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        top_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Recent critical
        recent_critical = [
            {
                "id": f.get("id"),
                "title": f.get("title", f.get("message", ""))[:100],
                "created_at": f.get("created_at"),
            }
            for f in findings
            if f.get("severity") == "critical"
        ][:5]

        summary = DashboardSummary(
            workspace_id=workspace_id,
            time_range=time_range,
            generated_at=datetime.now(timezone.utc),
            total_findings=total,
            open_findings=open_findings,
            critical_findings=critical,
            resolved_last_period=resolved,
            finding_trend=trend,
            trend_percentage=trend_pct,
            top_categories=top_categories,
            recent_critical=recent_critical,
        )

        self._set_cached(cache_key, summary)
        return summary

    async def get_finding_trends(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
        granularity: Granularity = Granularity.DAILY,
    ) -> list[FindingTrend]:
        """
        Get finding counts over time.

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis
            granularity: Time bucket granularity

        Returns:
            List of finding trends by time bucket
        """
        findings = await self._get_findings(workspace_id, time_range)

        # Group by time bucket
        buckets: dict[str, list[dict]] = {}
        for f in findings:
            created = f.get("created_at")
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                bucket_key = self._get_bucket_key(created, granularity)
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(f)

        # Build trends
        trends = []
        for bucket_key in sorted(buckets.keys()):
            bucket_findings = buckets[bucket_key]

            by_severity: dict[str, int] = {}
            by_category: dict[str, int] = {}
            by_status: dict[str, int] = {}

            for f in bucket_findings:
                sev = f.get("severity", "unknown")
                cat = f.get("category", "unknown")
                status = f.get("status", "unknown")

                by_severity[sev] = by_severity.get(sev, 0) + 1
                by_category[cat] = by_category.get(cat, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1

            trends.append(FindingTrend(
                timestamp=self._parse_bucket_key(bucket_key, granularity),
                total=len(bucket_findings),
                by_severity=by_severity,
                by_category=by_category,
                by_status=by_status,
            ))

        return trends

    async def get_remediation_metrics(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
    ) -> RemediationMetrics:
        """
        Get remediation performance metrics.

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis

        Returns:
            Remediation metrics including MTTR
        """
        findings = await self._get_findings(workspace_id, time_range)

        resolved = [f for f in findings if f.get("status") == "resolved"]
        open_findings = [f for f in findings if f.get("status") == "open"]
        false_positives = [f for f in findings if f.get("status") == "false_positive"]
        accepted_risk = [f for f in findings if f.get("status") == "accepted_risk"]

        # Calculate MTTR
        resolution_times = []
        resolution_by_severity: dict[str, list[float]] = {}

        for f in resolved:
            created = f.get("created_at")
            resolved_at = f.get("resolved_at")
            if created and resolved_at:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if isinstance(resolved_at, str):
                    resolved_at = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))

                hours = (resolved_at - created).total_seconds() / 3600
                resolution_times.append(hours)

                sev = f.get("severity", "unknown")
                if sev not in resolution_by_severity:
                    resolution_by_severity[sev] = []
                resolution_by_severity[sev].append(hours)

        mttr = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        mttr_by_sev = {
            sev: sum(times) / len(times)
            for sev, times in resolution_by_severity.items()
        }

        total = len(findings)
        fp_rate = len(false_positives) / total if total > 0 else 0
        ar_rate = len(accepted_risk) / total if total > 0 else 0

        return RemediationMetrics(
            total_resolved=len(resolved),
            total_open=len(open_findings),
            mttr_hours=mttr,
            mttr_by_severity=mttr_by_sev,
            false_positive_rate=fp_rate,
            accepted_risk_rate=ar_rate,
        )

    async def get_agent_metrics(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
    ) -> list[AgentMetrics]:
        """
        Get performance metrics for each agent.

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis

        Returns:
            List of agent performance metrics
        """
        # Get audit sessions with agent data
        sessions = await self._get_sessions(workspace_id, time_range)

        agent_data: dict[str, dict] = {}

        for session in sessions:
            for agent_result in session.get("agent_results", []):
                agent_id = agent_result.get("agent_id")
                agent_name = agent_result.get("agent_name", agent_id)

                if agent_id not in agent_data:
                    agent_data[agent_id] = {
                        "name": agent_name,
                        "findings": [],
                        "agreements": 0,
                        "disagreements": 0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "response_times": [],
                    }

                # Track findings
                findings = agent_result.get("findings", [])
                agent_data[agent_id]["findings"].extend(findings)

                # Track agreement with consensus
                if agent_result.get("agreed_with_consensus"):
                    agent_data[agent_id]["agreements"] += 1
                else:
                    agent_data[agent_id]["disagreements"] += 1

                # Track response time
                if "response_time_ms" in agent_result:
                    agent_data[agent_id]["response_times"].append(
                        agent_result["response_time_ms"]
                    )

        # Build metrics
        metrics = []
        for agent_id, data in agent_data.items():
            total_decisions = data["agreements"] + data["disagreements"]
            agreement_rate = (
                data["agreements"] / total_decisions if total_decisions > 0 else 0
            )

            # Calculate precision (using false positive status from findings)
            total_findings = len(data["findings"])
            fp_count = sum(
                1 for f in data["findings"]
                if f.get("status") == "false_positive"
            )
            precision = (
                (total_findings - fp_count) / total_findings
                if total_findings > 0 else 0
            )

            # Finding distribution by severity
            dist: dict[str, int] = {}
            for f in data["findings"]:
                sev = f.get("severity", "unknown")
                dist[sev] = dist.get(sev, 0) + 1

            avg_response = (
                sum(data["response_times"]) / len(data["response_times"])
                if data["response_times"] else 0
            )

            metrics.append(AgentMetrics(
                agent_id=agent_id,
                agent_name=data["name"],
                total_findings=total_findings,
                agreement_rate=agreement_rate,
                precision=precision,
                finding_distribution=dist,
                avg_response_time_ms=avg_response,
            ))

        return sorted(metrics, key=lambda m: m.total_findings, reverse=True)

    async def get_cost_metrics(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
    ) -> AuditCostMetrics:
        """
        Get cost metrics for audits.

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis

        Returns:
            Cost metrics including per-audit costs
        """
        sessions = await self._get_sessions(workspace_id, time_range)

        total_cost = 0.0
        cost_by_type: dict[str, float] = {}
        token_usage: dict[str, int] = {"input": 0, "output": 0}

        for session in sessions:
            cost = session.get("cost_usd", 0)
            total_cost += cost

            audit_type = session.get("audit_type", "unknown")
            cost_by_type[audit_type] = cost_by_type.get(audit_type, 0) + cost

            usage = session.get("token_usage", {})
            token_usage["input"] += usage.get("input_tokens", 0)
            token_usage["output"] += usage.get("output_tokens", 0)

        total_audits = len(sessions)
        avg_cost = total_cost / total_audits if total_audits > 0 else 0

        return AuditCostMetrics(
            total_audits=total_audits,
            total_cost_usd=total_cost,
            avg_cost_per_audit=avg_cost,
            cost_by_type=cost_by_type,
            token_usage=token_usage,
        )

    async def get_compliance_scorecard(
        self,
        workspace_id: str,
        frameworks: Optional[list[str]] = None,
    ) -> list[ComplianceScore]:
        """
        Get compliance scores for specified frameworks.

        Args:
            workspace_id: Workspace to analyze
            frameworks: List of frameworks to score (e.g., ["SOC2", "GDPR"])

        Returns:
            Compliance scores for each framework
        """
        if frameworks is None:
            frameworks = ["SOC2", "GDPR", "HIPAA", "PCI-DSS"]

        scores = []
        for framework in frameworks:
            score = await self._calculate_compliance_score(workspace_id, framework)
            scores.append(score)

        return scores

    async def get_risk_heatmap(
        self,
        workspace_id: str,
        time_range: TimeRange = TimeRange.LAST_30_DAYS,
    ) -> list[RiskHeatmapCell]:
        """
        Get risk heatmap data (category x severity).

        Args:
            workspace_id: Workspace to analyze
            time_range: Time range for analysis

        Returns:
            List of heatmap cells
        """
        findings = await self._get_findings(workspace_id, time_range)

        # Previous period for trend
        delta = time_range.to_timedelta()
        prev_findings = (
            await self._get_findings(workspace_id, time_range, offset=delta)
            if delta else []
        )

        # Count by category/severity
        current: dict[tuple[str, str], int] = {}
        previous: dict[tuple[str, str], int] = {}

        for f in findings:
            key = (f.get("category", "unknown"), f.get("severity", "unknown"))
            current[key] = current.get(key, 0) + 1

        for f in prev_findings:
            key = (f.get("category", "unknown"), f.get("severity", "unknown"))
            previous[key] = previous.get(key, 0) + 1

        # Build cells with trends
        cells = []
        all_keys = set(current.keys()) | set(previous.keys())

        for cat, sev in all_keys:
            curr_count = current.get((cat, sev), 0)
            prev_count = previous.get((cat, sev), 0)

            if curr_count > prev_count:
                trend = "up"
            elif curr_count < prev_count:
                trend = "down"
            else:
                trend = "stable"

            cells.append(RiskHeatmapCell(
                category=cat,
                severity=sev,
                count=curr_count,
                trend=trend,
            ))

        return cells

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_findings(
        self,
        workspace_id: str,
        time_range: TimeRange,
        offset: Optional[timedelta] = None,
    ) -> list[dict]:
        """Get findings for workspace in time range."""
        try:
            from aragora.audit import get_document_auditor

            auditor = get_document_auditor()

            delta = time_range.to_timedelta()
            now = datetime.now(timezone.utc)

            if offset:
                end_time = now - offset
                start_time = end_time - delta if delta else None
            else:
                end_time = now
                start_time = now - delta if delta else None

            findings = []
            for session in auditor._sessions.values():
                if session.workspace_id != workspace_id:
                    continue

                for finding in session.findings:
                    created = finding.created_at
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))

                    if start_time and created < start_time:
                        continue
                    if created > end_time:
                        continue

                    findings.append({
                        "id": finding.id,
                        "title": getattr(finding, "title", None),
                        "message": finding.message,
                        "severity": finding.severity,
                        "category": finding.category,
                        "status": getattr(finding, "status", "open"),
                        "created_at": finding.created_at,
                        "resolved_at": getattr(finding, "resolved_at", None),
                    })

            return findings

        except Exception as e:
            logger.warning(f"Failed to get findings: {e}")
            return []

    async def _get_sessions(
        self,
        workspace_id: str,
        time_range: TimeRange,
    ) -> list[dict]:
        """Get audit sessions for workspace in time range."""
        try:
            from aragora.audit import get_document_auditor

            auditor = get_document_auditor()

            delta = time_range.to_timedelta()
            now = datetime.now(timezone.utc)
            start_time = now - delta if delta else None

            sessions = []
            for session in auditor._sessions.values():
                if session.workspace_id != workspace_id:
                    continue

                created = session.created_at
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))

                if start_time and created < start_time:
                    continue

                sessions.append({
                    "id": session.id,
                    "audit_type": getattr(session, "audit_type", "unknown"),
                    "created_at": session.created_at,
                    "cost_usd": getattr(session, "cost_usd", 0),
                    "token_usage": getattr(session, "token_usage", {}),
                    "agent_results": getattr(session, "agent_results", []),
                })

            return sessions

        except Exception as e:
            logger.warning(f"Failed to get sessions: {e}")
            return []

    async def _calculate_compliance_score(
        self,
        workspace_id: str,
        framework: str,
    ) -> ComplianceScore:
        """Calculate compliance score for a framework."""
        # Get findings mapped to framework controls
        findings = await self._get_findings(workspace_id, TimeRange.ALL_TIME)

        # Framework control mappings (simplified)
        control_mappings = {
            "SOC2": {
                "access_control": ["authentication", "authorization"],
                "data_protection": ["encryption", "data_handling"],
                "monitoring": ["logging", "alerting"],
                "incident_response": ["security", "incident"],
            },
            "GDPR": {
                "data_processing": ["pii", "personal_data"],
                "consent": ["consent", "privacy"],
                "data_rights": ["deletion", "export"],
                "breach_notification": ["incident", "breach"],
            },
            "HIPAA": {
                "access_controls": ["authentication", "authorization"],
                "audit_controls": ["logging", "audit"],
                "integrity_controls": ["integrity", "validation"],
                "transmission_security": ["encryption", "transmission"],
            },
            "PCI-DSS": {
                "network_security": ["firewall", "network"],
                "data_protection": ["encryption", "cardholder"],
                "access_control": ["authentication", "authorization"],
                "monitoring": ["logging", "monitoring"],
            },
        }

        controls = control_mappings.get(framework, {})
        if not controls:
            return ComplianceScore(
                framework=framework,
                score=0,
                passing_controls=0,
                failing_controls=0,
                not_applicable=0,
            )

        passing = 0
        failing = 0
        critical_gaps = []

        for control_name, keywords in controls.items():
            # Check if any findings match this control
            control_findings = [
                f for f in findings
                if any(kw in f.get("category", "").lower() for kw in keywords)
            ]

            open_critical = [
                f for f in control_findings
                if f.get("status") == "open" and f.get("severity") == "critical"
            ]

            if open_critical:
                failing += 1
                critical_gaps.append(control_name)
            else:
                passing += 1

        total = passing + failing
        score = passing / total if total > 0 else 0

        return ComplianceScore(
            framework=framework,
            score=score,
            passing_controls=passing,
            failing_controls=failing,
            not_applicable=0,
            critical_gaps=critical_gaps,
        )

    def _get_bucket_key(self, dt: datetime, granularity: Granularity) -> str:
        """Get bucket key for a datetime."""
        if granularity == Granularity.HOURLY:
            return dt.strftime("%Y-%m-%d-%H")
        elif granularity == Granularity.DAILY:
            return dt.strftime("%Y-%m-%d")
        elif granularity == Granularity.WEEKLY:
            # Start of week (Monday)
            start = dt - timedelta(days=dt.weekday())
            return start.strftime("%Y-%m-%d")
        else:  # MONTHLY
            return dt.strftime("%Y-%m")

    def _parse_bucket_key(self, key: str, granularity: Granularity) -> datetime:
        """Parse bucket key back to datetime."""
        if granularity == Granularity.HOURLY:
            return datetime.strptime(key, "%Y-%m-%d-%H").replace(tzinfo=timezone.utc)
        elif granularity == Granularity.DAILY:
            return datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        elif granularity == Granularity.WEEKLY:
            return datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:  # MONTHLY
            return datetime.strptime(key + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)


# Global instance
_dashboard: Optional[AnalyticsDashboard] = None
_lock = threading.Lock()


def get_analytics_dashboard() -> AnalyticsDashboard:
    """Get the global analytics dashboard instance."""
    global _dashboard

    if _dashboard is None:
        with _lock:
            if _dashboard is None:
                _dashboard = AnalyticsDashboard()

    return _dashboard
