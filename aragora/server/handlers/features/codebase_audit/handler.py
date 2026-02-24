"""
Codebase Audit API Handler.

Stability: STABLE
Graduated from EXPERIMENTAL on 2026-02-02.

Provides unified REST APIs for codebase security and quality analysis:
- SAST vulnerability scanning
- Bug detection
- Secrets scanning
- Dependency vulnerability analysis
- Code metrics and quality analysis
- Scan orchestration and history

Endpoints:
- POST /api/v1/codebase/scan              - Start a comprehensive scan
- GET  /api/v1/codebase/scan/{id}         - Get scan status/results
- GET  /api/v1/codebase/scans             - List past scans
- POST /api/v1/codebase/sast              - Run SAST scan only
- POST /api/v1/codebase/bugs              - Run bug detection only
- POST /api/v1/codebase/secrets           - Run secrets scan only
- POST /api/v1/codebase/dependencies      - Run dependency scan only
- POST /api/v1/codebase/metrics           - Run metrics analysis only
- GET  /api/v1/codebase/findings          - Get aggregated findings
- GET  /api/v1/codebase/dashboard         - Get dashboard data
- POST /api/v1/codebase/findings/{id}/dismiss - Dismiss a finding
- POST /api/v1/codebase/findings/{id}/create-issue - Create GitHub issue
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ...base import (
    HandlerResult,
    error_response,
    success_response,
)
from ...secure import SecureHandler, ForbiddenError, UnauthorizedError
from ...utils.rate_limit import rate_limit
from aragora.resilience import CircuitBreaker, CircuitBreakerConfig

from .rules import (
    CODEBASE_AUDIT_READ_PERMISSION,
    CODEBASE_AUDIT_WRITE_PERMISSION,
    Finding,
    FindingStatus,
    ScanResult,
    ScanStatus,
    ScanType,
    sanitize_query_params,
    validate_dismiss_request,
    validate_finding_id,
    validate_github_repo,
    validate_limit,
    validate_repository_path,
    validate_scan_id,
    validate_scan_status_filter,
    validate_scan_types,
    validate_severity_filter,
    validate_status_filter,
)
from .scanning import _get_tenant_findings, _get_tenant_scans
from .reporting import (
    build_dashboard_data,
    build_demo_data,
    calculate_risk_score,
)

logger = logging.getLogger(__name__)


def _get_scanner_module():
    """Resolve scan functions via the package namespace so tests can patch them."""
    from aragora.server.handlers.features import codebase_audit as codebase_module

    return codebase_module


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

_codebase_audit_circuit_breaker = CircuitBreaker.from_config(
    CircuitBreakerConfig(  # type: ignore[arg-type]  # v2 config compat
        failure_threshold=5,
        cooldown_seconds=60.0,
        success_threshold=2,
    ),
    name="codebase_audit",
)


# =============================================================================
# Handler Class
# =============================================================================


class CodebaseAuditHandler(SecureHandler):
    """Handler for codebase audit API endpoints.

    RBAC Protected:
    - codebase_audit:read - required for GET endpoints (list scans, findings, dashboard)
    - codebase_audit:write - required for POST endpoints (start scans, dismiss, create issues)
    """

    ROUTES = [
        "/api/v1/codebase/scan",
        "/api/v1/codebase/scan/{scan_id}",
        "/api/v1/codebase/scans",
        "/api/v1/codebase/sast",
        "/api/v1/codebase/bugs",
        "/api/v1/codebase/secrets",
        "/api/v1/codebase/dependencies",
        "/api/v1/codebase/metrics",
        "/api/v1/codebase/findings",
        "/api/v1/codebase/findings/{finding_id}/dismiss",
        "/api/v1/codebase/findings/{finding_id}/create-issue",
        "/api/v1/codebase/dashboard",
        "/api/v1/codebase/demo",
    ]

    def __init__(self, server_context: dict[str, Any] | None = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context if server_context is not None else dict())

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/codebase")

    async def handle(  # type: ignore[override]
        self, request: Any, path: str, method: str = "GET", **kwargs: Any
    ) -> HandlerResult:
        """Route requests (delegates to handle_request)."""
        return await self.handle_request(request, path, method)

    async def handle_request(self, request: Any, path: str, method: str) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            # RBAC: Require authentication and appropriate permission
            required_permission = (
                CODEBASE_AUDIT_WRITE_PERMISSION
                if method == "POST"
                else CODEBASE_AUDIT_READ_PERMISSION
            )
            try:
                auth_context = await self.get_auth_context(request, require_auth=True)
                self.check_permission(auth_context, required_permission)
            except UnauthorizedError:
                return error_response("Authentication required for codebase audit", 401)
            except ForbiddenError as e:
                logger.warning("Codebase audit access denied: %s", e)
                return error_response("Permission denied", 403)

            tenant_id = self._get_tenant_id(request)

            # Comprehensive scan
            if path == "/api/v1/codebase/scan" and method == "POST":
                return await self._handle_comprehensive_scan(request, tenant_id)

            # List scans
            elif path == "/api/v1/codebase/scans" and method == "GET":
                return await self._handle_list_scans(request, tenant_id)

            # Individual scan types
            elif path == "/api/v1/codebase/sast" and method == "POST":
                return await self._handle_sast_scan(request, tenant_id)

            elif path == "/api/v1/codebase/bugs" and method == "POST":
                return await self._handle_bug_scan(request, tenant_id)

            elif path == "/api/v1/codebase/secrets" and method == "POST":
                return await self._handle_secrets_scan(request, tenant_id)

            elif path == "/api/v1/codebase/dependencies" and method == "POST":
                return await self._handle_dependency_scan(request, tenant_id)

            elif path == "/api/v1/codebase/metrics" and method == "POST":
                return await self._handle_metrics_analysis(request, tenant_id)

            # Findings
            elif path == "/api/v1/codebase/findings" and method == "GET":
                return await self._handle_list_findings(request, tenant_id)

            # Dashboard
            elif path == "/api/v1/codebase/dashboard" and method == "GET":
                return await self._handle_dashboard(request, tenant_id)

            # Demo data
            elif path == "/api/v1/codebase/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Scan-specific paths
            # Path format: /api/v1/codebase/scan/{scan_id}
            # parts[0]='' [1]='api' [2]='v1' [3]='codebase' [4]='scan' [5]=scan_id
            elif path.startswith("/api/v1/codebase/scan/"):
                parts = path.split("/")
                if len(parts) == 6:
                    scan_id = parts[5]
                    if method == "GET":
                        return await self._handle_get_scan(request, tenant_id, scan_id)

            # Finding-specific paths
            # Path format: /api/v1/codebase/findings/{finding_id}/{action}
            # parts[0]='' [1]='api' [2]='v1' [3]='codebase' [4]='findings' [5]=finding_id [6]=action
            elif path.startswith("/api/v1/codebase/findings/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    finding_id = parts[5]
                    if len(parts) == 7:
                        action = parts[6]
                        if action == "dismiss" and method == "POST":
                            return await self._handle_dismiss_finding(
                                request, tenant_id, finding_id
                            )
                        elif action == "create-issue" and method == "POST":
                            return await self._handle_create_issue(request, tenant_id, finding_id)

            return error_response("Not found", 404)

        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Error in codebase audit handler: %s", e)
            return error_response("Internal server error", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Comprehensive Scan
    # =========================================================================

    @rate_limit(requests_per_minute=10, limiter_name="codebase_audit_scan")
    async def _handle_comprehensive_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Start a comprehensive codebase scan.

        Request body:
        {
            "target_path": "/path/to/code",
            "scan_types": ["sast", "bugs", "secrets", "dependencies", "metrics"],
            "languages": ["python", "javascript"]  // Optional
        }

        Protected by:
        - Rate limit: 10 requests per minute
        - Circuit breaker: Opens after 5 consecutive failures
        """
        # Check circuit breaker
        if not _codebase_audit_circuit_breaker.can_proceed():
            logger.warning("Circuit breaker is open for codebase audit scans")
            return error_response(
                "Service temporarily unavailable. Please try again later.",
                503,
            )

        try:
            body = await self._get_json_body(request)

            target_path = body.get("target_path", ".")
            scan_types = body.get("scan_types", ["sast", "bugs", "secrets", "dependencies"])
            languages = body.get("languages")

            # Validate target_path to prevent directory traversal
            is_valid, error_msg = validate_repository_path(target_path)
            if not is_valid:
                logger.warning("Invalid target_path rejected: %s", error_msg)
                return error_response(f"Invalid target_path: {error_msg}", 400)

            # Validate scan_types
            is_valid, error_msg = validate_scan_types(scan_types)
            if not is_valid:
                logger.warning("Invalid scan_types rejected: %s", error_msg)
                return error_response(f"Invalid scan_types: {error_msg}", 400)

            # Create scan result
            scan_id = f"scan_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=ScanType.COMPREHENSIVE,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            # Store scan
            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            # Run scans in parallel
            all_findings: list[Finding] = []
            metrics: dict[str, Any] = {}

            scanners = _get_scanner_module()
            tasks: list[tuple[str, Any]] = []
            if "sast" in scan_types:
                tasks.append(
                    ("sast", scanners.run_sast_scan(target_path, scan_id, tenant_id, languages))
                )
            if "bugs" in scan_types:
                tasks.append(("bugs", scanners.run_bug_scan(target_path, scan_id, tenant_id)))
            if "secrets" in scan_types:
                tasks.append(
                    ("secrets", scanners.run_secrets_scan(target_path, scan_id, tenant_id))
                )
            if "dependencies" in scan_types:
                tasks.append(
                    ("dependencies", scanners.run_dependency_scan(target_path, scan_id, tenant_id))
                )
            if "metrics" in scan_types:
                tasks.append(
                    ("metrics", scanners.run_metrics_analysis(target_path, scan_id, tenant_id))
                )

            # Execute in parallel
            results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )

            for i, (scan_type_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, BaseException):
                    logger.error("%s scan failed: %s", scan_type_name, result)
                elif scan_type_name == "metrics":
                    if isinstance(result, dict):
                        metrics = result
                else:
                    if isinstance(result, list):
                        all_findings.extend(result)

            # Update scan result
            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.findings = all_findings
            scan_result.findings_count = len(all_findings)
            scan_result.metrics = metrics
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            # Count severities
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            for finding in all_findings:
                severity_counts[finding.severity.value] += 1
            scan_result.severity_counts = severity_counts

            # Store findings
            findings_store = _get_tenant_findings(tenant_id)
            for finding in all_findings:
                findings_store[finding.id] = finding

            # Record success in circuit breaker
            _codebase_audit_circuit_breaker.record_success()

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "findings": [f.to_dict() for f in all_findings],
                    "metrics": metrics,
                    "summary": {
                        "total_findings": len(all_findings),
                        "severity_counts": severity_counts,
                        "scan_types_run": [t[0] for t in tasks],
                    },
                }
            )

        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            # Record failure in circuit breaker
            _codebase_audit_circuit_breaker.record_failure()
            logger.exception("Comprehensive scan error: %s", e)
            return error_response("Scan operation failed", 500)

    # =========================================================================
    # Individual Scan Types
    # =========================================================================

    async def _handle_sast_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run SAST-only scan."""
        return await self._run_single_scan(
            request, tenant_id, ScanType.SAST, _get_scanner_module().run_sast_scan
        )

    async def _handle_bug_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run bug detection scan."""
        return await self._run_single_scan(
            request, tenant_id, ScanType.BUGS, _get_scanner_module().run_bug_scan
        )

    async def _handle_secrets_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run secrets scan."""
        return await self._run_single_scan(
            request, tenant_id, ScanType.SECRETS, _get_scanner_module().run_secrets_scan
        )

    async def _handle_dependency_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run dependency vulnerability scan."""
        return await self._run_single_scan(
            request,
            tenant_id,
            ScanType.DEPENDENCIES,
            _get_scanner_module().run_dependency_scan,
        )

    @rate_limit(requests_per_minute=20, limiter_name="codebase_audit_single_scan")
    async def _run_single_scan(
        self,
        request: Any,
        tenant_id: str,
        scan_type: ScanType,
        scan_func: Any,
    ) -> HandlerResult:
        """Run a single type of scan.

        Protected by:
        - Rate limit: 20 requests per minute
        - Circuit breaker: Opens after 5 consecutive failures
        """
        # Check circuit breaker
        if not _codebase_audit_circuit_breaker.can_proceed():
            logger.warning("Circuit breaker is open for %s scan", scan_type.value)
            return error_response(
                "Service temporarily unavailable. Please try again later.",
                503,
            )

        try:
            body = await self._get_json_body(request)
            target_path = body.get("target_path", ".")

            # Validate target_path to prevent directory traversal
            is_valid, error_msg = validate_repository_path(target_path)
            if not is_valid:
                logger.warning(
                    "Invalid target_path rejected in %s scan: %s", scan_type.value, error_msg
                )
                return error_response(f"Invalid target_path: {error_msg}", 400)

            scan_id = f"scan_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=scan_type,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            # Run scan
            if scan_type == ScanType.SAST:
                languages = body.get("languages")
                findings = await scan_func(target_path, scan_id, tenant_id, languages)
            else:
                findings = await scan_func(target_path, scan_id, tenant_id)

            # Update result
            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.findings = findings
            scan_result.findings_count = len(findings)
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            # Store findings
            findings_store = _get_tenant_findings(tenant_id)
            for finding in findings:
                findings_store[finding.id] = finding

            # Record success in circuit breaker
            _codebase_audit_circuit_breaker.record_success()

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "findings": [f.to_dict() for f in findings],
                }
            )

        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            # Record failure in circuit breaker
            _codebase_audit_circuit_breaker.record_failure()
            logger.exception("%s scan error: %s", scan_type.value, e)
            return error_response("Scan operation failed", 500)

    @rate_limit(requests_per_minute=30, limiter_name="codebase_audit_metrics")
    async def _handle_metrics_analysis(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run code metrics analysis."""
        try:
            body = await self._get_json_body(request)
            target_path = body.get("target_path", ".")

            # Validate target_path to prevent directory traversal
            is_valid, error_msg = validate_repository_path(target_path)
            if not is_valid:
                logger.warning("Invalid target_path rejected in metrics analysis: %s", error_msg)
                return error_response(f"Invalid target_path: {error_msg}", 400)

            scan_id = f"metrics_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=ScanType.METRICS,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            metrics = await _get_scanner_module().run_metrics_analysis(
                target_path, scan_id, tenant_id
            )

            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.metrics = metrics
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "metrics": metrics,
                }
            )

        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as e:
            logger.exception("Metrics analysis error: %s", e)
            return error_response("Analysis operation failed", 500)

    # =========================================================================
    # Scan Management
    # =========================================================================

    async def _handle_list_scans(self, request: Any, tenant_id: str) -> HandlerResult:
        """List past scans."""
        params = self._get_query_params(request)
        scan_type = params.get("type")
        status = params.get("status")

        # Validate scan type filter
        if scan_type:
            is_valid, error_msg = validate_scan_types([scan_type])
            if not is_valid:
                return error_response(f"Invalid type filter: {error_msg}", 400)

        # Validate scan status filter
        if status:
            is_valid, error_msg = validate_scan_status_filter(status)
            if not is_valid:
                return error_response(f"Invalid status filter: {error_msg}", 400)

        # Validate and parse limit
        limit, error_msg = validate_limit(params.get("limit"))
        if error_msg:
            return error_response(error_msg, 400)

        scans = _get_tenant_scans(tenant_id)
        results = list(scans.values())

        # Filter by type
        if scan_type:
            results = [s for s in results if s.scan_type.value == scan_type]

        # Filter by status
        if status:
            results = [s for s in results if s.status.value == status]

        # Sort by start time (newest first)
        results.sort(key=lambda s: s.started_at, reverse=True)

        return success_response(
            {
                "scans": [s.to_dict() for s in results[:limit]],
                "total": len(results),
            }
        )

    async def _handle_get_scan(self, request: Any, tenant_id: str, scan_id: str) -> HandlerResult:
        """Get scan details."""
        # Validate scan_id format
        is_valid, error_msg = validate_scan_id(scan_id)
        if not is_valid:
            return error_response(f"Invalid scan_id: {error_msg}", 400)

        scans = _get_tenant_scans(tenant_id)
        scan_result = scans.get(scan_id)

        if not scan_result:
            return error_response("Scan not found", 404)

        return success_response(
            {
                "scan": scan_result.to_dict(),
                "findings": [f.to_dict() for f in scan_result.findings],
                "metrics": scan_result.metrics,
            }
        )

    # =========================================================================
    # Findings Management
    # =========================================================================

    async def _handle_list_findings(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all findings."""
        params = self._get_query_params(request)
        severity = params.get("severity")
        scan_type = params.get("type")
        status = params.get("status", "open")

        # Validate severity filter
        if severity:
            is_valid, error_msg = validate_severity_filter(severity)
            if not is_valid:
                return error_response(f"Invalid severity filter: {error_msg}", 400)

        # Validate scan type filter
        if scan_type:
            is_valid, error_msg = validate_scan_types([scan_type])
            if not is_valid:
                return error_response(f"Invalid type filter: {error_msg}", 400)

        # Validate status filter
        if status:
            is_valid, error_msg = validate_status_filter(status)
            if not is_valid:
                return error_response(f"Invalid status filter: {error_msg}", 400)

        # Validate and parse limit
        limit, error_msg = validate_limit(params.get("limit"))
        if error_msg:
            return error_response(error_msg, 400)

        findings = list(_get_tenant_findings(tenant_id).values())

        # Filter by severity
        if severity:
            findings = [f for f in findings if f.severity.value == severity]

        # Filter by scan type
        if scan_type:
            findings = [f for f in findings if f.scan_type.value == scan_type]

        # Filter by status
        if status:
            findings = [f for f in findings if f.status.value == status]

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        findings.sort(key=lambda f: severity_order.get(f.severity.value, 5))

        return success_response(
            {
                "findings": [f.to_dict() for f in findings[:limit]],
                "total": len(findings),
            }
        )

    async def _handle_dismiss_finding(
        self, request: Any, tenant_id: str, finding_id: str
    ) -> HandlerResult:
        """Dismiss a finding."""
        try:
            # Validate finding_id format
            is_valid, error_msg = validate_finding_id(finding_id)
            if not is_valid:
                return error_response(f"Invalid finding_id: {error_msg}", 400)

            body = await self._get_json_body(request)

            # Validate dismiss request body
            is_valid, error_msg = validate_dismiss_request(body)
            if not is_valid:
                return error_response(f"Invalid request: {error_msg}", 400)

            reason = body.get("reason", "")
            status_type = body.get("status", "dismissed")

            findings = _get_tenant_findings(tenant_id)
            finding = findings.get(finding_id)

            if not finding:
                return error_response("Finding not found", 404)

            # Update status
            status_map = {
                "dismissed": FindingStatus.DISMISSED,
                "false_positive": FindingStatus.FALSE_POSITIVE,
                "accepted_risk": FindingStatus.ACCEPTED_RISK,
                "fixed": FindingStatus.FIXED,
            }
            finding.status = status_map.get(status_type, FindingStatus.DISMISSED)
            finding.dismissed_by = getattr(request, "user_id", "api_user")
            finding.dismissed_reason = reason

            return success_response(
                {
                    "status": "dismissed",
                    "finding": finding.to_dict(),
                }
            )

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.exception("Error dismissing finding: %s", e)
            return error_response("Dismiss operation failed", 500)

    async def _handle_create_issue(
        self, request: Any, tenant_id: str, finding_id: str
    ) -> HandlerResult:
        """Create GitHub issue for finding."""
        try:
            # Validate finding_id format
            is_valid, error_msg = validate_finding_id(finding_id)
            if not is_valid:
                return error_response(f"Invalid finding_id: {error_msg}", 400)

            body = await self._get_json_body(request)
            repo = body.get("repo", "")

            # Validate GitHub repository format
            is_valid, error_msg = validate_github_repo(repo)
            if not is_valid:
                return error_response(f"Invalid repository: {error_msg}", 400)

            findings = _get_tenant_findings(tenant_id)
            finding = findings.get(finding_id)

            if not finding:
                return error_response("Finding not found", 404)

            # Create issue title and body
            issue_title = f"[{finding.severity.value.upper()}] {finding.title}"
            f"""## Security Finding

**Severity:** {finding.severity.value.upper()}
**File:** `{finding.file_path}:{finding.line_number or ""}`
**Rule:** {finding.rule_id or "N/A"}
**CWE:** {finding.cwe_id or "N/A"}

### Description
{finding.description}

### Code
```
{finding.code_snippet or "N/A"}
```

### Remediation
{finding.remediation or "No remediation guidance available."}

---
*Automatically created by Aragora Codebase Audit*
"""

            # In real implementation, would call GitHub API here
            # For now, return mock response
            mock_issue_url = f"https://github.com/{repo}/issues/123"
            finding.github_issue_url = mock_issue_url

            return success_response(
                {
                    "status": "created",
                    "issue_url": mock_issue_url,
                    "title": issue_title,
                }
            )

        except (KeyError, ValueError, TypeError, AttributeError, ConnectionError, OSError) as e:
            logger.exception("Error creating issue: %s", e)
            return error_response("Issue creation failed", 500)

    # =========================================================================
    # Dashboard
    # =========================================================================

    async def _handle_dashboard(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get dashboard summary data."""
        return success_response(build_dashboard_data(tenant_id))

    def _calculate_risk_score(self, findings: list[Finding]) -> float:
        """Calculate overall risk score (0-100)."""
        return calculate_risk_score(findings)

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo dashboard data."""
        return success_response(build_demo_data())

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Extract JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                body = await request.json()
                return body if isinstance(body, dict) else {}
            return request.json if isinstance(request.json, dict) else {}
        return {}

    def _get_query_params(self, request: Any) -> dict[str, str]:
        """Extract and sanitize query parameters from request.

        Applies sanitization to prevent injection attacks and logs
        any unknown parameters that are filtered out.
        """
        raw_params: dict[str, Any] = {}
        if hasattr(request, "query"):
            raw_params = dict(request.query)
        elif hasattr(request, "args"):
            raw_params = dict(request.args)

        # Sanitize parameters to prevent injection attacks
        return sanitize_query_params(raw_params)


# =============================================================================
# Handler Registration
# =============================================================================

_handler_instance: CodebaseAuditHandler | None = None


def get_codebase_audit_handler() -> CodebaseAuditHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = CodebaseAuditHandler()
    return _handler_instance


async def handle_codebase_audit(request: Any, path: str, method: str) -> HandlerResult:
    """Entry point for codebase audit requests."""
    handler = get_codebase_audit_handler()
    return await handler.handle_request(request, path, method)
