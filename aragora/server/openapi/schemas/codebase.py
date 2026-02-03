"""
Codebase Analysis OpenAPI Schema Definitions.

Schemas for vulnerability scanning, dependency analysis, code metrics,
decisions, deliberations, and GitHub PR review.
"""

from typing import Any

CODEBASE_SCHEMAS: dict[str, Any] = {
    # Vulnerability and dependency schemas
    "VulnerabilityReference": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "source": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    },
    "VulnerabilityFinding": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string"},
            "cvss_score": {"type": "number", "nullable": True},
            "package_name": {"type": "string", "nullable": True},
            "package_ecosystem": {"type": "string", "nullable": True},
            "vulnerable_versions": {"type": "array", "items": {"type": "string"}},
            "patched_versions": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string"},
            "references": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityReference"},
            },
            "cwe_ids": {"type": "array", "items": {"type": "string"}},
            "fix_available": {"type": "boolean"},
            "recommended_version": {"type": "string", "nullable": True},
        },
    },
    "DependencyInfo": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "ecosystem": {"type": "string"},
            "direct": {"type": "boolean"},
            "dev_dependency": {"type": "boolean"},
            "license": {"type": "string", "nullable": True},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "has_vulnerabilities": {"type": "boolean"},
            "highest_severity": {"type": "string", "nullable": True},
        },
    },
    "CodebaseScanSummary": {
        "type": "object",
        "properties": {
            "total_dependencies": {"type": "integer"},
            "vulnerable_dependencies": {"type": "integer"},
            "critical_count": {"type": "integer"},
            "high_count": {"type": "integer"},
            "medium_count": {"type": "integer"},
            "low_count": {"type": "integer"},
        },
    },
    "CodebaseScanResult": {
        "type": "object",
        "properties": {
            "scan_id": {"type": "string"},
            "repository": {"type": "string"},
            "branch": {"type": "string", "nullable": True},
            "commit_sha": {"type": "string", "nullable": True},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time", "nullable": True},
            "status": {"type": "string"},
            "error": {"type": "string", "nullable": True},
            "dependencies": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DependencyInfo"},
            },
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "summary": {"$ref": "#/components/schemas/CodebaseScanSummary"},
        },
    },
    "CodebaseScanStartResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_id": {"type": "string"},
            "status": {"type": "string"},
            "repository": {"type": "string"},
        },
    },
    "CodebaseScanResultResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_result": {"$ref": "#/components/schemas/CodebaseScanResult"},
        },
    },
    "CodebaseScanListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scans": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
    },
    "CodebaseVulnerabilityListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "scan_id": {"type": "string"},
        },
    },
    "CodebasePackageVulnerabilityResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "package": {"type": "string"},
            "ecosystem": {"type": "string"},
            "version": {"type": "string", "nullable": True},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "total": {"type": "integer"},
        },
    },
    "CodebaseCVEResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "vulnerability": {"$ref": "#/components/schemas/VulnerabilityFinding"},
        },
    },
    "CodebaseMetricsStartResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "analysis_id": {"type": "string"},
            "status": {"type": "string"},
            "repository": {"type": "string"},
        },
    },
    "CodebaseMetricsReportResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "report": {"type": "object"},
        },
    },
    "CodebaseHotspot": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "function_name": {"type": "string", "nullable": True},
            "class_name": {"type": "string", "nullable": True},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "complexity": {"type": "number"},
            "lines_of_code": {"type": "integer"},
            "risk_score": {"type": "number"},
        },
    },
    "CodebaseHotspotListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "hotspots": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CodebaseHotspot"},
            },
            "total": {"type": "integer"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseDuplicateListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "duplicates": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseFileMetricsResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "file": {"type": "object"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseMetricsHistoryResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "analyses": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
    },
    "CodebaseDependencyAnalysisRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "include_dev": {"type": "boolean"},
            "ecosystems": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["repository"],
    },
    "CodebaseDependencyAnalysisResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "analysis_id": {"type": "string"},
            "status": {"type": "string"},
        },
    },
    "CodebaseDependencyScanRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "severity_threshold": {"type": "string"},
        },
        "required": ["repository"],
    },
    "CodebaseDependencyScanResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_id": {"type": "string"},
            "status": {"type": "string"},
        },
    },
    "CodebaseLicenseCheckRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "allowed_licenses": {"type": "array", "items": {"type": "string"}},
            "blocked_licenses": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["repository"],
    },
    "CodebaseLicenseCheckResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "check_id": {"type": "string"},
            "status": {"type": "string"},
        },
    },
    "CodebaseSBOMRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "format": {"type": "string", "enum": ["spdx", "cyclonedx"]},
            "include_dev": {"type": "boolean"},
        },
        "required": ["repository"],
    },
    "CodebaseSBOMResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "sbom_id": {"type": "string"},
            "format": {"type": "string"},
        },
    },
    "CodebaseSecretsScanRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "include_history": {"type": "boolean"},
            "patterns": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["repository"],
    },
    "CodebaseSecretsScanStartResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_id": {"type": "string"},
            "status": {"type": "string"},
            "repository": {"type": "string"},
            "include_history": {"type": "boolean"},
        },
    },
    "CodebaseSecretsScanResultResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_id": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
        },
    },
    "CodebaseSecretsListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "secrets": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
        },
    },
    "CodebaseSecretsScanListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scans": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
    },
    # Decision and deliberation schemas
    "DecisionRequest": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "content": {"type": "string"},
            "decision_type": {"type": "string"},
            "source": {"type": "string"},
            "response_channels": {"type": "array", "items": {"type": "object"}},
            "context": {"type": "object"},
            "config": {"type": "object"},
            "priority": {"type": "string"},
            "attachments": {"type": "array", "items": {"type": "object"}},
            "evidence": {"type": "array", "items": {"type": "object"}},
            "documents": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
    "DecisionResult": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "decision_type": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "consensus_reached": {"type": "boolean"},
            "reasoning": {"type": "string", "nullable": True},
            "evidence_used": {"type": "array", "items": {"type": "object"}},
            "agent_contributions": {"type": "array", "items": {"type": "object"}},
            "duration_seconds": {"type": "number"},
            "completed_at": {"type": "string", "format": "date-time"},
            "success": {"type": "boolean"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "decision_type", "answer", "confidence"],
    },
    "DecisionStatus": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    "DecisionSummary": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id"],
    },
    "DecisionList": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DecisionSummary"},
            },
            "total": {"type": "integer"},
        },
        "required": ["decisions", "total"],
    },
    "DeliberationRequest": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "content": {"type": "string"},
            "decision_type": {"type": "string"},
            "async": {"type": "boolean"},
            "priority": {"type": "string"},
            "timeout_seconds": {"type": "number"},
            "required_capabilities": {"type": "array", "items": {"type": "string"}},
            "response_channels": {"type": "array", "items": {"type": "object"}},
            "metadata": {"type": "object"},
        },
        "required": ["content"],
    },
    "DeliberationQueuedResponse": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "request_id": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["task_id", "request_id", "status"],
    },
    "DeliberationSyncResponse": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "decision_type": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "consensus_reached": {"type": "boolean"},
            "reasoning": {"type": "string", "nullable": True},
            "evidence_used": {"type": "array", "items": {"type": "object"}},
            "duration_seconds": {"type": "number"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    "DeliberationRecord": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "result": {"$ref": "#/components/schemas/DecisionResult"},
            "completed_at": {"type": "string", "format": "date-time"},
            "error": {"type": "string", "nullable": True},
            "metrics": {"type": "object"},
        },
        "required": ["request_id", "status"],
    },
    "DeliberationStatus": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    # GitHub PR review schemas
    "GitHubReviewComment": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "file_path": {"type": "string"},
            "line": {"type": "integer"},
            "body": {"type": "string"},
            "side": {"type": "string"},
            "suggestion": {"type": "string", "nullable": True},
            "severity": {"type": "string"},
            "category": {"type": "string"},
        },
        "required": ["id", "file_path", "line", "body"],
    },
    "GitHubPRReviewResult": {
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "pr_number": {"type": "integer"},
            "repository": {"type": "string"},
            "status": {"type": "string"},
            "verdict": {"type": "string", "nullable": True},
            "summary": {"type": "string", "nullable": True},
            "comments": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GitHubReviewComment"},
            },
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time", "nullable": True},
            "error": {"type": "string", "nullable": True},
            "metrics": {"type": "object"},
        },
        "required": ["review_id", "pr_number", "repository", "status", "comments", "started_at"],
    },
    "GitHubPRDetails": {
        "type": "object",
        "properties": {
            "number": {"type": "integer"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "state": {"type": "string"},
            "author": {"type": "string"},
            "base_branch": {"type": "string"},
            "head_branch": {"type": "string"},
            "changed_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "status": {"type": "string"},
                        "additions": {"type": "integer"},
                        "deletions": {"type": "integer"},
                        "patch": {"type": "string"},
                    },
                },
            },
            "commits": {"type": "array", "items": {"type": "object"}},
            "labels": {"type": "array", "items": {"type": "string"}},
            "created_at": {"type": "string", "format": "date-time", "nullable": True},
            "updated_at": {"type": "string", "format": "date-time", "nullable": True},
        },
        "required": ["number", "title", "state", "author", "base_branch", "head_branch"],
    },
    "GitHubPRReviewTriggerRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "owner/repo"},
            "pr_number": {"type": "integer"},
            "review_type": {"type": "string", "enum": ["comprehensive", "quick", "security"]},
            "workspace_id": {"type": "string"},
        },
        "required": ["repository", "pr_number"],
    },
    "GitHubPRReviewTriggerResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "review_id": {"type": "string"},
            "status": {"type": "string"},
            "pr_number": {"type": "integer"},
            "repository": {"type": "string"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRDetailsResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "pr": {"$ref": "#/components/schemas/GitHubPRDetails"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRReviewStatusResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "review": {"$ref": "#/components/schemas/GitHubPRReviewResult"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRReviewListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "reviews": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GitHubPRReviewResult"},
            },
            "total": {"type": "integer"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success", "reviews", "total"],
    },
    "GitHubPRSubmitReviewRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "owner/repo"},
            "event": {
                "type": "string",
                "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
            },
            "body": {"type": "string"},
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "position": {"type": "integer"},
                        "body": {"type": "string"},
                    },
                    "required": ["path", "position", "body"],
                },
            },
        },
        "required": ["repository", "event"],
    },
    "GitHubPRSubmitReviewResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "demo": {"type": "boolean"},
            "data": {"type": "object"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
}


__all__ = ["CODEBASE_SCHEMAS"]
