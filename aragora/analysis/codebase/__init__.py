"""
Codebase Analysis Module.

Provides tools for analyzing codebases:
- CVE/vulnerability database integration
- Dependency vulnerability scanning
- Code quality metrics
- Security analysis
"""

from .models import (
    VulnerabilityFinding,
    VulnerabilitySeverity,
    DependencyInfo,
    ScanResult,
    CodeMetric,
    HotspotFinding,
    MetricType,
)
from .cve_client import CVEClient
from .scanner import DependencyScanner
from .metrics import (
    CodeMetricsAnalyzer,
    MetricsReport,
    FileMetrics,
    FunctionMetrics,
    DuplicateBlock,
)

__all__ = [
    # Models
    "VulnerabilityFinding",
    "VulnerabilitySeverity",
    "DependencyInfo",
    "ScanResult",
    "CodeMetric",
    "HotspotFinding",
    "MetricType",
    # CVE Client
    "CVEClient",
    # Scanner
    "DependencyScanner",
    # Metrics
    "CodeMetricsAnalyzer",
    "MetricsReport",
    "FileMetrics",
    "FunctionMetrics",
    "DuplicateBlock",
]
