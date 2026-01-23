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
)
from .cve_client import CVEClient
from .scanner import DependencyScanner

__all__ = [
    "VulnerabilityFinding",
    "VulnerabilitySeverity",
    "DependencyInfo",
    "ScanResult",
    "CodeMetric",
    "CVEClient",
    "DependencyScanner",
]
