"""
Codebase Analysis Module.

Provides tools for analyzing codebases:
- CVE/vulnerability database integration
- Dependency vulnerability scanning
- Secrets detection and credential scanning
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
    SecretType,
    SecretFinding,
    SecretsScanResult,
)
from .cve_client import CVEClient
from .scanner import DependencyScanner
from .secrets_scanner import SecretsScanner, scan_repository_for_secrets
from .metrics import (
    CodeMetricsAnalyzer,
    MetricsReport,
    FileMetrics,
    FunctionMetrics,
    DuplicateBlock,
)
from .sast_scanner import (
    SASTScanner,
    SASTScanResult,
    SASTFinding,
    SASTSeverity,
    SASTConfig,
    OWASPCategory,
    scan_for_vulnerabilities,
)
from .sbom_generator import (
    SBOMGenerator,
    SBOMFormat,
    SBOMResult,
    SBOMComponent,
    SBOMMetadata,
    ComponentType,
    HashAlgorithm,
    generate_sbom,
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
    "SecretType",
    "SecretFinding",
    "SecretsScanResult",
    # CVE Client
    "CVEClient",
    # Scanners
    "DependencyScanner",
    "SecretsScanner",
    "scan_repository_for_secrets",
    # Metrics
    "CodeMetricsAnalyzer",
    "MetricsReport",
    "FileMetrics",
    "FunctionMetrics",
    "DuplicateBlock",
    # SAST Scanner
    "SASTScanner",
    "SASTScanResult",
    "SASTFinding",
    "SASTSeverity",
    "SASTConfig",
    "OWASPCategory",
    "scan_for_vulnerabilities",
    # SBOM Generator
    "SBOMGenerator",
    "SBOMFormat",
    "SBOMResult",
    "SBOMComponent",
    "SBOMMetadata",
    "ComponentType",
    "HashAlgorithm",
    "generate_sbom",
]
