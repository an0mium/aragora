"""
SAST Scanner package - Static Application Security Testing.

Integrates with Semgrep for comprehensive static analysis with OWASP mapping.
Falls back to local pattern matching when Semgrep is not available.

Features:
- OWASP Top 10 vulnerability detection
- CWE ID mapping for findings
- Multi-language support (Python, JavaScript, Go, Java, TypeScript, Ruby)
- Custom rule support
- Severity classification
- False positive filtering via confidence scoring
- Async scanning with progress reporting
- SecurityEventEmitter integration for critical findings

Usage:
    from aragora.analysis.codebase.sast import SASTScanner

    scanner = SASTScanner()
    await scanner.initialize()

    # Scan a repository
    result = await scanner.scan_repository("/path/to/repo")
    print(f"Found {len(result.findings)} issues")

    # Scan with specific rules
    result = await scanner.scan_with_rules(
        path="/path/to/repo",
        rule_sets=["owasp-top-10", "cwe-top-25"],
    )

    # Get available rulesets
    rulesets = await scanner.get_available_rulesets()

    # Scan with progress reporting
    async def on_progress(current, total, message):
        print(f"[{current}/{total}] {message}")

    result = await scanner.scan_repository("/path/to/repo", progress_callback=on_progress)

Semgrep Installation:
    If Semgrep is not installed, install it with:
        pip install semgrep
    Or:
        brew install semgrep  # macOS
        python3 -m pip install semgrep  # Python
    See: https://semgrep.dev/docs/getting-started/
"""

from aragora.analysis.codebase.sast.models import (
    OWASPCategory,
    ProgressCallback,
    SASTConfig,
    SASTFinding,
    SASTScanResult,
    SASTSeverity,
)
from aragora.analysis.codebase.sast.rules import (
    AVAILABLE_RULESETS,
    CWE_FIX_RECOMMENDATIONS,
    CWE_TO_OWASP,
    LANGUAGE_EXTENSIONS,
    LOCAL_PATTERNS,
)
from aragora.analysis.codebase.sast.scanner import (
    SASTScanner,
    check_semgrep_installation,
    get_available_rulesets,
    scan_for_vulnerabilities,
)

__all__ = [
    # Main classes
    "SASTScanner",
    "SASTScanResult",
    "SASTFinding",
    "SASTSeverity",
    "SASTConfig",
    "OWASPCategory",
    # Convenience functions
    "scan_for_vulnerabilities",
    "get_available_rulesets",
    "check_semgrep_installation",
    # Type aliases
    "ProgressCallback",
    # Constants
    "AVAILABLE_RULESETS",
    "CWE_TO_OWASP",
    "CWE_FIX_RECOMMENDATIONS",
    "LOCAL_PATTERNS",
    "LANGUAGE_EXTENSIONS",
]
