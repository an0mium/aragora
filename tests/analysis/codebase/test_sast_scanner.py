"""
Tests for SAST Scanner module.

Tests static application security testing: severity levels, OWASP categories,
findings, scan results, scanner configuration, local pattern matching,
Semgrep integration, and convenience functions.
"""

import os
import re
import tempfile
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.sast_scanner import (
    AVAILABLE_RULESETS,
    CWE_FIX_RECOMMENDATIONS,
    CWE_TO_OWASP,
    LANGUAGE_EXTENSIONS,
    LOCAL_PATTERNS,
    OWASPCategory,
    SASTConfig,
    SASTFinding,
    SASTScanResult,
    SASTScanner,
    SASTSeverity,
    check_semgrep_installation,
    scan_for_vulnerabilities,
)


# ============================================================
# SASTSeverity
# ============================================================


class TestSASTSeverity:
    """Tests for severity enum and comparison."""

    def test_severity_values(self):
        """All four severity levels exist with expected values."""
        assert SASTSeverity.INFO.value == "info"
        assert SASTSeverity.WARNING.value == "warning"
        assert SASTSeverity.ERROR.value == "error"
        assert SASTSeverity.CRITICAL.value == "critical"

    def test_severity_numeric_levels(self):
        """Numeric level increases with severity."""
        assert SASTSeverity.INFO.level == 0
        assert SASTSeverity.WARNING.level == 1
        assert SASTSeverity.ERROR.level == 2
        assert SASTSeverity.CRITICAL.level == 3

    def test_severity_comparison_operators(self):
        """Comparison operators work correctly across severity levels."""
        assert SASTSeverity.CRITICAL > SASTSeverity.ERROR
        assert SASTSeverity.ERROR > SASTSeverity.WARNING
        assert SASTSeverity.WARNING > SASTSeverity.INFO

        assert SASTSeverity.INFO < SASTSeverity.WARNING
        assert SASTSeverity.WARNING < SASTSeverity.ERROR

        assert SASTSeverity.ERROR >= SASTSeverity.ERROR
        assert SASTSeverity.CRITICAL >= SASTSeverity.WARNING

        assert SASTSeverity.INFO <= SASTSeverity.INFO
        assert SASTSeverity.WARNING <= SASTSeverity.ERROR

    def test_severity_comparison_not_implemented_for_non_severity(self):
        """Comparison with non-SASTSeverity returns NotImplemented."""
        assert SASTSeverity.INFO.__ge__(42) is NotImplemented
        assert SASTSeverity.INFO.__gt__("high") is NotImplemented
        assert SASTSeverity.INFO.__le__(None) is NotImplemented
        assert SASTSeverity.INFO.__lt__(3.14) is NotImplemented


# ============================================================
# OWASPCategory
# ============================================================


class TestOWASPCategory:
    """Tests for OWASP Top 10 category enum."""

    def test_owasp_categories_exist(self):
        """All OWASP Top 10 2021 categories are defined."""
        expected_prefixes = [
            "A01",
            "A02",
            "A03",
            "A04",
            "A05",
            "A06",
            "A07",
            "A08",
            "A09",
            "A10",
        ]
        category_values = [c.value for c in OWASPCategory if c != OWASPCategory.UNKNOWN]
        for prefix in expected_prefixes:
            assert any(prefix in v for v in category_values), f"Missing OWASP category {prefix}"

    def test_unknown_category(self):
        """Unknown category exists for unmapped findings."""
        assert OWASPCategory.UNKNOWN.value == "Unknown"


# ============================================================
# SASTFinding
# ============================================================


class TestSASTFinding:
    """Tests for SAST finding dataclass."""

    def _make_finding(self, **overrides) -> SASTFinding:
        """Create a finding with sensible defaults."""
        defaults = dict(
            rule_id="test-rule",
            file_path="app.py",
            line_start=10,
            line_end=10,
            column_start=1,
            column_end=20,
            message="Test finding",
            severity=SASTSeverity.WARNING,
            confidence=0.8,
            language="python",
        )
        defaults.update(overrides)
        return SASTFinding(**defaults)

    def test_finding_defaults(self):
        """Default field values are applied correctly."""
        finding = self._make_finding()
        assert finding.snippet == ""
        assert finding.cwe_ids == []
        assert finding.owasp_category == OWASPCategory.UNKNOWN
        assert finding.source == "semgrep"
        assert finding.is_false_positive is False
        assert finding.triaged is False
        assert finding.metadata == {}

    def test_finding_id_generated(self):
        """Each finding gets a unique auto-generated ID."""
        f1 = self._make_finding()
        f2 = self._make_finding()
        assert f1.finding_id != f2.finding_id
        assert len(f1.finding_id) == 12

    def test_finding_to_dict(self):
        """to_dict returns a complete dictionary representation."""
        finding = self._make_finding(
            cwe_ids=["CWE-89"],
            owasp_category=OWASPCategory.A03_INJECTION,
            remediation="Use parameterized queries",
        )
        d = finding.to_dict()
        assert d["rule_id"] == "test-rule"
        assert d["severity"] == "warning"
        assert d["cwe_ids"] == ["CWE-89"]
        assert d["owasp_category"] == OWASPCategory.A03_INJECTION.value
        assert d["remediation"] == "Use parameterized queries"
        assert d["is_false_positive"] is False

    def test_finding_to_dict_contains_all_expected_keys(self):
        """to_dict includes all serializable fields."""
        finding = self._make_finding()
        d = finding.to_dict()
        expected_keys = {
            "rule_id",
            "file_path",
            "line_start",
            "line_end",
            "column_start",
            "column_end",
            "message",
            "severity",
            "confidence",
            "language",
            "snippet",
            "cwe_ids",
            "owasp_category",
            "vulnerability_class",
            "remediation",
            "source",
            "rule_name",
            "rule_url",
            "metadata",
            "is_false_positive",
            "triaged",
        }
        assert expected_keys == set(d.keys())


# ============================================================
# SASTScanResult
# ============================================================


class TestSASTScanResult:
    """Tests for scan result aggregation."""

    def _make_result(self, findings=None, **overrides) -> SASTScanResult:
        defaults = dict(
            repository_path="/repo",
            scan_id="test-scan",
            findings=findings or [],
            scanned_files=10,
            skipped_files=2,
            scan_duration_ms=500.0,
            languages_detected=["python"],
            rules_used=["local"],
        )
        defaults.update(overrides)
        return SASTScanResult(**defaults)

    def _make_finding(self, severity=SASTSeverity.WARNING, owasp=OWASPCategory.UNKNOWN, **kw):
        defaults = dict(
            rule_id="r",
            file_path="f.py",
            line_start=1,
            line_end=1,
            column_start=1,
            column_end=1,
            message="m",
            severity=severity,
            confidence=0.8,
            language="python",
            owasp_category=owasp,
        )
        defaults.update(kw)
        return SASTFinding(**defaults)

    def test_findings_by_severity(self):
        """findings_by_severity counts each severity level."""
        findings = [
            self._make_finding(severity=SASTSeverity.CRITICAL),
            self._make_finding(severity=SASTSeverity.CRITICAL),
            self._make_finding(severity=SASTSeverity.WARNING),
            self._make_finding(severity=SASTSeverity.INFO),
        ]
        result = self._make_result(findings=findings)
        by_sev = result.findings_by_severity
        assert by_sev["critical"] == 2
        assert by_sev["warning"] == 1
        assert by_sev["info"] == 1

    def test_findings_by_owasp(self):
        """findings_by_owasp counts each OWASP category."""
        findings = [
            self._make_finding(owasp=OWASPCategory.A03_INJECTION),
            self._make_finding(owasp=OWASPCategory.A03_INJECTION),
            self._make_finding(owasp=OWASPCategory.A01_BROKEN_ACCESS_CONTROL),
        ]
        result = self._make_result(findings=findings)
        by_owasp = result.findings_by_owasp
        assert by_owasp[OWASPCategory.A03_INJECTION.value] == 2
        assert by_owasp[OWASPCategory.A01_BROKEN_ACCESS_CONTROL.value] == 1

    def test_to_dict(self):
        """to_dict includes summary with severity and owasp breakdowns."""
        result = self._make_result(
            findings=[self._make_finding(severity=SASTSeverity.ERROR)],
        )
        d = result.to_dict()
        assert d["findings_count"] == 1
        assert d["scanned_files"] == 10
        assert "by_severity" in d["summary"]
        assert "by_owasp" in d["summary"]
        assert "scanned_at" in d

    def test_empty_result(self):
        """Empty result has zero counts and empty collections."""
        result = self._make_result()
        assert result.findings_by_severity == {}
        assert result.findings_by_owasp == {}
        assert result.to_dict()["findings_count"] == 0


# ============================================================
# SASTConfig
# ============================================================


class TestSASTConfig:
    """Tests for scanner configuration."""

    def test_default_config(self):
        """Default configuration values are reasonable."""
        cfg = SASTConfig()
        assert cfg.use_semgrep is True
        assert cfg.semgrep_timeout == 300
        assert cfg.max_file_size_kb == 500
        assert cfg.min_severity == SASTSeverity.WARNING
        assert cfg.max_concurrent_files == 10
        assert cfg.min_confidence_threshold == 0.5
        assert cfg.enable_false_positive_filtering is True

    def test_default_rule_sets(self):
        """Default rule sets include OWASP and security audit."""
        cfg = SASTConfig()
        assert "p/owasp-top-ten" in cfg.default_rule_sets
        assert "p/security-audit" in cfg.default_rule_sets

    def test_excluded_patterns(self):
        """Default excluded patterns cover common vendor/build directories."""
        cfg = SASTConfig()
        assert "node_modules/" in cfg.excluded_patterns
        assert ".git/" in cfg.excluded_patterns
        assert "venv/" in cfg.excluded_patterns

    def test_supported_languages(self):
        """Default supported languages include common targets."""
        cfg = SASTConfig()
        for lang in ["python", "javascript", "typescript", "go", "java"]:
            assert lang in cfg.supported_languages

    def test_custom_config_overrides(self):
        """Custom values override defaults."""
        cfg = SASTConfig(
            use_semgrep=False,
            min_severity=SASTSeverity.CRITICAL,
            max_file_size_kb=100,
        )
        assert cfg.use_semgrep is False
        assert cfg.min_severity == SASTSeverity.CRITICAL
        assert cfg.max_file_size_kb == 100


# ============================================================
# Constants: LOCAL_PATTERNS, CWE mappings, AVAILABLE_RULESETS
# ============================================================


class TestConstants:
    """Tests for module-level constants and pattern definitions."""

    def test_local_patterns_have_required_fields(self):
        """Every local pattern has required keys."""
        required_keys = {"pattern", "languages", "message", "severity", "cwe", "owasp"}
        for rule_id, rule_data in LOCAL_PATTERNS.items():
            missing = required_keys - set(rule_data.keys())
            assert not missing, f"Rule {rule_id} missing keys: {missing}"

    def test_local_patterns_compile(self):
        """All local patterns are valid regular expressions."""
        for rule_id, rule_data in LOCAL_PATTERNS.items():
            try:
                re.compile(rule_data["pattern"], re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                pytest.fail(f"Pattern {rule_id} failed to compile: {exc}")

    def test_cwe_to_owasp_mapping_coverage(self):
        """CWE_TO_OWASP maps to valid OWASP categories."""
        for cwe, owasp in CWE_TO_OWASP.items():
            assert cwe.startswith("CWE-"), f"Invalid CWE format: {cwe}"
            assert isinstance(owasp, OWASPCategory)
            assert owasp != OWASPCategory.UNKNOWN

    def test_cwe_fix_recommendations_exist(self):
        """Fix recommendations cover important CWE IDs."""
        critical_cwes = ["CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-502"]
        for cwe in critical_cwes:
            assert cwe in CWE_FIX_RECOMMENDATIONS, f"Missing fix for {cwe}"

    def test_available_rulesets_structure(self):
        """Each available ruleset has name, description, and category."""
        for rs_id, rs_info in AVAILABLE_RULESETS.items():
            assert "name" in rs_info
            assert "description" in rs_info
            assert "category" in rs_info

    def test_language_extensions_mapping(self):
        """Language extension mapping covers expected file types."""
        assert ".py" in LANGUAGE_EXTENSIONS["python"]
        assert ".js" in LANGUAGE_EXTENSIONS["javascript"]
        assert ".ts" in LANGUAGE_EXTENSIONS["typescript"]
        assert ".go" in LANGUAGE_EXTENSIONS["go"]


# ============================================================
# SASTScanner - Initialization and helpers
# ============================================================


class TestSASTScannerInit:
    """Tests for scanner initialization and utility methods."""

    def test_default_init(self):
        """Scanner initializes with default config and compiled patterns."""
        scanner = SASTScanner()
        assert scanner.config is not None
        assert scanner._semgrep_available is None
        assert len(scanner._compiled_patterns) > 0

    def test_init_with_custom_config(self):
        """Scanner accepts custom configuration."""
        cfg = SASTConfig(use_semgrep=False, min_severity=SASTSeverity.CRITICAL)
        scanner = SASTScanner(config=cfg)
        assert scanner.config.use_semgrep is False
        assert scanner.config.min_severity == SASTSeverity.CRITICAL

    def test_detect_language_python(self):
        """Language detection for Python files."""
        scanner = SASTScanner()
        assert scanner._detect_language("app.py") == "python"
        assert scanner._detect_language("script.pyw") == "python"

    def test_detect_language_javascript(self):
        """Language detection for JavaScript files."""
        scanner = SASTScanner()
        assert scanner._detect_language("index.js") == "javascript"
        assert scanner._detect_language("component.jsx") == "javascript"

    def test_detect_language_unknown(self):
        """Unknown extension returns 'unknown'."""
        scanner = SASTScanner()
        assert scanner._detect_language("data.csv") == "unknown"
        assert scanner._detect_language("README") == "unknown"

    def test_is_semgrep_available_before_init(self):
        """Before initialization semgrep_available defaults to False."""
        scanner = SASTScanner()
        assert scanner.is_semgrep_available() is False

    def test_get_install_instructions(self):
        """Install instructions contain pip install command."""
        scanner = SASTScanner()
        instructions = scanner.get_install_instructions()
        assert "pip install semgrep" in instructions


# ============================================================
# SASTScanner - initialize()
# ============================================================


class TestSASTScannerInitialize:
    """Tests for the async initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_without_semgrep(self):
        """With use_semgrep=False, semgrep is marked unavailable."""
        cfg = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=cfg)
        await scanner.initialize()
        assert scanner._semgrep_available is False

    @pytest.mark.asyncio
    async def test_initialize_semgrep_check_called(self):
        """When use_semgrep=True, _check_semgrep is called."""
        scanner = SASTScanner()
        scanner._check_semgrep = AsyncMock(return_value=(False, None))
        await scanner.initialize()
        scanner._check_semgrep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_semgrep_found(self):
        """When semgrep is available, version is stored."""
        scanner = SASTScanner()
        scanner._check_semgrep = AsyncMock(return_value=(True, "1.50.0"))
        await scanner.initialize()
        assert scanner._semgrep_available is True
        assert scanner._semgrep_version == "1.50.0"


# ============================================================
# SASTScanner - Local pattern scanning
# ============================================================


class TestLocalPatternScanning:
    """Tests for local fallback pattern matching on file content."""

    @pytest.mark.asyncio
    async def test_scan_file_detects_eval(self):
        """eval() usage is detected by local patterns."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("result = eval(user_input)\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                rule_ids = [finding.rule_id for finding in findings]
                assert "eval-injection" in rule_ids
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_detects_exec(self):
        """exec() usage is detected by local patterns."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("exec(code_string)\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                rule_ids = [finding.rule_id for finding in findings]
                assert "exec-injection" in rule_ids
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_detects_hardcoded_password(self):
        """Hardcoded password assignment is detected."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('password = "SuperSecret123!"\n')
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                rule_ids = [finding.rule_id for finding in findings]
                assert "hardcoded-password" in rule_ids
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_detects_pickle_load(self):
        """pickle.load usage is detected as insecure deserialization."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import pickle\ndata = pickle.load(open('data.pkl', 'rb'))\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                rule_ids = [finding.rule_id for finding in findings]
                assert "pickle-load" in rule_ids
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_skips_wrong_language(self):
        """Patterns only match their declared languages."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
            # exec-injection is python-only
            f.write("exec(code)\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.go", "go")
                rule_ids = [finding.rule_id for finding in findings]
                assert "exec-injection" not in rule_ids
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_finding_metadata(self):
        """Findings from local scan include correct metadata."""
        scanner = SASTScanner(
            config=SASTConfig(
                use_semgrep=False,
                min_severity=SASTSeverity.INFO,
            )
        )
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("result = eval(user_input)\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                eval_findings = [f2 for f2 in findings if f2.rule_id == "eval-injection"]
                assert len(eval_findings) >= 1
                finding = eval_findings[0]
                assert finding.source == "local"
                assert finding.confidence == 0.7
                assert "CWE-95" in finding.cwe_ids
                assert finding.owasp_category == OWASPCategory.A03_INJECTION
                assert finding.language == "python"
                assert finding.line_start >= 1
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_scan_file_severity_filtering(self):
        """Findings below min_severity are excluded."""
        scanner = SASTScanner(
            config=SASTConfig(
                use_semgrep=False,
                min_severity=SASTSeverity.CRITICAL,
            )
        )
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # eval-injection has ERROR severity, not CRITICAL
            f.write("result = eval(user_input)\n")
            f.flush()
            try:
                findings = await scanner._scan_file_local(f.name, "test.py", "python")
                # ERROR < CRITICAL, so should be filtered
                eval_findings = [f2 for f2 in findings if f2.rule_id == "eval-injection"]
                assert len(eval_findings) == 0
            finally:
                os.unlink(f.name)


# ============================================================
# SASTScanner - scan_repository
# ============================================================


class TestScanRepository:
    """Tests for repository scanning."""

    @pytest.mark.asyncio
    async def test_scan_nonexistent_repo(self):
        """Scanning a nonexistent path returns error result."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False
        result = await scanner.scan_repository("/nonexistent/path/xyz123")
        assert result.scanned_files == 0
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_scan_repository_local_fallback(self):
        """Repository scan with local patterns returns findings."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with a known vulnerability
            vuln_file = os.path.join(tmpdir, "vulnerable.py")
            with open(vuln_file, "w") as f:
                f.write('password = "hunter2_secret"\nresult = eval(data)\n')

            result = await scanner.scan_repository(tmpdir)
            assert result.scanned_files >= 1
            assert len(result.findings) >= 1
            assert result.repository_path == os.path.abspath(tmpdir)
            assert result.scan_duration_ms > 0

    @pytest.mark.asyncio
    async def test_scan_repository_false_positive_filtering(self):
        """Low-confidence findings are filtered when enabled."""
        scanner = SASTScanner(
            config=SASTConfig(
                use_semgrep=False,
                enable_false_positive_filtering=True,
                min_confidence_threshold=0.9,
            )
        )
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            vuln_file = os.path.join(tmpdir, "test.py")
            with open(vuln_file, "w") as f:
                f.write("result = eval(data)\n")

            result = await scanner.scan_repository(tmpdir)
            # Local patterns have confidence 0.7, threshold is 0.9 -- filtered
            assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_scan_repository_adds_remediation(self):
        """Scan adds CWE-based remediation to findings."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            vuln_file = os.path.join(tmpdir, "test.py")
            with open(vuln_file, "w") as f:
                f.write("result = eval(data)\n")

            result = await scanner.scan_repository(tmpdir)
            eval_findings = [f for f in result.findings if f.rule_id == "eval-injection"]
            if eval_findings:
                assert eval_findings[0].remediation != ""

    @pytest.mark.asyncio
    async def test_scan_repository_progress_callback(self):
        """Progress callback is invoked during scan."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False
        progress_calls = []

        async def track_progress(current, total, message):
            progress_calls.append((current, total, message))

        with tempfile.TemporaryDirectory() as tmpdir:
            vuln_file = os.path.join(tmpdir, "test.py")
            with open(vuln_file, "w") as f:
                f.write("x = 1\n")

            await scanner.scan_repository(tmpdir, progress_callback=track_progress)

        # Should have at least start and completion callbacks
        assert len(progress_calls) >= 2
        # Last callback should report completion
        assert progress_calls[-1][0] == 100


# ============================================================
# SASTScanner - scan_file (single file public API)
# ============================================================


class TestScanFile:
    """Tests for single-file scanning public API."""

    @pytest.mark.asyncio
    async def test_scan_nonexistent_file(self):
        """Scanning a nonexistent file returns empty list."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False
        findings = await scanner.scan_file("/nonexistent/file.py")
        assert findings == []

    @pytest.mark.asyncio
    async def test_scan_file_with_language_override(self):
        """Language override is used instead of auto-detection."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("result = eval(data)\n")
            f.flush()
            try:
                findings = await scanner.scan_file(f.name, language="python")
                rule_ids = [finding.rule_id for finding in findings]
                assert "eval-injection" in rule_ids
            finally:
                os.unlink(f.name)


# ============================================================
# SASTScanner - Semgrep result parsing
# ============================================================


class TestSemgrepParsing:
    """Tests for parsing Semgrep JSON output."""

    def test_parse_semgrep_result_basic(self):
        """Basic Semgrep result is parsed into a SASTFinding."""
        scanner = SASTScanner()
        raw = {
            "check_id": "python.lang.security.eval-injection",
            "path": "app.py",
            "start": {"line": 5, "col": 1},
            "end": {"line": 5, "col": 25},
            "extra": {
                "message": "Avoid eval",
                "severity": "ERROR",
                "lines": "eval(input())",
                "metadata": {
                    "cwe": ["CWE-95"],
                    "confidence": 0.9,
                    "language": "python",
                    "source": "https://semgrep.dev/r/rule",
                },
            },
        }
        finding = scanner._parse_semgrep_result(raw)
        assert finding is not None
        assert finding.rule_id == "python.lang.security.eval-injection"
        assert finding.severity == SASTSeverity.ERROR
        assert finding.line_start == 5
        assert finding.cwe_ids == ["CWE-95"]
        assert finding.source == "semgrep"
        assert finding.confidence == 0.9

    def test_parse_semgrep_result_cwe_string(self):
        """CWE provided as string is wrapped in a list."""
        scanner = SASTScanner()
        raw = {
            "check_id": "rule",
            "path": "f.py",
            "start": {"line": 1, "col": 1},
            "end": {"line": 1, "col": 1},
            "extra": {
                "message": "msg",
                "severity": "WARNING",
                "metadata": {"cwe": "CWE-89"},
            },
        }
        finding = scanner._parse_semgrep_result(raw)
        assert finding is not None
        assert finding.cwe_ids == ["CWE-89"]

    def test_parse_semgrep_result_owasp_mapping(self):
        """CWE ID is mapped to correct OWASP category."""
        scanner = SASTScanner()
        raw = {
            "check_id": "sql-injection",
            "path": "f.py",
            "start": {"line": 1, "col": 1},
            "end": {"line": 1, "col": 1},
            "extra": {
                "message": "SQL injection",
                "severity": "CRITICAL",
                "metadata": {"cwe": ["CWE-89"]},
            },
        }
        finding = scanner._parse_semgrep_result(raw)
        assert finding is not None
        assert finding.owasp_category == OWASPCategory.A03_INJECTION

    def test_parse_semgrep_result_malformed(self):
        """Malformed Semgrep result returns None."""
        scanner = SASTScanner()
        # Trigger exception via bad data
        finding = scanner._parse_semgrep_result(None)
        assert finding is None


# ============================================================
# SASTScanner - OWASP summary
# ============================================================


class TestOWASPSummary:
    """Tests for OWASP summary generation."""

    @pytest.mark.asyncio
    async def test_owasp_summary_structure(self):
        """OWASP summary contains expected keys."""
        scanner = SASTScanner()
        findings = [
            SASTFinding(
                rule_id="r1",
                file_path="f.py",
                line_start=1,
                line_end=1,
                column_start=1,
                column_end=1,
                message="injection",
                severity=SASTSeverity.CRITICAL,
                confidence=0.9,
                language="python",
                owasp_category=OWASPCategory.A03_INJECTION,
            ),
        ]
        summary = await scanner.get_owasp_summary(findings)
        assert "owasp_top_10" in summary
        assert "total_findings" in summary
        assert summary["total_findings"] == 1
        assert summary["owasp_top_10"][OWASPCategory.A03_INJECTION.value]["count"] == 1

    @pytest.mark.asyncio
    async def test_owasp_summary_empty(self):
        """Empty findings produce zero counts."""
        scanner = SASTScanner()
        summary = await scanner.get_owasp_summary([])
        assert summary["total_findings"] == 0


# ============================================================
# Convenience functions
# ============================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_check_semgrep_installation(self):
        """check_semgrep_installation returns a dict with installed key."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1.50.0\n")
            result = check_semgrep_installation()
            assert result["installed"] is True
            assert result["version"] == "1.50.0"

    def test_check_semgrep_installation_not_found(self):
        """When semgrep is missing, installed is False with instructions."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = check_semgrep_installation()
            assert result["installed"] is False
            assert "instructions" in result

    @pytest.mark.asyncio
    async def test_scan_for_vulnerabilities_directory(self):
        """scan_for_vulnerabilities works on a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vuln_file = os.path.join(tmpdir, "test.py")
            with open(vuln_file, "w") as f:
                f.write("result = eval(data)\n")

            result = await scan_for_vulnerabilities(tmpdir)
            assert isinstance(result, SASTScanResult)
            assert result.scanned_files >= 1

    @pytest.mark.asyncio
    async def test_scan_for_vulnerabilities_file(self):
        """scan_for_vulnerabilities works on a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("result = eval(data)\n")
            f.flush()
            try:
                result = await scan_for_vulnerabilities(f.name)
                assert isinstance(result, SASTScanResult)
                assert result.scan_id == "quick"
            finally:
                os.unlink(f.name)


# ============================================================
# SASTScanner - get_available_rulesets
# ============================================================


class TestGetAvailableRulesets:
    """Tests for listing available rulesets."""

    @pytest.mark.asyncio
    async def test_rulesets_without_semgrep(self):
        """Without semgrep, rulesets are listed but marked unavailable."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False
        rulesets = await scanner.get_available_rulesets()
        assert len(rulesets) == len(AVAILABLE_RULESETS)
        for rs in rulesets:
            assert rs["available"] is False

    @pytest.mark.asyncio
    async def test_rulesets_with_semgrep(self):
        """With semgrep available, rulesets are marked available."""
        scanner = SASTScanner()
        scanner._semgrep_available = True
        rulesets = await scanner.get_available_rulesets()
        assert len(rulesets) >= len(AVAILABLE_RULESETS)
        for rs in rulesets:
            assert "id" in rs
            assert "name" in rs
