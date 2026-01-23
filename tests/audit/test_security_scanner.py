"""
Tests for security vulnerability scanner.

Tests pattern matching for secrets, injection vulnerabilities,
and other security issues.
"""

import pytest
from pathlib import Path

from aragora.audit.security_scanner import (
    SecurityScanner,
    SecuritySeverity,
    VulnerabilityCategory,
    SecurityPattern,
    SecurityFinding,
    SecurityReport,
    quick_security_scan,
)


# Sample code with various security issues
CODE_WITH_SECRETS = """
import os

# Hardcoded AWS credentials
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# GitHub token
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# OpenAI API key
OPENAI_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Database connection with password
DATABASE_URL = "postgres://user:password123@localhost:5432/db"

# JWT token
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"

def get_api_key():
    api_key = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx"
    return api_key
"""

CODE_WITH_INJECTION = """
import sqlite3
import os
import subprocess

def vulnerable_sql(user_input):
    # SQL injection via f-string
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")
    return cursor.fetchall()

def vulnerable_sql_format(user_input):
    # SQL injection via format
    query = "SELECT * FROM users WHERE id = {}".format(user_input)
    cursor.execute(query)

def vulnerable_command(filename):
    # Command injection via f-string
    os.system(f"cat {filename}")

def vulnerable_subprocess(user_cmd):
    # Command injection with shell=True
    subprocess.call(user_cmd + " | grep test", shell=True)

def dangerous_eval(user_input):
    # Arbitrary code execution
    result = eval(user_input)
    return result

def dangerous_exec(code):
    # Arbitrary code execution
    exec(code)
"""

CODE_WITH_XSS = """
// JavaScript XSS vulnerabilities

function displayUserContent(content) {
    // XSS via innerHTML
    document.getElementById('output').innerHTML = content + userInput;
}

function writeContent(data) {
    // XSS via document.write
    document.write(data);
}

function ReactComponent({ html }) {
    // React XSS
    return <div dangerouslySetInnerHTML={{ __html: html }} />;
}
"""

CODE_WITH_WEAK_CRYPTO = """
import hashlib
from Crypto.Cipher import DES

def hash_password(password):
    # Weak hash: MD5
    return hashlib.md5(password.encode()).hexdigest()

def hash_sha1(data):
    # Weak hash: SHA1
    return hashlib.sha1(data.encode()).hexdigest()

def encrypt_des(data, key):
    # Weak encryption: DES
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(data)

def weak_iv():
    # Hardcoded IV
    iv = "1234567890123456"
    nonce = "fixed_nonce_value"
"""

CODE_WITH_INSECURE_CONFIG = """
import requests

# Debug mode enabled
DEBUG = True
DEBUG_MODE = true

# SSL verification disabled
response = requests.get(url, verify=False)

# Wildcard CORS
Access-Control-Allow-Origin: *

# HTTP instead of HTTPS
API_URL = "http://api.example.com/data"

# Binding to all interfaces
HOST = "0.0.0.0"
"""

CODE_WITH_DESERIALIZATION = """
import pickle
import yaml
import marshal

def load_pickle(data):
    # Insecure deserialization
    return pickle.loads(data)

def load_yaml(data):
    # Insecure YAML loading
    return yaml.load(data)

def load_marshal(data):
    # Insecure marshal loading
    return marshal.loads(data)
"""

CLEAN_CODE = '''
import os
import hashlib
from typing import Optional

def secure_function(user_id: int) -> Optional[dict]:
    """A secure function with no issues."""
    # Using environment variable for secrets
    api_key = os.environ.get("API_KEY")

    # Parameterized query would go here
    # Using SHA-256 for hashing
    hash_value = hashlib.sha256(str(user_id).encode()).hexdigest()

    return {"user_id": user_id, "hash": hash_value}

class SecureClass:
    """A secure class."""

    def __init__(self):
        self.data = []

    def process(self, item):
        self.data.append(item)
        return len(self.data)
'''


class TestSecuritySeverity:
    """Tests for SecuritySeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert SecuritySeverity.CRITICAL.value == "critical"
        assert SecuritySeverity.HIGH.value == "high"
        assert SecuritySeverity.MEDIUM.value == "medium"
        assert SecuritySeverity.LOW.value == "low"
        assert SecuritySeverity.INFO.value == "info"


class TestVulnerabilityCategory:
    """Tests for VulnerabilityCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert VulnerabilityCategory.HARDCODED_SECRET.value == "hardcoded_secret"
        assert VulnerabilityCategory.SQL_INJECTION.value == "sql_injection"
        assert VulnerabilityCategory.COMMAND_INJECTION.value == "command_injection"
        assert VulnerabilityCategory.XSS.value == "xss"
        assert VulnerabilityCategory.WEAK_CRYPTO.value == "weak_crypto"


class TestSecurityFinding:
    """Tests for SecurityFinding dataclass."""

    def test_create_finding(self):
        """Test creating a security finding."""
        finding = SecurityFinding(
            id="SEC-000001",
            title="Hardcoded API Key",
            description="API key found in source code",
            category=VulnerabilityCategory.HARDCODED_SECRET,
            severity=SecuritySeverity.HIGH,
            confidence=0.95,
            file_path="/test/file.py",
            line_number=10,
        )

        assert finding.id == "SEC-000001"
        assert finding.severity == SecuritySeverity.HIGH
        assert finding.category == VulnerabilityCategory.HARDCODED_SECRET

    def test_finding_to_dict(self):
        """Test serializing finding to dictionary."""
        finding = SecurityFinding(
            id="SEC-000001",
            title="Test Finding",
            description="Test description",
            category=VulnerabilityCategory.HARDCODED_SECRET,
            severity=SecuritySeverity.HIGH,
            confidence=0.9,
            file_path="/test.py",
            line_number=5,
        )

        data = finding.to_dict()
        assert data["id"] == "SEC-000001"
        assert data["severity"] == "high"
        assert data["category"] == "hardcoded_secret"


class TestSecurityReport:
    """Tests for SecurityReport dataclass."""

    def test_create_report(self):
        """Test creating a security report."""
        from datetime import datetime, timezone

        report = SecurityReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
        )

        assert report.scan_id == "scan_001"
        assert report.total_findings == 0

    def test_report_summary(self):
        """Test report summary calculation."""
        from datetime import datetime, timezone

        report = SecurityReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
            findings=[
                SecurityFinding(
                    id="1",
                    title="Critical",
                    description="",
                    category=VulnerabilityCategory.HARDCODED_SECRET,
                    severity=SecuritySeverity.CRITICAL,
                    confidence=0.9,
                    file_path="/test.py",
                    line_number=1,
                ),
                SecurityFinding(
                    id="2",
                    title="High",
                    description="",
                    category=VulnerabilityCategory.SQL_INJECTION,
                    severity=SecuritySeverity.HIGH,
                    confidence=0.9,
                    file_path="/test.py",
                    line_number=2,
                ),
            ],
        )

        report.calculate_summary()
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.has_critical is True

    def test_risk_score(self):
        """Test risk score calculation."""
        from datetime import datetime, timezone

        report = SecurityReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
            findings=[
                SecurityFinding(
                    id="1",
                    title="Critical",
                    description="",
                    category=VulnerabilityCategory.HARDCODED_SECRET,
                    severity=SecuritySeverity.CRITICAL,
                    confidence=1.0,
                    file_path="/test.py",
                    line_number=1,
                ),
            ],
        )

        assert report.risk_score > 0


class TestSecurityScanner:
    """Tests for SecurityScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create a security scanner."""
        return SecurityScanner()

    @pytest.fixture
    def secrets_file(self, tmp_path):
        """Create a file with hardcoded secrets."""
        file_path = tmp_path / "secrets.py"
        file_path.write_text(CODE_WITH_SECRETS)
        return str(file_path)

    @pytest.fixture
    def injection_file(self, tmp_path):
        """Create a file with injection vulnerabilities."""
        file_path = tmp_path / "injection.py"
        file_path.write_text(CODE_WITH_INJECTION)
        return str(file_path)

    @pytest.fixture
    def xss_file(self, tmp_path):
        """Create a file with XSS vulnerabilities."""
        file_path = tmp_path / "xss.js"
        file_path.write_text(CODE_WITH_XSS)
        return str(file_path)

    @pytest.fixture
    def crypto_file(self, tmp_path):
        """Create a file with weak crypto."""
        file_path = tmp_path / "crypto.py"
        file_path.write_text(CODE_WITH_WEAK_CRYPTO)
        return str(file_path)

    @pytest.fixture
    def clean_file(self, tmp_path):
        """Create a clean file with no issues."""
        file_path = tmp_path / "clean.py"
        file_path.write_text(CLEAN_CODE)
        return str(file_path)

    def test_detect_aws_keys(self, scanner, secrets_file):
        """Test detecting AWS access keys."""
        findings = scanner.scan_file(secrets_file)

        aws_findings = [f for f in findings if "AWS" in f.title or "aws" in f.title.lower()]
        assert len(aws_findings) >= 1

    def test_detect_github_token(self, scanner, secrets_file):
        """Test detecting GitHub tokens."""
        findings = scanner.scan_file(secrets_file)

        github_findings = [f for f in findings if "GitHub" in f.title]
        assert len(github_findings) >= 1

    def test_detect_openai_key(self, scanner, secrets_file):
        """Test detecting OpenAI API keys."""
        findings = scanner.scan_file(secrets_file)

        # Should detect sk-xxx pattern
        api_findings = [f for f in findings if f.category == VulnerabilityCategory.HARDCODED_SECRET]
        assert len(api_findings) >= 1

    def test_detect_database_credentials(self, scanner, secrets_file):
        """Test detecting database connection strings."""
        findings = scanner.scan_file(secrets_file)

        db_findings = [f for f in findings if "Database" in f.title or "Connection" in f.title]
        assert len(db_findings) >= 1

    def test_detect_sql_injection(self, scanner, injection_file):
        """Test detecting SQL injection vulnerabilities."""
        findings = scanner.scan_file(injection_file)

        sql_findings = [f for f in findings if f.category == VulnerabilityCategory.SQL_INJECTION]
        assert len(sql_findings) >= 1

    def test_detect_command_injection(self, scanner, injection_file):
        """Test detecting command injection vulnerabilities."""
        findings = scanner.scan_file(injection_file)

        cmd_findings = [
            f for f in findings if f.category == VulnerabilityCategory.COMMAND_INJECTION
        ]
        assert len(cmd_findings) >= 1

    def test_detect_eval_exec(self, scanner, injection_file):
        """Test detecting eval/exec usage."""
        findings = scanner.scan_file(injection_file)

        eval_findings = [
            f for f in findings if "eval" in f.title.lower() or "exec" in f.title.lower()
        ]
        assert len(eval_findings) >= 1

    def test_detect_xss_innerhtml(self, scanner, xss_file):
        """Test detecting innerHTML XSS."""
        findings = scanner.scan_file(xss_file)

        xss_findings = [f for f in findings if f.category == VulnerabilityCategory.XSS]
        assert len(xss_findings) >= 1

    def test_detect_weak_hash_md5(self, scanner, crypto_file):
        """Test detecting MD5 usage."""
        findings = scanner.scan_file(crypto_file)

        md5_findings = [f for f in findings if "MD5" in f.title]
        assert len(md5_findings) >= 1

    def test_detect_weak_encryption_des(self, scanner, crypto_file):
        """Test detecting DES encryption."""
        findings = scanner.scan_file(crypto_file)

        des_findings = [f for f in findings if "DES" in f.title]
        assert len(des_findings) >= 1

    def test_clean_code_minimal_findings(self, scanner, clean_file):
        """Test that clean code has minimal findings."""
        findings = scanner.scan_file(clean_file)

        # Should have very few or no high/critical findings
        critical_high = [
            f for f in findings if f.severity in [SecuritySeverity.CRITICAL, SecuritySeverity.HIGH]
        ]
        assert len(critical_high) == 0

    def test_scan_directory(self, scanner, tmp_path):
        """Test scanning a directory."""
        (tmp_path / "module1.py").write_text(CODE_WITH_SECRETS)
        (tmp_path / "module2.py").write_text(CLEAN_CODE)

        report = scanner.scan_directory(str(tmp_path))

        assert isinstance(report, SecurityReport)
        assert report.files_scanned >= 2
        assert report.total_findings > 0

    def test_scan_with_exclusions(self, scanner, tmp_path):
        """Test scanning with file exclusions."""
        (tmp_path / "main.py").write_text(CODE_WITH_SECRETS)
        (tmp_path / "skip_module.py").write_text(CODE_WITH_SECRETS)

        # Use explicit exclude that only matches skip_ pattern, not pytest temp path
        report = scanner.scan_directory(str(tmp_path), exclude_patterns=["skip_module"])

        # skip_module.py should be excluded
        assert report.files_scanned == 1

    def test_scanner_with_low_severity_disabled(self, tmp_path):
        """Test scanner with low severity findings disabled."""
        scanner = SecurityScanner(include_low_severity=False)

        (tmp_path / "config.py").write_text(CODE_WITH_INSECURE_CONFIG)
        findings = scanner.scan_file(str(tmp_path / "config.py"))

        # Should not include LOW severity findings
        low_findings = [f for f in findings if f.severity == SecuritySeverity.LOW]
        assert len(low_findings) == 0

    def test_finding_has_cwe_id(self, scanner, secrets_file):
        """Test that findings include CWE IDs."""
        findings = scanner.scan_file(secrets_file)

        # At least some findings should have CWE IDs
        findings_with_cwe = [f for f in findings if f.cwe_id]
        assert len(findings_with_cwe) > 0


class TestQuickSecurityScan:
    """Tests for quick_security_scan function."""

    def test_quick_scan_file(self, tmp_path):
        """Test quick scan of a single file."""
        file_path = tmp_path / "test.py"
        file_path.write_text(CODE_WITH_SECRETS)

        result = quick_security_scan(str(file_path))

        assert "findings" in result or "details" in result
        assert result.get("critical", 0) >= 0 or result.get("high", 0) >= 0

    def test_quick_scan_directory(self, tmp_path):
        """Test quick scan of a directory."""
        (tmp_path / "module.py").write_text(CODE_WITH_SECRETS)

        result = quick_security_scan(str(tmp_path))

        assert "total_findings" in result or "findings" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def scanner(self):
        return SecurityScanner()

    def test_empty_file(self, scanner, tmp_path):
        """Test scanning an empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        findings = scanner.scan_file(str(empty_file))
        assert len(findings) == 0

    def test_binary_file(self, scanner, tmp_path):
        """Test handling binary file."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        # Should not crash
        findings = scanner.scan_file(str(binary_file))
        assert isinstance(findings, list)

    def test_nonexistent_file(self, scanner):
        """Test scanning nonexistent file."""
        findings = scanner.scan_file("/nonexistent/file.py")
        assert findings == []

    def test_unicode_content(self, scanner, tmp_path):
        """Test handling Unicode content."""
        unicode_file = tmp_path / "unicode.py"
        # Use a key that's long enough to match patterns (>=20 chars after prefix)
        unicode_file.write_text(
            '# 中文注释\npassword = "密码123"\nAPI_KEY = "sk-test12345678901234567890"',
            encoding="utf-8",
        )

        findings = scanner.scan_file(str(unicode_file))
        # Should find the API key (OpenAI format) or password
        assert any(
            "API" in f.title
            or "Key" in f.title
            or "password" in f.title.lower()
            or "Secret" in f.title
            for f in findings
        )

    def test_very_long_lines(self, scanner, tmp_path):
        """Test handling very long lines."""
        long_line = "x = '" + "a" * 10000 + "'"
        long_file = tmp_path / "long.py"
        long_file.write_text(long_line)

        # Should not crash or hang
        findings = scanner.scan_file(str(long_file))
        assert isinstance(findings, list)
