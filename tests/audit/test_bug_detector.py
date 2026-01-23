"""
Tests for bug detection module.

Tests heuristic-based bug pattern detection.
"""

import pytest
from pathlib import Path

from aragora.audit.bug_detector import (
    BugDetector,
    BugSeverity,
    BugCategory,
    BugPattern,
    PotentialBug,
    BugReport,
    quick_bug_scan,
)


# Sample code with various bug patterns
CODE_WITH_NULL_ISSUES = '''
from typing import Optional

def process_data(data: Optional[dict]):
    """Process data without null check."""
    # Bug: Using data before checking if None
    result = data.get("key")
    if data is None:
        return None
    return result

def optional_param(value: Optional[str] = None):
    """Optional parameter used without check."""
    return value.upper()
'''

CODE_WITH_RESOURCE_LEAKS = '''
import sqlite3
import socket

def leaky_file():
    """File opened without context manager."""
    f = open("data.txt", "r")
    content = f.read()
    # Missing f.close()
    return content

def leaky_database():
    """Database connection without close."""
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()

def leaky_socket():
    """Socket without close."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 8080))
    return s.recv(1024)
'''

CODE_WITH_RACE_CONDITIONS = '''
import os
from pathlib import Path

shared_state = []

def add_to_shared(value):
    """Modifying global state."""
    global shared_state
    shared_state.append(value)

def toctou_file(filepath):
    """Time-of-check to time-of-use vulnerability."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            return f.read()

def mutable_default(items=[]):
    """Mutable default argument bug."""
    items.append(1)
    return items
'''

CODE_WITH_OFF_BY_ONE = '''
def fence_post_error(items):
    """Range fence post error."""
    for i in range(len(items) + 1):
        print(items[i])

def last_element_bug(items):
    """Accessing array at len(array)."""
    return items[len(items)]

def js_style_loop():
    """JavaScript-style inclusive loop error."""
    items = [1, 2, 3]
    # for (let i = 0; i <= items.length; i++)
'''

CODE_WITH_EXCEPTION_ISSUES = '''
def bare_except():
    """Bare except clause."""
    try:
        do_something()
    except:
        pass

def swallowed_exception():
    """Exception caught and ignored."""
    try:
        risky_operation()
    except ValueError:
        pass

def return_in_finally():
    """Return in finally block."""
    try:
        return compute()
    finally:
        return None
'''

CODE_WITH_LOGIC_ERRORS = '''
def self_comparison(x):
    """Variable compared to itself."""
    if x == x:
        return True

def constant_condition():
    """Constant condition."""
    if True:
        return "always"

def duplicate_key():
    """Duplicate dictionary key."""
    data = {
        "name": "Alice",
        "age": 30,
        "name": "Bob",
    }
    return data

def double_negation(value):
    """Double negation."""
    return not not value

def unreachable_code():
    """Unreachable code after return."""
    return 42
    x = 10  # Never executed
'''

CODE_WITH_API_MISUSE = '''
import re

async def missing_await():
    """Async function with no await."""
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    return x + y

def format_without_call():
    """Format string without .format()."""
    template = "Hello {name}"
    return template

def wrong_regex():
    """re.match with $ anchor."""
    pattern = r"test$"
    return re.match(pattern, text)
'''

CODE_WITH_SMELLS = '''
def many_params(a, b, c, d, e, f, g, h):
    """Function with too many parameters."""
    return a + b + c + d + e + f + g + h

def magic_numbers():
    """Magic numbers in code."""
    if value == 42:
        return True
    if count > 100:
        process()
'''

CLEAN_CODE = '''
from typing import Optional

def clean_function(data: Optional[dict]) -> str:
    """A clean function with proper handling."""
    if data is None:
        return ""

    with open("file.txt") as f:
        content = f.read()

    items = data.get("items", [])
    for item in items:
        process(item)

    return content

class CleanClass:
    """A clean class."""

    def __init__(self, value: int):
        self.value = value

    def process(self) -> int:
        try:
            return self.value * 2
        except ValueError as e:
            print(f"Error: {e}")
            raise
'''


class TestBugSeverity:
    """Tests for BugSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert BugSeverity.CRITICAL.value == "critical"
        assert BugSeverity.HIGH.value == "high"
        assert BugSeverity.MEDIUM.value == "medium"
        assert BugSeverity.LOW.value == "low"
        assert BugSeverity.INFO.value == "info"


class TestBugCategory:
    """Tests for BugCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert BugCategory.NULL_REFERENCE.value == "null_reference"
        assert BugCategory.RESOURCE_LEAK.value == "resource_leak"
        assert BugCategory.RACE_CONDITION.value == "race_condition"
        assert BugCategory.OFF_BY_ONE.value == "off_by_one"
        assert BugCategory.EXCEPTION_HANDLING.value == "exception_handling"
        assert BugCategory.LOGIC_ERROR.value == "logic_error"


class TestPotentialBug:
    """Tests for PotentialBug dataclass."""

    def test_create_bug(self):
        """Test creating a potential bug."""
        bug = PotentialBug(
            id="BUG-000001",
            title="Null Reference Risk",
            description="Object used before null check",
            category=BugCategory.NULL_REFERENCE,
            severity=BugSeverity.HIGH,
            confidence=0.85,
            file_path="/test/file.py",
            line_number=10,
        )

        assert bug.id == "BUG-000001"
        assert bug.category == BugCategory.NULL_REFERENCE
        assert bug.severity == BugSeverity.HIGH

    def test_bug_to_dict(self):
        """Test serializing bug to dictionary."""
        bug = PotentialBug(
            id="BUG-000001",
            title="Test Bug",
            description="Test description",
            category=BugCategory.LOGIC_ERROR,
            severity=BugSeverity.MEDIUM,
            confidence=0.8,
            file_path="/test.py",
            line_number=5,
        )

        data = bug.to_dict()
        assert data["id"] == "BUG-000001"
        assert data["severity"] == "medium"
        assert data["category"] == "logic_error"


class TestBugReport:
    """Tests for BugReport dataclass."""

    def test_create_report(self):
        """Test creating a bug report."""
        from datetime import datetime, timezone

        report = BugReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
        )

        assert report.scan_id == "scan_001"
        assert report.total_bugs == 0

    def test_report_summary(self):
        """Test report summary calculation."""
        from datetime import datetime, timezone

        report = BugReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
            bugs=[
                PotentialBug(
                    id="1",
                    title="Critical Bug",
                    description="",
                    category=BugCategory.NULL_REFERENCE,
                    severity=BugSeverity.CRITICAL,
                    confidence=0.9,
                    file_path="/test.py",
                    line_number=1,
                ),
                PotentialBug(
                    id="2",
                    title="High Bug",
                    description="",
                    category=BugCategory.RESOURCE_LEAK,
                    severity=BugSeverity.HIGH,
                    confidence=0.9,
                    file_path="/test.py",
                    line_number=2,
                ),
            ],
        )

        report.calculate_summary()
        assert report.critical_count == 1
        assert report.high_count == 1

    def test_bugs_by_category(self):
        """Test grouping bugs by category."""
        from datetime import datetime, timezone

        report = BugReport(
            scan_id="scan_001",
            repository="/test/repo",
            started_at=datetime.now(timezone.utc),
            bugs=[
                PotentialBug(
                    id="1",
                    title="Bug 1",
                    description="",
                    category=BugCategory.NULL_REFERENCE,
                    severity=BugSeverity.HIGH,
                    confidence=0.9,
                    file_path="/test.py",
                    line_number=1,
                ),
                PotentialBug(
                    id="2",
                    title="Bug 2",
                    description="",
                    category=BugCategory.NULL_REFERENCE,
                    severity=BugSeverity.MEDIUM,
                    confidence=0.8,
                    file_path="/test.py",
                    line_number=2,
                ),
            ],
        )

        by_category = report.bugs_by_category
        assert by_category.get("null_reference", 0) == 2


class TestBugDetector:
    """Tests for BugDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a bug detector."""
        return BugDetector()

    @pytest.fixture
    def null_file(self, tmp_path):
        """Create a file with null reference issues."""
        file_path = tmp_path / "null.py"
        file_path.write_text(CODE_WITH_NULL_ISSUES)
        return str(file_path)

    @pytest.fixture
    def leak_file(self, tmp_path):
        """Create a file with resource leaks."""
        file_path = tmp_path / "leaks.py"
        file_path.write_text(CODE_WITH_RESOURCE_LEAKS)
        return str(file_path)

    @pytest.fixture
    def race_file(self, tmp_path):
        """Create a file with race conditions."""
        file_path = tmp_path / "race.py"
        file_path.write_text(CODE_WITH_RACE_CONDITIONS)
        return str(file_path)

    @pytest.fixture
    def exception_file(self, tmp_path):
        """Create a file with exception handling issues."""
        file_path = tmp_path / "exceptions.py"
        file_path.write_text(CODE_WITH_EXCEPTION_ISSUES)
        return str(file_path)

    @pytest.fixture
    def logic_file(self, tmp_path):
        """Create a file with logic errors."""
        file_path = tmp_path / "logic.py"
        file_path.write_text(CODE_WITH_LOGIC_ERRORS)
        return str(file_path)

    @pytest.fixture
    def clean_file(self, tmp_path):
        """Create a clean file."""
        file_path = tmp_path / "clean.py"
        file_path.write_text(CLEAN_CODE)
        return str(file_path)

    def test_detect_file_without_context_manager(self, detector, leak_file):
        """Test detecting files opened without context manager."""
        bugs = detector.detect_in_file(leak_file)

        leak_bugs = [b for b in bugs if b.category == BugCategory.RESOURCE_LEAK]
        assert len(leak_bugs) >= 1

    def test_detect_mutable_default(self, detector, race_file):
        """Test detecting mutable default arguments."""
        bugs = detector.detect_in_file(race_file)

        race_bugs = [b for b in bugs if b.category == BugCategory.RACE_CONDITION]
        # Should find mutable default
        mutable_bugs = [
            b for b in race_bugs if "Mutable" in b.title or "default" in b.title.lower()
        ]
        assert len(mutable_bugs) >= 1

    def test_detect_bare_except(self, detector, exception_file):
        """Test detecting bare except clause."""
        bugs = detector.detect_in_file(exception_file)

        except_bugs = [b for b in bugs if b.category == BugCategory.EXCEPTION_HANDLING]
        bare_except = [b for b in except_bugs if "Bare" in b.title or "bare" in b.title.lower()]
        assert len(bare_except) >= 1

    def test_detect_swallowed_exception(self, detector, exception_file):
        """Test detecting swallowed exceptions."""
        bugs = detector.detect_in_file(exception_file)

        swallowed = [b for b in bugs if "Swallowed" in b.title or "pass" in b.description.lower()]
        assert len(swallowed) >= 1

    def test_detect_self_comparison(self, detector, logic_file):
        """Test detecting self-comparison."""
        bugs = detector.detect_in_file(logic_file)

        logic_bugs = [b for b in bugs if b.category == BugCategory.LOGIC_ERROR]
        self_cmp = [b for b in logic_bugs if "self" in b.title.lower() or "Self" in b.title]
        assert len(self_cmp) >= 1

    def test_detect_duplicate_key(self, detector, logic_file):
        """Test detecting duplicate dictionary keys."""
        bugs = detector.detect_in_file(logic_file)

        dup_bugs = [b for b in bugs if "Duplicate" in b.title or "duplicate" in b.title.lower()]
        assert len(dup_bugs) >= 1

    def test_clean_code_minimal_bugs(self, detector, clean_file):
        """Test that clean code has minimal bugs."""
        bugs = detector.detect_in_file(clean_file)

        # Should have few or no high-severity bugs
        high_bugs = [b for b in bugs if b.severity in [BugSeverity.CRITICAL, BugSeverity.HIGH]]
        assert len(high_bugs) <= 1

    def test_detect_in_directory(self, detector, tmp_path):
        """Test detecting bugs in directory."""
        (tmp_path / "module1.py").write_text(CODE_WITH_EXCEPTION_ISSUES)
        (tmp_path / "module2.py").write_text(CLEAN_CODE)

        report = detector.detect_in_directory(str(tmp_path))

        assert isinstance(report, BugReport)
        assert report.files_scanned >= 2
        assert report.total_bugs > 0

    def test_detect_with_exclusions(self, detector, tmp_path):
        """Test detecting with exclusion patterns."""
        (tmp_path / "main.py").write_text(CODE_WITH_EXCEPTION_ISSUES)
        (tmp_path / "test_module.py").write_text(CODE_WITH_EXCEPTION_ISSUES)

        report = detector.detect_in_directory(str(tmp_path), exclude_patterns=["test"])

        # test_module.py should be excluded
        assert report.files_scanned == 1

    def test_detector_without_smells(self, tmp_path):
        """Test detector with code smells disabled."""
        detector = BugDetector(include_smells=False)

        (tmp_path / "code.py").write_text(CODE_WITH_SMELLS)
        bugs = detector.detect_in_file(str(tmp_path / "code.py"))

        # Should not include code smell patterns
        smell_bugs = [b for b in bugs if b.category == BugCategory.CODE_SMELL]
        assert len(smell_bugs) == 0


class TestQuickBugScan:
    """Tests for quick_bug_scan function."""

    def test_quick_scan_file(self, tmp_path):
        """Test quick scan of a single file."""
        file_path = tmp_path / "test.py"
        file_path.write_text(CODE_WITH_EXCEPTION_ISSUES)

        result = quick_bug_scan(str(file_path))

        assert "bugs_found" in result or "total_bugs" in result

    def test_quick_scan_directory(self, tmp_path):
        """Test quick scan of a directory."""
        (tmp_path / "module.py").write_text(CODE_WITH_LOGIC_ERRORS)

        result = quick_bug_scan(str(tmp_path))

        assert "total_bugs" in result or "bugs_found" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def detector(self):
        return BugDetector()

    def test_empty_file(self, detector, tmp_path):
        """Test scanning an empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        bugs = detector.detect_in_file(str(empty_file))
        assert len(bugs) == 0

    def test_nonexistent_file(self, detector):
        """Test scanning nonexistent file."""
        bugs = detector.detect_in_file("/nonexistent/file.py")
        assert bugs == []

    def test_binary_file(self, detector, tmp_path):
        """Test handling binary file."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        # Should not crash
        bugs = detector.detect_in_file(str(binary_file))
        assert isinstance(bugs, list)

    def test_unicode_content(self, detector, tmp_path):
        """Test handling Unicode content."""
        unicode_file = tmp_path / "unicode.py"
        unicode_file.write_text(
            "# 中文注释\ndef 函数():\n    pass",
            encoding="utf-8",
        )

        bugs = detector.detect_in_file(str(unicode_file))
        assert isinstance(bugs, list)

    def test_very_long_file(self, detector, tmp_path):
        """Test handling very long file."""
        # Generate a file with many functions
        lines = []
        for i in range(200):
            lines.append(f"def func_{i}():\n    pass\n")
        long_file = tmp_path / "long.py"
        long_file.write_text("\n".join(lines))

        # Should not hang
        bugs = detector.detect_in_file(str(long_file))
        assert isinstance(bugs, list)

    def test_malformed_python(self, detector, tmp_path):
        """Test handling malformed Python."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n  # incomplete")

        # Should not crash
        bugs = detector.detect_in_file(str(bad_file))
        assert isinstance(bugs, list)
