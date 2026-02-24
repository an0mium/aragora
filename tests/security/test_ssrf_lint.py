"""
Tests for the SSRF guard CI lint script.
"""

import importlib.util
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Import the script module by file path (it's not a package)
_script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "lint_ssrf_guard.py"
_spec = importlib.util.spec_from_file_location("lint_ssrf_guard", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_find_http_calls = _mod._find_http_calls
_has_ssrf_guard = _mod._has_ssrf_guard
_has_allowlist_comment = _mod._has_allowlist_comment
_should_skip = _mod._should_skip
scan_directory = _mod.scan_directory


class TestFindHttpCalls:
    """Tests for detecting HTTP call patterns."""

    def test_requests_get(self):
        calls = _find_http_calls("resp = requests.get(url)")
        assert len(calls) == 1
        assert calls[0][0] == 1  # line 1

    def test_requests_post(self):
        calls = _find_http_calls("requests.post(url, json=data)")
        assert len(calls) == 1

    def test_httpx_get(self):
        calls = _find_http_calls("r = httpx.get(url)")
        assert len(calls) == 1

    def test_httpx_async_client(self):
        calls = _find_http_calls("async with httpx.AsyncClient() as client:")
        assert len(calls) == 1

    def test_aiohttp_session(self):
        calls = _find_http_calls("session = aiohttp.ClientSession()")
        assert len(calls) == 1

    def test_urllib_urlopen(self):
        calls = _find_http_calls("urllib.request.urlopen(url)")
        assert len(calls) == 1

    def test_no_http_calls(self):
        calls = _find_http_calls("print('hello')\nx = 1 + 2")
        assert len(calls) == 0

    def test_comments_skipped(self):
        calls = _find_http_calls("# requests.get(url)")
        assert len(calls) == 0

    def test_multiple_calls(self):
        code = "requests.get(url1)\nrequests.post(url2)"
        calls = _find_http_calls(code)
        assert len(calls) == 2
        assert calls[0][0] == 1
        assert calls[1][0] == 2

    def test_httpx_request(self):
        calls = _find_http_calls("httpx.request('GET', url)")
        assert len(calls) == 1


class TestHasSsrfGuard:
    """Tests for SSRF guard detection."""

    def test_import_detected(self):
        code = "from aragora.security.ssrf_protection import validate_url"
        assert _has_ssrf_guard(code) is True

    def test_module_import_detected(self):
        code = "import aragora.security.ssrf_protection"
        assert _has_ssrf_guard(code) is True

    def test_validate_url_usage(self):
        code = "result = validate_url(url)"
        assert _has_ssrf_guard(code) is True

    def test_is_url_safe_usage(self):
        code = "if is_url_safe(url):"
        assert _has_ssrf_guard(code) is True

    def test_ssrf_protection_reference(self):
        code = "from foo import ssrf_protection"
        assert _has_ssrf_guard(code) is True

    def test_no_guard(self):
        code = "import json\nrequests.get(url)"
        assert _has_ssrf_guard(code) is False


class TestHasAllowlistComment:
    """Tests for allowlist comment detection."""

    def test_allowlist_present(self):
        code = "# ssrf-safe: internal API only\nrequests.get(url)"
        assert _has_allowlist_comment(code) is True

    def test_no_allowlist(self):
        code = "requests.get(url)"
        assert _has_allowlist_comment(code) is False


class TestShouldSkip:
    """Tests for file skip logic."""

    def test_skip_tests(self):
        base = Path("/project")
        assert _should_skip(Path("/project/tests/test_foo.py"), base) is True

    def test_skip_scripts(self):
        base = Path("/project")
        assert _should_skip(Path("/project/scripts/lint.py"), base) is True

    def test_skip_ssrf_module(self):
        base = Path("/project")
        assert _should_skip(
            Path("/project/aragora/security/ssrf_protection.py"), base
        ) is True

    def test_allow_normal_file(self):
        base = Path("/project")
        assert _should_skip(
            Path("/project/aragora/connectors/slack.py"), base
        ) is False


class TestScanDirectory:
    """Tests for full directory scanning."""

    def test_finds_unguarded_calls(self, tmp_path):
        """Should detect files with HTTP calls but no SSRF guard."""
        module = tmp_path / "bad_module.py"
        module.write_text("import httpx\nr = httpx.get(url)\n")

        violations = scan_directory(tmp_path)
        assert len(violations) == 1
        assert violations[0]["file"] == "bad_module.py"
        assert violations[0]["line"] == 2

    def test_skips_guarded_files(self, tmp_path):
        """Should not flag files that import SSRF protection."""
        module = tmp_path / "good_module.py"
        module.write_text(
            "from aragora.security.ssrf_protection import validate_url\n"
            "import httpx\n"
            "validate_url(url)\n"
            "httpx.get(url)\n"
        )

        violations = scan_directory(tmp_path)
        assert len(violations) == 0

    def test_skips_allowlisted_files(self, tmp_path):
        """Should not flag files with allowlist comment."""
        module = tmp_path / "allowed.py"
        module.write_text(
            "# ssrf-safe: internal health check endpoint only\n"
            "import requests\n"
            "requests.get('http://localhost/health')\n"
        )

        violations = scan_directory(tmp_path)
        assert len(violations) == 0

    def test_no_false_positives_on_clean_files(self, tmp_path):
        """Should not flag files without HTTP calls."""
        module = tmp_path / "clean.py"
        module.write_text("import json\nx = json.loads('{}')\n")

        violations = scan_directory(tmp_path)
        assert len(violations) == 0

    def test_multiple_violations_in_file(self, tmp_path):
        """Should report each unguarded call."""
        module = tmp_path / "multi.py"
        module.write_text(
            "import requests\nimport httpx\n"
            "requests.get(url)\n"
            "httpx.post(url)\n"
        )

        violations = scan_directory(tmp_path)
        assert len(violations) == 2

    def test_skips_test_directories(self, tmp_path):
        """Should skip tests/ directories."""
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        module = test_dir / "test_foo.py"
        module.write_text("import requests\nrequests.get(url)\n")

        violations = scan_directory(tmp_path)
        assert len(violations) == 0
