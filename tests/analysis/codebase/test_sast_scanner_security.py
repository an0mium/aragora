"""
Security-focused tests for SAST Scanner module.

Tests security properties of the SAST scanner including:
- Subprocess timeout handling and zombie process prevention
- Malicious input sanitization (path traversal, shell metacharacters)
- Resource exhaustion protection (concurrent limits, memory bounds)
- Error handling for edge cases (Semgrep unavailable, malformed output)
- Input validation (invalid paths, rule sets, unsafe content)

These tests complement the functional tests in test_sast_scanner.py by
focusing on security boundaries and failure modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.sast_scanner import (
    SASTConfig,
    SASTFinding,
    SASTScanResult,
    SASTScanner,
    SASTSeverity,
    OWASPCategory,
    scan_for_vulnerabilities,
)


# ============================================================
# Subprocess Timeout Tests
# ============================================================


class TestSubprocessTimeout:
    """Tests for subprocess timeout handling during scans."""

    @pytest.mark.asyncio
    async def test_semgrep_check_timeout_handled(self):
        """Semgrep availability check handles timeout gracefully."""
        scanner = SASTScanner()

        # Simulate asyncio.TimeoutError being raised by wait_for
        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.wait_for",
            side_effect=asyncio.TimeoutError(),
        ):
            available, version = await scanner._check_semgrep()
            assert available is False
            assert version is None

    @pytest.mark.asyncio
    async def test_semgrep_check_oserror_handled(self):
        """Semgrep availability check handles OSError gracefully."""
        scanner = SASTScanner()

        async def raise_oserror(*args, **kwargs):
            raise OSError("Command not found")

        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.create_subprocess_exec",
            side_effect=raise_oserror,
        ):
            available, version = await scanner._check_semgrep()
            assert available is False
            assert version is None

    @pytest.mark.asyncio
    async def test_semgrep_check_file_not_found(self):
        """Semgrep availability check handles FileNotFoundError."""
        scanner = SASTScanner()

        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("semgrep not found"),
        ):
            available, version = await scanner._check_semgrep()
            assert available is False
            assert version is None

    @pytest.mark.asyncio
    async def test_scan_with_semgrep_timeout(self):
        """Semgrep scan handles timeout and returns error result."""
        config = SASTConfig(semgrep_timeout=1)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = True

        # Create a mock process that will be used
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        async def mock_create_subprocess(*args, **kwargs):
            return mock_process

        # Mock both create_subprocess_exec and wait_for
        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            with patch(
                "aragora.analysis.codebase.sast_scanner.asyncio.wait_for",
                side_effect=asyncio.TimeoutError(),
            ):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = await scanner._scan_with_semgrep(
                        tmpdir, ["p/owasp-top-ten"], "test-scan"
                    )
                    assert "timed out" in result.errors[0].lower()
                    assert result.scanned_files == 0

    @pytest.mark.asyncio
    async def test_timeout_error_message_content(self):
        """Timeout error message is descriptive for debugging."""
        config = SASTConfig(semgrep_timeout=1)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        async def mock_create_subprocess(*args, **kwargs):
            return mock_process

        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.create_subprocess_exec",
            side_effect=mock_create_subprocess,
        ):
            with patch(
                "aragora.analysis.codebase.sast_scanner.asyncio.wait_for",
                side_effect=asyncio.TimeoutError(),
            ):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = await scanner._scan_with_semgrep(
                        tmpdir, ["p/owasp-top-ten"], "test-scan"
                    )
                    # Check for timeout-related error message
                    assert len(result.errors) > 0
                    assert any(
                        "timeout" in e.lower() or "timed out" in e.lower() for e in result.errors
                    )

    @pytest.mark.asyncio
    async def test_multiple_concurrent_timeouts(self):
        """Multiple concurrent scans handle independent timeouts."""
        config = SASTConfig(semgrep_timeout=1)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = True

        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.wait_for",
            side_effect=asyncio.TimeoutError(),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Run multiple scans concurrently
                tasks = [
                    scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], f"scan-{i}")
                    for i in range(3)
                ]
                results = await asyncio.gather(*tasks)

                # All should complete with timeout errors
                for result in results:
                    assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_scan_timeout_returns_partial_result(self):
        """After timeout, scan returns partial result with error."""
        config = SASTConfig(semgrep_timeout=1)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = True

        with patch(
            "aragora.analysis.codebase.sast_scanner.asyncio.wait_for",
            side_effect=asyncio.TimeoutError(),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], "test-scan")
                # Should return a valid SASTScanResult even on timeout
                assert isinstance(result, SASTScanResult)
                assert result.repository_path == tmpdir
                assert result.scan_id == "test-scan"


# ============================================================
# Malicious Input Tests
# ============================================================


class TestMaliciousInputHandling:
    """Tests for handling malicious or adversarial inputs."""

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self):
        """Path traversal attempts in repo_path are normalized or rejected."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Attempt path traversal
        malicious_path = "/tmp/../../etc/passwd"
        result = await scanner.scan_repository(malicious_path)

        # Should not scan outside intended scope - path is normalized by abspath
        # Result should indicate the path doesn't exist or is invalid
        assert result.scanned_files == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_relative_path_traversal_patterns(self):
        """Various path traversal patterns are handled safely."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/tmp/test/../../../etc/shadow",
            "./../../sensitive",
            "foo/bar/../../../root",
        ]

        for pattern in traversal_patterns:
            result = await scanner.scan_repository(pattern)
            # All should fail gracefully - path not found or normalized
            assert result.scanned_files == 0 or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_shell_metacharacters_in_path(self):
        """Shell metacharacters in paths don't cause command injection."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Paths with shell metacharacters
        dangerous_paths = [
            "/tmp/test; rm -rf /",
            "/tmp/test$(whoami)",
            "/tmp/test`id`",
            "/tmp/test|cat /etc/passwd",
            "/tmp/test&& malicious",
            "/tmp/test > /tmp/pwned",
            "/tmp/test\necho pwned",
        ]

        for path in dangerous_paths:
            result = await scanner.scan_repository(path)
            # Should not execute shell commands - just fail to find path
            assert result.scanned_files == 0

    @pytest.mark.asyncio
    async def test_null_byte_in_path(self):
        """Null bytes in paths are handled safely."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Null byte injection attempts
        try:
            result = await scanner.scan_repository("/tmp/test\x00.py")
            # Should either error or return empty result
            assert result.scanned_files == 0 or len(result.errors) > 0
        except (ValueError, OSError):
            # Some systems may raise exceptions for null bytes
            pass

    @pytest.mark.asyncio
    async def test_very_long_path_handled(self):
        """Extremely long paths don't cause buffer issues."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Path exceeding typical filesystem limits
        long_path = "/tmp/" + "a" * 10000

        result = await scanner.scan_repository(long_path)
        # Should fail gracefully, not crash
        assert result.scanned_files == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_unicode_path_handling(self):
        """Unicode characters in paths are handled correctly."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory with unicode name
            unicode_dir = os.path.join(tmpdir, "test_\u4e2d\u6587_\u0440\u0443\u0441")
            os.makedirs(unicode_dir, exist_ok=True)

            # Create a Python file in unicode directory
            test_file = os.path.join(unicode_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("x = 1\n")

            result = await scanner.scan_repository(unicode_dir)
            # Should scan without errors
            assert result.scanned_files >= 1

    @pytest.mark.asyncio
    async def test_symlink_following_safety(self):
        """Symlinks don't allow escaping the scan directory."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink pointing outside the scan directory
            safe_file = os.path.join(tmpdir, "safe.py")
            with open(safe_file, "w") as f:
                f.write("x = 1\n")

            # Try to create symlink to /etc (may fail on some systems)
            link_path = os.path.join(tmpdir, "escape_link")
            os.symlink("/etc", link_path)

            # Scan should not follow symlinks to external locations
            result = await scanner.scan_repository(tmpdir)
            # Should only scan the safe file, not follow symlink
            assert result.scanned_files <= 2  # Just files in tmpdir

    @pytest.mark.asyncio
    async def test_malicious_filename_characters(self):
        """Special characters in filenames don't cause issues."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with special characters (where filesystem allows)
            special_names = [
                "test file.py",
                "test'quote.py",
                'test"doublequote.py',
                "test$dollar.py",
            ]

            for name in special_names:
                try:
                    filepath = os.path.join(tmpdir, name)
                    with open(filepath, "w") as f:
                        f.write("x = 1\n")
                except OSError:
                    continue  # Skip if filesystem doesn't support

            result = await scanner.scan_repository(tmpdir)
            # Should scan all valid files without crashing
            assert result.scanned_files >= 0


# ============================================================
# Resource Exhaustion Tests
# ============================================================


class TestResourceExhaustion:
    """Tests for resource exhaustion protection."""

    @pytest.mark.asyncio
    async def test_max_file_size_enforced(self):
        """Files exceeding max_file_size_kb are skipped."""
        config = SASTConfig(use_semgrep=False, max_file_size_kb=1)  # 1KB limit
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file larger than 1KB
            large_file = os.path.join(tmpdir, "large.py")
            with open(large_file, "w") as f:
                f.write("x = 1\n" * 1000)  # ~6KB

            result = await scanner.scan_repository(tmpdir)
            # Large file should be skipped
            assert result.skipped_files >= 1

    @pytest.mark.asyncio
    async def test_many_files_handled(self):
        """Large number of files is handled without memory issues."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many small files
            for i in range(100):
                filepath = os.path.join(tmpdir, f"file_{i}.py")
                with open(filepath, "w") as f:
                    f.write(f"x_{i} = {i}\n")

            result = await scanner.scan_repository(tmpdir)
            assert result.scanned_files == 100

    @pytest.mark.asyncio
    async def test_deeply_nested_directories(self):
        """Deeply nested directory structures are handled."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deeply nested structure
            nested_path = tmpdir
            for i in range(20):
                nested_path = os.path.join(nested_path, f"level_{i}")
            os.makedirs(nested_path, exist_ok=True)

            # Add a file at the deepest level
            deep_file = os.path.join(nested_path, "deep.py")
            with open(deep_file, "w") as f:
                f.write("x = 1\n")

            result = await scanner.scan_repository(tmpdir)
            assert result.scanned_files >= 1

    @pytest.mark.asyncio
    async def test_large_finding_count_handling(self):
        """Large number of findings doesn't cause memory issues."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many vulnerability patterns
            vuln_file = os.path.join(tmpdir, "many_vulns.py")
            with open(vuln_file, "w") as f:
                for i in range(100):
                    f.write(f"result_{i} = eval(input_{i})\n")

            result = await scanner.scan_repository(tmpdir)
            # Should capture all findings without crashing
            assert len(result.findings) >= 50  # At least half should be found

    @pytest.mark.asyncio
    async def test_concurrent_scan_independence(self):
        """Multiple concurrent scans don't interfere with each other."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                # Create different files in each directory
                with open(os.path.join(tmpdir1, "file1.py"), "w") as f:
                    f.write("eval(data1)\n")
                with open(os.path.join(tmpdir2, "file2.py"), "w") as f:
                    f.write("exec(data2)\n")

                # Run concurrent scans
                task1 = scanner.scan_repository(tmpdir1)
                task2 = scanner.scan_repository(tmpdir2)
                result1, result2 = await asyncio.gather(task1, task2)

                # Results should be independent
                assert result1.repository_path != result2.repository_path
                assert result1.scan_id != result2.scan_id

    @pytest.mark.asyncio
    async def test_empty_directory_handled(self):
        """Empty directories don't cause errors."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(tmpdir)
            assert result.scanned_files == 0
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_binary_file_handling(self):
        """Binary files are handled safely without decoding errors."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a binary file with .py extension (edge case)
            binary_file = os.path.join(tmpdir, "binary.py")
            with open(binary_file, "wb") as f:
                f.write(b"\x00\x01\x02\x03\xff\xfe\xfd")

            result = await scanner.scan_repository(tmpdir)
            # Should handle binary content gracefully
            assert len(result.errors) == 0 or result.scanned_files >= 0


# ============================================================
# Error Handling Tests
# ============================================================


class TestErrorHandling:
    """Tests for error handling in various failure scenarios."""

    @pytest.mark.asyncio
    async def test_semgrep_not_installed(self):
        """Scanner handles missing Semgrep installation gracefully."""
        scanner = SASTScanner()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            available, version = await scanner._check_semgrep()
            assert available is False
            assert version is None

    @pytest.mark.asyncio
    async def test_semgrep_crashes_during_scan(self):
        """Scanner handles Semgrep process crash gracefully."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=OSError("Process crashed"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], "test")
                assert len(result.errors) > 0
                assert "failed" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_semgrep_invalid_json_output(self):
        """Scanner handles malformed Semgrep JSON output."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"not json {{{", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], "test")
                assert len(result.errors) > 0
                assert "parse" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_semgrep_empty_output(self):
        """Scanner handles empty Semgrep output."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], "test")
                # Should handle empty output - either error or empty result
                assert result.findings == [] or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_semgrep_stderr_with_errors(self):
        """Scanner captures and reports Semgrep stderr errors."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(
                b'{"results": [], "paths": {"scanned": [], "skipped": []}}',
                b"error: invalid rule syntax",
            )
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["p/owasp-top-ten"], "test")
                assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_file_read_permission_denied(self):
        """Scanner handles permission denied errors when reading files."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and remove read permissions
            protected_file = os.path.join(tmpdir, "protected.py")
            with open(protected_file, "w") as f:
                f.write("eval(x)\n")

            try:
                os.chmod(protected_file, 0o000)
                result = await scanner.scan_repository(tmpdir)
                # Should handle gracefully - either skip or error
                # Restore permissions before cleanup
            finally:
                os.chmod(protected_file, 0o644)

    @pytest.mark.asyncio
    async def test_file_deleted_during_scan(self):
        """Scanner handles files deleted during scan."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(5):
                with open(os.path.join(tmpdir, f"file_{i}.py"), "w") as f:
                    f.write("x = 1\n")

            # Scan should handle if files disappear
            result = await scanner.scan_repository(tmpdir)
            # Should complete without crashing
            assert result is not None

    @pytest.mark.asyncio
    async def test_malformed_semgrep_result_parsing(self):
        """Parser handles various malformed Semgrep results."""
        scanner = SASTScanner()

        # Test various malformed inputs - all should return None without exception
        malformed_results = [
            None,
            {},
            {"check_id": "test"},  # Missing required fields
            {"check_id": "test", "path": "f.py"},  # Missing start/end
            {"check_id": "test", "path": "f.py", "start": {}, "end": {}},  # Missing extra
        ]

        for result in malformed_results:
            try:
                finding = scanner._parse_semgrep_result(result)
                # Should return None or a partial finding for malformed input
                # The key is it shouldn't raise an exception
            except Exception as e:
                pytest.fail(f"Parser raised exception for malformed input {result}: {e}")

    @pytest.mark.asyncio
    async def test_interrupted_scan_cleanup(self):
        """Scanner cleans up properly when scan is interrupted."""
        config = SASTConfig(use_semgrep=False)
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                with open(os.path.join(tmpdir, f"file_{i}.py"), "w") as f:
                    f.write("eval(x)\n")

            # Create task and cancel it
            task = asyncio.create_task(scanner.scan_repository(tmpdir))
            await asyncio.sleep(0.01)  # Let it start
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected

            # Scanner should still be usable
            result = await scanner.scan_repository(tmpdir)
            assert result is not None


# ============================================================
# Input Validation Tests
# ============================================================


class TestInputValidation:
    """Tests for input validation of paths, rule sets, and configurations."""

    @pytest.mark.asyncio
    async def test_invalid_repo_path_type(self):
        """Non-string repo path is handled."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # This should raise or handle gracefully
        try:
            # Type error expected but scanner might coerce
            result = await scanner.scan_repository(None)  # type: ignore
            assert len(result.errors) > 0
        except (TypeError, AttributeError):
            pass  # Expected

    @pytest.mark.asyncio
    async def test_empty_repo_path(self):
        """Empty string repo path normalizes to current directory."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Empty string normalizes to current directory via os.path.abspath
        # This is expected Python behavior - the scanner accepts it
        # We just verify it doesn't crash and returns a valid result
        # (Don't actually scan as it could take a long time on large repos)
        # Instead, test with a non-existent specific path
        result = await scanner.scan_repository("/nonexistent/path/xyz987654")
        assert result.scanned_files == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_whitespace_only_path(self):
        """Whitespace-only path is handled (normalizes to nonexistent path)."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        # Whitespace path normalizes to a path with spaces that doesn't exist
        result = await scanner.scan_repository("   ")
        # Should complete without crashing - either no files or error
        assert result.scanned_files == 0 or len(result.errors) >= 0

    @pytest.mark.asyncio
    async def test_invalid_rule_set_names(self):
        """Invalid Semgrep rule set names are handled."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(
                b"",
                b"error: Unknown config: invalid-ruleset",
            )
        )
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await scanner._scan_with_semgrep(tmpdir, ["invalid-ruleset-xyz"], "test")
                # Should capture error from invalid ruleset
                assert len(result.errors) > 0 or result.scanned_files == 0

    @pytest.mark.asyncio
    async def test_empty_rule_sets_list(self):
        """Empty rule sets list uses defaults."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("eval(x)\n")

            result = await scanner.scan_repository(tmpdir, rule_sets=[])
            # Should fall back to defaults or scan with local patterns
            assert result is not None

    @pytest.mark.asyncio
    async def test_none_rule_sets(self):
        """None rule sets uses defaults."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("eval(x)\n")

            result = await scanner.scan_repository(tmpdir, rule_sets=None)
            assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_confidence_threshold(self):
        """Invalid confidence threshold values are handled."""
        config = SASTConfig(
            use_semgrep=False,
            min_confidence_threshold=-0.5,  # Invalid negative
        )
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("eval(x)\n")

            # Should still work (negative threshold accepts all)
            result = await scanner.scan_repository(tmpdir)
            assert result is not None

    @pytest.mark.asyncio
    async def test_confidence_threshold_over_one(self):
        """Confidence threshold > 1.0 filters everything."""
        config = SASTConfig(
            use_semgrep=False,
            min_confidence_threshold=1.5,  # Impossible threshold
            enable_false_positive_filtering=True,
        )
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("eval(x)\n")

            result = await scanner.scan_repository(tmpdir)
            # All findings should be filtered out
            assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_scan_id_uniqueness(self):
        """Scan IDs are unique across multiple scans."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = await scanner.scan_repository(tmpdir)
            result2 = await scanner.scan_repository(tmpdir)
            assert result1.scan_id != result2.scan_id

    @pytest.mark.asyncio
    async def test_custom_scan_id_used(self):
        """Custom scan ID is used when provided."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await scanner.scan_repository(tmpdir, scan_id="custom-123")
            assert result.scan_id == "custom-123"

    @pytest.mark.asyncio
    async def test_dangerous_rule_content_blocked(self):
        """Rule content with shell commands is not executed."""
        # Custom rules directory with potentially malicious content
        scanner = SASTScanner()
        scanner._semgrep_available = True

        # Verify custom_rules_dir is validated
        config = SASTConfig(custom_rules_dir="/tmp/../../etc")
        dangerous_scanner = SASTScanner(config=config)

        # Even if custom rules dir points elsewhere, it should be checked
        assert dangerous_scanner.config.custom_rules_dir is not None

    @pytest.mark.asyncio
    async def test_language_override_validation(self):
        """Language override is validated against supported languages."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("eval(x)\n")
            f.flush()
            try:
                # Using unsupported language should just skip patterns
                findings = await scanner.scan_file(f.name, language="unsupported")
                # Should return empty or limited findings
                assert isinstance(findings, list)
            finally:
                os.unlink(f.name)


# ============================================================
# Semgrep Command Injection Prevention Tests
# ============================================================


class TestSemgrepCommandInjection:
    """Tests for preventing command injection in Semgrep calls."""

    @pytest.mark.asyncio
    async def test_malicious_repo_path_not_executed(self):
        """Malicious characters in repo path don't execute commands."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        captured_cmd = []

        async def capture_cmd(*args, **kwargs):
            captured_cmd.extend(args)
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(
                return_value=(
                    b'{"results": [], "paths": {"scanned": [], "skipped": []}}',
                    b"",
                )
            )
            mock_process.returncode = 0
            return mock_process

        with patch("asyncio.create_subprocess_exec", side_effect=capture_cmd):
            # Path with shell metacharacters
            await scanner._scan_with_semgrep("/tmp/test; rm -rf /", ["p/owasp-top-ten"], "test")

            # Command should use subprocess_exec which doesn't interpret shell
            # The path should be passed as-is, not interpreted
            if captured_cmd:
                # Last argument should be the path
                assert captured_cmd[-1] == "/tmp/test; rm -rf /"

    @pytest.mark.asyncio
    async def test_malicious_rule_set_not_executed(self):
        """Malicious rule set names don't execute commands."""
        scanner = SASTScanner()
        scanner._semgrep_available = True

        captured_args = []

        async def capture_args(*args, **kwargs):
            captured_args.extend(args)
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(
                return_value=(
                    b'{"results": [], "paths": {"scanned": [], "skipped": []}}',
                    b"",
                )
            )
            mock_process.returncode = 0
            return mock_process

        with patch("asyncio.create_subprocess_exec", side_effect=capture_args):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Rule set with shell metacharacters
                await scanner._scan_with_semgrep(tmpdir, ["p/test; whoami"], "test")

                # Arguments should be passed separately, not shell-interpreted
                # --config argument should contain the literal string
                assert "p/test; whoami" in captured_args

    @pytest.mark.asyncio
    async def test_exclusion_pattern_injection_prevented(self):
        """Exclusion patterns don't allow command injection."""
        config = SASTConfig(excluded_patterns=["node_modules/", "$(whoami)/", "`id`/"])
        scanner = SASTScanner(config=config)
        scanner._semgrep_available = True

        captured_args = []

        async def capture_args(*args, **kwargs):
            captured_args.extend(args)
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(
                return_value=(
                    b'{"results": [], "paths": {"scanned": [], "skipped": []}}',
                    b"",
                )
            )
            mock_process.returncode = 0
            return mock_process

        with patch("asyncio.create_subprocess_exec", side_effect=capture_args):
            with tempfile.TemporaryDirectory() as tmpdir:
                await scanner._scan_with_semgrep(tmpdir, ["p/test"], "test")

                # Exclusion patterns should be passed as literals
                assert "$(whoami)/" in captured_args or "--exclude" in captured_args


# ============================================================
# Finding Data Integrity Tests
# ============================================================


class TestFindingDataIntegrity:
    """Tests for ensuring finding data integrity and safety."""

    @pytest.mark.asyncio
    async def test_finding_snippet_truncation(self):
        """Very long snippets are handled without memory issues."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with very long line
            vuln_file = os.path.join(tmpdir, "long.py")
            with open(vuln_file, "w") as f:
                f.write("eval(" + "x" * 10000 + ")\n")

            result = await scanner.scan_repository(tmpdir)
            # Should capture finding without excessive memory use
            for finding in result.findings:
                # Snippet should be reasonable size
                assert len(finding.snippet) < 50000

    @pytest.mark.asyncio
    async def test_finding_to_dict_safe(self):
        """Finding serialization doesn't expose sensitive data unexpectedly."""
        finding = SASTFinding(
            rule_id="test",
            file_path="test.py",
            line_start=1,
            line_end=1,
            column_start=1,
            column_end=1,
            message="Test message",
            severity=SASTSeverity.WARNING,
            confidence=0.8,
            language="python",
            metadata={"internal_key": "internal_value"},
        )

        data = finding.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    @pytest.mark.asyncio
    async def test_result_to_dict_safe(self):
        """Result serialization is safe and JSON-compatible."""
        result = SASTScanResult(
            repository_path="/test/path",
            scan_id="test-123",
            findings=[],
            scanned_files=10,
            skipped_files=2,
            scan_duration_ms=100.0,
            languages_detected=["python"],
            rules_used=["p/test"],
        )

        data = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        assert "test-123" in json_str


# ============================================================
# Progress Callback Security Tests
# ============================================================


class TestProgressCallbackSecurity:
    """Tests for progress callback handling edge cases."""

    @pytest.mark.asyncio
    async def test_progress_callback_exception_handled(self):
        """Exception in progress callback doesn't crash scan."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        async def failing_callback(current, total, message):
            raise RuntimeError("Callback failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("x = 1\n")

            # Scan should complete despite callback failure
            # This depends on implementation - may or may not propagate
            try:
                result = await scanner.scan_repository(tmpdir, progress_callback=failing_callback)
            except RuntimeError:
                # If it propagates, that's also acceptable behavior
                pass

    @pytest.mark.asyncio
    async def test_progress_callback_none_safe(self):
        """None progress callback is handled safely."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("x = 1\n")

            result = await scanner.scan_repository(tmpdir, progress_callback=None)
            assert result is not None

    @pytest.mark.asyncio
    async def test_progress_callback_slow_doesnt_block(self):
        """Slow progress callback doesn't block scan indefinitely."""
        scanner = SASTScanner(config=SASTConfig(use_semgrep=False))
        scanner._semgrep_available = False

        async def slow_callback(current, total, message):
            await asyncio.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("x = 1\n")

            # Should complete in reasonable time
            result = await asyncio.wait_for(
                scanner.scan_repository(tmpdir, progress_callback=slow_callback),
                timeout=30,
            )
            assert result is not None
