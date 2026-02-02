"""Tests for Codebase namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestCodebaseAnalyze:
    """Tests for codebase analysis."""

    def test_analyze_repository(self) -> None:
        """Analyze a repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "repo": "myorg/myrepo",
                "analysis_id": "ana_123",
                "status": "completed",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.analyze("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/analyze", json={}
            )
            assert result["analysis_id"] == "ana_123"
            client.close()

    def test_analyze_with_options(self) -> None:
        """Analyze a repository with custom options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"analysis_id": "ana_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            options = {"depth": "full", "include_tests": True}
            client.codebase.analyze("myorg/myrepo", options=options)

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/analyze", json=options
            )
            client.close()

    def test_understand_codebase(self) -> None:
        """Query codebase understanding."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "answer": "The authentication flow uses JWT tokens",
                "references": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.understand("myorg/myrepo", "How does auth work?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/myorg/myrepo/understand",
                json={"query": "How does auth work?"},
            )
            assert "answer" in result
            client.close()


class TestCodebaseSymbols:
    """Tests for code symbols operations."""

    def test_get_symbols(self) -> None:
        """Get code symbols from repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"symbols": [{"name": "MyClass", "type": "class"}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_symbols("myorg/myrepo")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/symbols", params={}
            )
            assert len(result["symbols"]) == 1
            client.close()

    def test_get_symbols_filtered_by_file(self) -> None:
        """Get symbols filtered by file path."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"symbols": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_symbols("myorg/myrepo", file_path="src/main.py")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/symbols",
                params={"file_path": "src/main.py"},
            )
            client.close()

    def test_get_callgraph(self) -> None:
        """Get call graph for repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"nodes": [], "edges": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_callgraph("myorg/myrepo")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/callgraph", params={}
            )
            assert "nodes" in result
            client.close()

    def test_get_callgraph_centered_on_function(self) -> None:
        """Get call graph centered on specific function."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"nodes": [], "edges": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_callgraph("myorg/myrepo", function="main")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/callgraph",
                params={"function": "main"},
            )
            client.close()

    def test_analyze_impact(self) -> None:
        """Analyze impact of file changes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "affected_files": ["tests/test_main.py"],
                "impact_score": 0.7,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.analyze_impact(
                "myorg/myrepo", files=["src/main.py", "src/utils.py"]
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/myorg/myrepo/impact",
                json={"files": ["src/main.py", "src/utils.py"]},
            )
            assert result["impact_score"] == 0.7
            client.close()


class TestCodebaseCodeQuality:
    """Tests for code quality operations."""

    def test_get_deadcode(self) -> None:
        """Find dead code in repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "deadcode": [{"file": "old_utils.py", "function": "unused_helper"}]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_deadcode("myorg/myrepo")

            mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/deadcode")
            assert len(result["deadcode"]) == 1
            client.close()

    def test_get_duplicates(self) -> None:
        """Find code duplicates."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"duplicates": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_duplicates("myorg/myrepo")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/duplicates",
                params={"min_lines": 10},
            )
            client.close()

    def test_get_duplicates_custom_min_lines(self) -> None:
        """Find code duplicates with custom minimum lines."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"duplicates": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_duplicates("myorg/myrepo", min_lines=20)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/duplicates",
                params={"min_lines": 20},
            )
            client.close()

    def test_get_hotspots(self) -> None:
        """Find code hotspots."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"hotspots": [{"file": "core.py", "change_frequency": 15}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_hotspots("myorg/myrepo")

            mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/hotspots")
            assert result["hotspots"][0]["file"] == "core.py"
            client.close()


class TestCodebaseMetrics:
    """Tests for code metrics operations."""

    def test_get_metrics(self) -> None:
        """Get code metrics for repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "lines_of_code": 10000,
                "complexity": 25,
                "coverage": 0.85,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_metrics("myorg/myrepo")

            mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/metrics")
            assert result["coverage"] == 0.85
            client.close()

    def test_analyze_metrics(self) -> None:
        """Run metrics analysis."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"analysis_id": "met_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.analyze_metrics("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/metrics/analyze", json={}
            )
            client.close()

    def test_get_metrics_analysis(self) -> None:
        """Get metrics analysis results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"analysis_id": "met_123", "status": "completed"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_metrics_analysis("myorg/myrepo", "met_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/metrics/met_123"
            )
            client.close()

    def test_get_file_metrics(self) -> None:
        """Get metrics for a specific file."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"complexity": 5, "lines": 100}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_file_metrics("myorg/myrepo", "src/main.py")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/metrics/file/src/main.py"
            )
            client.close()

    def test_get_metrics_history(self) -> None:
        """Get metrics history."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_metrics_history("myorg/myrepo", days=7)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/metrics/history",
                params={"days": 7},
            )
            client.close()


class TestCodebaseScanning:
    """Tests for security scanning operations."""

    def test_scan(self) -> None:
        """Run a security scan."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "scn_123", "status": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.scan("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/myorg/myrepo/scan",
                json={"scan_type": "full"},
            )
            assert result["scan_id"] == "scn_123"
            client.close()

    def test_scan_quick(self) -> None:
        """Run a quick security scan."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "scn_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.scan("myorg/myrepo", scan_type="quick")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/myorg/myrepo/scan",
                json={"scan_type": "quick"},
            )
            client.close()

    def test_get_scan(self) -> None:
        """Get scan results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "scn_123", "findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_scan("myorg/myrepo", "scn_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/scan/scn_123"
            )
            client.close()

    def test_get_latest_scan(self) -> None:
        """Get latest scan results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "scn_latest"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_latest_scan("myorg/myrepo")

            mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/scan/latest")
            client.close()

    def test_list_scans(self) -> None:
        """List all scans for repository."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scans": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.list_scans("myorg/myrepo", limit=50)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/scans", params={"limit": 50}
            )
            client.close()


class TestCodebaseSAST:
    """Tests for SAST scanning operations."""

    def test_scan_sast(self) -> None:
        """Run SAST scan."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "sast_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.scan_sast("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/scan/sast", json={}
            )
            client.close()

    def test_get_sast_scan(self) -> None:
        """Get SAST scan results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_sast_scan("myorg/myrepo", "sast_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/scan/sast/sast_123"
            )
            client.close()

    def test_get_sast_findings(self) -> None:
        """Get SAST findings."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_sast_findings("myorg/myrepo")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/sast/findings", params={}
            )
            client.close()

    def test_get_sast_findings_by_severity(self) -> None:
        """Get SAST findings filtered by severity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_sast_findings("myorg/myrepo", severity="critical")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/codebase/myorg/myrepo/sast/findings",
                params={"severity": "critical"},
            )
            client.close()

    def test_get_owasp_summary(self) -> None:
        """Get OWASP Top 10 summary."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"owasp": {"A01": 2, "A02": 1}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.codebase.get_owasp_summary("myorg/myrepo")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/sast/owasp-summary"
            )
            assert "owasp" in result
            client.close()


class TestCodebaseSecrets:
    """Tests for secrets detection operations."""

    def test_scan_secrets(self) -> None:
        """Scan for secrets in code."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "sec_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.scan_secrets("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/scan/secrets"
            )
            client.close()

    def test_get_secrets_scan(self) -> None:
        """Get secrets scan results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_secrets_scan("myorg/myrepo", "sec_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/scan/secrets/sec_123"
            )
            client.close()

    def test_get_secrets(self) -> None:
        """Get all detected secrets."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"secrets": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_secrets("myorg/myrepo")

            mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/secrets")
            client.close()


class TestCodebaseDependencies:
    """Tests for dependency and SBOM operations."""

    def test_analyze_dependencies(self) -> None:
        """Analyze dependencies from manifest."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"dependencies": [], "vulnerabilities": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            manifest = '{"dependencies": {"react": "^18.0.0"}}'
            client.codebase.analyze_dependencies(manifest, manifest_type="npm")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/analyze-dependencies",
                json={"manifest": manifest, "type": "npm"},
            )
            client.close()

    def test_generate_sbom(self) -> None:
        """Generate Software Bill of Materials."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sbom": {}, "format": "spdx"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.generate_sbom("myorg/myrepo", format="cyclonedx")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/sbom",
                json={"repo": "myorg/myrepo", "format": "cyclonedx"},
            )
            client.close()

    def test_check_licenses(self) -> None:
        """Check dependency licenses."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"compliant": True, "licenses": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.check_licenses("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/check-licenses",
                json={"repo": "myorg/myrepo"},
            )
            client.close()

    def test_scan_vulnerabilities(self) -> None:
        """Scan for known vulnerabilities."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"vulnerabilities": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.scan_vulnerabilities("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/scan-vulnerabilities",
                json={"repo": "myorg/myrepo"},
            )
            client.close()


class TestCodebaseAudit:
    """Tests for audit operations."""

    def test_audit(self) -> None:
        """Run a code audit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"audit_id": "aud_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.audit("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/codebase/myorg/myrepo/audit", json={}
            )
            client.close()

    def test_get_audit(self) -> None:
        """Get audit results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"audit_id": "aud_123", "findings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.get_audit("myorg/myrepo", "aud_123")

            mock_request.assert_called_once_with(
                "GET", "/api/v1/codebase/myorg/myrepo/audit/aud_123"
            )
            client.close()

    def test_clear_cache(self) -> None:
        """Clear analysis cache."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cleared": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.codebase.clear_cache("myorg/myrepo")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/codebase/clear-cache",
                json={"repo": "myorg/myrepo"},
            )
            client.close()


class TestAsyncCodebase:
    """Tests for async codebase API."""

    @pytest.mark.asyncio
    async def test_async_analyze(self) -> None:
        """Analyze a repository asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"analysis_id": "ana_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.codebase.analyze("myorg/myrepo")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/codebase/myorg/myrepo/analyze", json={}
                )
                assert result["analysis_id"] == "ana_123"

    @pytest.mark.asyncio
    async def test_async_get_metrics(self) -> None:
        """Get metrics asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"coverage": 0.9}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.codebase.get_metrics("myorg/myrepo")

                mock_request.assert_called_once_with("GET", "/api/v1/codebase/myorg/myrepo/metrics")
                assert result["coverage"] == 0.9

    @pytest.mark.asyncio
    async def test_async_scan(self) -> None:
        """Run a security scan asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"scan_id": "scn_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.codebase.scan("myorg/myrepo", scan_type="sast")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/codebase/myorg/myrepo/scan",
                    json={"scan_type": "sast"},
                )

    @pytest.mark.asyncio
    async def test_async_scan_vulnerabilities(self) -> None:
        """Scan for vulnerabilities asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"vulnerabilities": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.codebase.scan_vulnerabilities("myorg/myrepo")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/codebase/scan-vulnerabilities",
                    json={"repo": "myorg/myrepo"},
                )
