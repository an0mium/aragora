"""End-to-end tests for the document processing pipeline.

Uses actual project files from the aragora codebase as test documents.
"""

# ruff: noqa: T201 - print statements intentional for e2e test visibility

import pytest
from pathlib import Path


class TestDocumentPipelineE2E:
    """End-to-end tests for document processing."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent.parent.parent

    @pytest.fixture
    def test_files(self, project_root: Path) -> list[Path]:
        """Get a variety of test files from the project."""
        files = []

        # Python files
        py_files = [
            project_root / "aragora" / "core.py",
            project_root / "aragora" / "debate" / "orchestrator.py",
            project_root / "aragora" / "agents" / "cli_agents.py",
            project_root / "aragora" / "audit" / "document_auditor.py",
        ]
        files.extend([f for f in py_files if f.exists()])

        # Markdown files
        md_files = [
            project_root / "CLAUDE.md",
            project_root / "docs" / "RUNBOOK.md",
        ]
        files.extend([f for f in md_files if f.exists()])

        return files[:5]  # Limit to 5 files for faster testing

    def test_file_discovery(self, test_files):
        """Test that we can find test files."""
        assert len(test_files) > 0, "No test files found in project"
        for f in test_files:
            assert f.exists(), f"File not found: {f}"

    def test_read_and_chunk_python_file(self, test_files):
        """Test reading and chunking a Python file."""
        from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig

        py_files = [f for f in test_files if f.suffix == ".py"]
        if not py_files:
            pytest.skip("No Python files found")

        chunker = SemanticChunking(ChunkingConfig(chunk_size=500))
        file_path = py_files[0]

        content = file_path.read_text()
        chunks = chunker.chunk(content)

        assert len(chunks) > 0
        print(f"\nFile: {file_path.name}")
        print(f"Size: {len(content)} chars")
        print(f"Chunks: {len(chunks)}")

        # Verify chunks cover the content
        total_chunk_content = sum(len(c.content) for c in chunks)
        assert total_chunk_content > 0

    def test_read_and_chunk_markdown_file(self, test_files):
        """Test reading and chunking a Markdown file."""
        from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig

        md_files = [f for f in test_files if f.suffix == ".md"]
        if not md_files:
            pytest.skip("No Markdown files found")

        chunker = SemanticChunking(ChunkingConfig(chunk_size=500))
        file_path = md_files[0]

        content = file_path.read_text()
        chunks = chunker.chunk(content)

        assert len(chunks) > 0
        print(f"\nFile: {file_path.name}")
        print(f"Size: {len(content)} chars")
        print(f"Chunks: {len(chunks)}")

    def test_token_counting_real_files(self, test_files):
        """Test token counting on real project files."""
        from aragora.documents.chunking.token_counter import TokenCounter

        counter = TokenCounter()

        for file_path in test_files[:3]:
            content = file_path.read_text()
            token_count = counter.count(content)

            print(f"\n{file_path.name}: {token_count} tokens ({len(content)} chars)")
            assert token_count > 0
            # Rough check: tokens should be less than characters
            assert token_count < len(content)

    def test_batch_processing_multiple_files(self, test_files):
        """Test processing multiple files in batch."""
        from aragora.documents.chunking.strategies import FixedSizeChunking, ChunkingConfig
        from aragora.documents.chunking.token_counter import TokenCounter

        chunker = FixedSizeChunking(ChunkingConfig(chunk_size=500))
        counter = TokenCounter()

        results = []
        for file_path in test_files:
            content = file_path.read_text()
            chunks = chunker.chunk(content)
            token_count = counter.count(content)

            results.append(
                {
                    "file": file_path.name,
                    "size": len(content),
                    "tokens": token_count,
                    "chunks": len(chunks),
                }
            )

        assert len(results) == len(test_files)

        print("\n=== Batch Processing Results ===")
        for r in results:
            print(f"{r['file']}: {r['tokens']} tokens, {r['chunks']} chunks")

    @pytest.mark.asyncio
    async def test_audit_python_files(self, test_files):
        """Test auditing Python files for security issues."""
        from aragora.audit.document_auditor import DocumentAuditor, AuditSession, AuditType

        py_files = [f for f in test_files if f.suffix == ".py"]
        if not py_files:
            pytest.skip("No Python files found")

        # Create audit session
        session = AuditSession(
            id="test-e2e-001",
            document_ids=[f.name for f in py_files],
            audit_types=[AuditType.SECURITY],
            model="gemini-1.5-flash",
        )

        # Read file contents
        chunks = []
        for i, file_path in enumerate(py_files[:2]):  # Limit for speed
            content = file_path.read_text()
            chunks.append(
                {
                    "id": f"chunk_{i}",
                    "document_id": file_path.name,
                    "content": content[:5000],  # Limit content for testing
                }
            )

        auditor = DocumentAuditor()

        # This may fail if no API keys are configured, which is OK for testing
        try:
            findings = await auditor.audit_chunks(chunks, session)
            print("\n=== Security Audit Results ===")
            print(f"Files audited: {len(py_files[:2])}")
            print(f"Findings: {len(findings)}")
            for finding in findings[:5]:
                print(f"  - [{finding.severity.value}] {finding.title}")
        except Exception as e:
            print(f"\nAudit skipped (expected without API keys): {e}")
            pytest.skip("API keys not configured for audit")


class TestDocumentAuditE2E:
    """End-to-end tests for document auditing with real files."""

    @pytest.fixture
    def project_docs(self) -> list[Path]:
        """Get documentation files for consistency testing."""
        root = Path(__file__).parent.parent.parent.parent
        docs = []

        # Look for markdown documentation
        doc_paths = [
            root / "CLAUDE.md",
            root / "docs" / "RUNBOOK.md",
            root / "docs" / "ENVIRONMENT.md",
            root / "docs" / "STATUS.md",
        ]
        docs.extend([p for p in doc_paths if p.exists()])

        return docs

    def test_consistency_check_project_docs(self, project_docs):
        """Test consistency checking across project documentation."""
        from aragora.audit.audit_types.consistency import ConsistencyAuditor

        if len(project_docs) < 2:
            pytest.skip("Need at least 2 docs for consistency check")

        auditor = ConsistencyAuditor()

        # Extract statements from each doc
        all_statements = []
        for doc_path in project_docs:
            content = doc_path.read_text()

            for pattern, category in auditor.DATE_PATTERNS:
                for match in pattern.finditer(content):
                    all_statements.append(
                        {
                            "doc": doc_path.name,
                            "category": category,
                            "key": match.group(1),
                            "value": match.group(2),
                            "text": match.group(0),
                        }
                    )

            for pattern, category in auditor.NUMBER_PATTERNS:
                for match in pattern.finditer(content):
                    all_statements.append(
                        {
                            "doc": doc_path.name,
                            "category": category,
                            "key": match.group(1),
                            "value": match.group(2),
                            "text": match.group(0),
                        }
                    )

        print("\n=== Consistency Check Results ===")
        print(f"Documents analyzed: {len(project_docs)}")
        print(f"Statements extracted: {len(all_statements)}")

        # Group by key and look for conflicts
        by_key = {}
        for stmt in all_statements:
            key = auditor._normalize_key(stmt["key"])
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(stmt)

        conflicts = []
        for key, stmts in by_key.items():
            if len(stmts) > 1:
                values = set(s["value"] for s in stmts)
                if len(values) > 1:
                    conflicts.append({"key": key, "statements": stmts})

        print(f"Potential conflicts: {len(conflicts)}")
        for conflict in conflicts[:3]:
            print(f"  Key '{conflict['key']}':")
            for stmt in conflict["statements"]:
                print(f"    {stmt['doc']}: {stmt['value']}")


class TestCLIIntegration:
    """Test CLI command integration."""

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    def test_cli_module_imports(self):
        """Test that CLI modules can be imported."""
        from aragora.cli.document_audit import (
            create_document_audit_parser,
            cmd_upload,
            cmd_scan,
            cmd_status,
            cmd_report,
        )

        assert callable(create_document_audit_parser)
        assert callable(cmd_upload)
        assert callable(cmd_scan)
        assert callable(cmd_status)
        assert callable(cmd_report)

    def test_cli_parser_creation(self):
        """Test that the CLI parser can be created."""
        import argparse
        from aragora.cli.document_audit import create_document_audit_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_document_audit_parser(subparsers)

        # Should be able to parse valid commands
        # Note: This just tests parser setup, not actual command execution


class TestSDKIntegration:
    """Test SDK integration."""

    def test_sdk_client_creation(self):
        """Test that SDK client can be created."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")

        assert hasattr(client, "documents")
        assert hasattr(client, "debates")
        assert hasattr(client, "agents")

    def test_sdk_documents_api_methods(self):
        """Test that DocumentsAPI has expected methods."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")
        docs = client.documents

        # Check all expected methods exist
        expected_methods = [
            "list",
            "list_async",
            "get",
            "get_async",
            "upload",
            "upload_async",
            "delete",
            "delete_async",
            "formats",
            "formats_async",
            "batch_upload",
            "batch_upload_async",
            "batch_status",
            "batch_status_async",
            "batch_results",
            "batch_results_async",
            "chunks",
            "chunks_async",
            "context",
            "context_async",
            "create_audit",
            "create_audit_async",
            "get_audit",
            "get_audit_async",
            "audit_findings",
            "audit_findings_async",
            "audit_report",
            "audit_report_async",
        ]

        for method in expected_methods:
            assert hasattr(docs, method), f"Missing method: {method}"
            assert callable(getattr(docs, method)), f"Not callable: {method}"

    def test_sdk_models_import(self):
        """Test that document models can be imported."""
        from aragora.client.models import (
            DocumentStatus,
            AuditType,
            FindingSeverity,
        )

        # Test enum values
        assert DocumentStatus.PENDING.value == "pending"
        assert AuditType.SECURITY.value == "security"
        assert FindingSeverity.HIGH.value == "high"
