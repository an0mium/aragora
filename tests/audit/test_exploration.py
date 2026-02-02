"""Tests for the iterative document exploration module."""

import pytest
from pathlib import Path


class TestExplorationSession:
    """Tests for ExplorationSession state management."""

    def test_session_creation(self):
        """Test creating an exploration session."""
        from aragora.audit.exploration.session import ExplorationSession, ExplorationPhase

        session = ExplorationSession(
            objective="Find security issues",
            document_ids=["doc1.md", "doc2.py"],
        )

        assert session.id is not None
        assert session.objective == "Find security issues"
        assert len(session.document_ids) == 2
        assert session.current_phase == ExplorationPhase.READ
        assert session.iteration == 0
        assert not session.is_complete

    def test_session_progress_tracking(self):
        """Test tracking exploration progress."""
        from aragora.audit.exploration.session import (
            ExplorationSession,
            ChunkUnderstanding,
        )

        session = ExplorationSession(
            objective="Test",
            document_ids=["doc1.md"],
        )

        # Add pending chunks
        session.chunks_pending = ["chunk1", "chunk2", "chunk3"]
        assert session.progress == 0.0

        # Mark one as explored
        understanding = ChunkUnderstanding(
            chunk_id="chunk1",
            document_id="doc1.md",
            summary="Test summary",
            confidence=0.8,
        )
        session.mark_chunk_explored("chunk1", understanding)

        assert "chunk1" in session.chunks_explored
        assert "chunk1" not in session.chunks_pending
        assert session.progress == pytest.approx(0.333, rel=0.1)

    def test_session_checkpoint(self):
        """Test session checkpoint and restore."""
        from aragora.audit.exploration.session import (
            ExplorationSession,
            ExplorationPhase,
            Insight,
        )

        session = ExplorationSession(
            objective="Test",
            document_ids=["doc1.md"],
        )
        session.iteration = 5
        session.current_phase = ExplorationPhase.VERIFY
        session.chunks_explored = ["chunk1", "chunk2"]
        session.insights.append(
            Insight(
                title="Test insight",
                description="Description",
                confidence=0.9,
            )
        )

        # Checkpoint
        checkpoint = session.to_checkpoint()
        assert checkpoint["iteration"] == 5
        assert checkpoint["current_phase"] == "verify"
        assert len(checkpoint["insights"]) == 1

        # Restore
        restored = ExplorationSession.from_checkpoint(checkpoint)
        assert restored.iteration == 5
        assert restored.current_phase == ExplorationPhase.VERIFY
        assert len(restored.insights) == 1


class TestExplorationMemory:
    """Tests for ExplorationMemory."""

    @pytest.fixture
    def memory(self):
        """Create a memory instance."""
        from aragora.audit.exploration.memory import ExplorationMemory

        return ExplorationMemory(enable_embeddings=False)

    @pytest.mark.asyncio
    async def test_store_insight(self, memory):
        """Test storing an insight."""
        from aragora.audit.exploration.session import Insight
        from aragora.audit.exploration.memory import MemoryTier

        insight = Insight(
            title="Security Issue",
            description="Found potential SQL injection",
            confidence=0.9,
        )

        stored = await memory.store_insight(
            insight,
            tier=MemoryTier.FAST,
            importance=0.8,
            session_id="test_session",
        )

        assert stored.id == insight.id
        assert stored.tier == MemoryTier.FAST
        assert stored.importance == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_relevant(self, memory):
        """Test retrieving relevant insights."""
        from aragora.audit.exploration.session import Insight
        from aragora.audit.exploration.memory import MemoryTier

        # Store some insights
        await memory.store_insight(
            Insight(title="SQL Injection", description="Database query vulnerability"),
            tier=MemoryTier.FAST,
        )
        await memory.store_insight(
            Insight(title="XSS Issue", description="Cross-site scripting found"),
            tier=MemoryTier.FAST,
        )
        await memory.store_insight(
            Insight(title="Performance", description="Slow database queries"),
            tier=MemoryTier.FAST,
        )

        # Search for database-related
        results = await memory.retrieve_relevant("database vulnerability", limit=5)
        assert len(results) >= 1
        # SQL Injection should be more relevant
        titles = [r.insight.title for r in results]
        assert "SQL Injection" in titles or "Performance" in titles

    @pytest.mark.asyncio
    async def test_tier_promotion(self, memory):
        """Test promoting insights between tiers."""
        from aragora.audit.exploration.session import Insight
        from aragora.audit.exploration.memory import MemoryTier

        insight = Insight(title="Important", description="Verified finding")
        _stored = await memory.store_insight(insight, tier=MemoryTier.FAST)

        # Promote to SLOW
        promoted = await memory.promote_insight(insight.id, MemoryTier.SLOW)
        assert promoted.tier == MemoryTier.SLOW

        # Verify it's in the new tier
        slow_insights = await memory.retrieve_by_tier(MemoryTier.SLOW)
        assert len(slow_insights) == 1

    def test_memory_stats(self, memory):
        """Test memory statistics."""
        stats = memory.get_stats()
        assert "total_insights" in stats
        assert "by_tier" in stats
        assert stats["total_insights"] == 0


class TestQueryGenerator:
    """Tests for QueryGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a query generator."""
        from aragora.audit.exploration.query_gen import QueryGenerator

        return QueryGenerator()

    def test_identify_gaps_low_confidence(self, generator):
        """Test identifying gaps from low-confidence understandings."""
        from aragora.audit.exploration.session import ChunkUnderstanding

        understandings = [
            ChunkUnderstanding(
                chunk_id="chunk1",
                document_id="doc1",
                summary="Some content",
                confidence=0.3,  # Low confidence
            ),
        ]

        gaps = generator.identify_gaps(understandings, [], "Find issues")
        assert len(gaps) >= 1
        assert gaps[0].gap_type == "unclear"

    def test_identify_gaps_questions_raised(self, generator):
        """Test identifying gaps from raised questions."""
        from aragora.audit.exploration.session import ChunkUnderstanding

        understandings = [
            ChunkUnderstanding(
                chunk_id="chunk1",
                document_id="doc1",
                summary="Content",
                confidence=0.8,
                questions_raised=["What is X?", "Why does Y happen?"],
            ),
        ]

        gaps = generator.identify_gaps(understandings, [], "Find issues")
        # Should have gaps for the raised questions
        assert len(gaps) >= 2
        assert any("What is X?" in g.description for g in gaps)

    @pytest.mark.asyncio
    async def test_generate_questions(self, generator):
        """Test generating questions from gaps."""
        from aragora.audit.exploration.query_gen import UnderstandingGap

        gaps = [
            UnderstandingGap(
                description="What does API key mean?",
                gap_type="missing_info",
                priority=0.8,
            ),
        ]

        questions = await generator.generate_questions(gaps, "Security audit")
        assert len(questions) >= 1
        assert questions[0].priority > 0

    def test_question_deduplication(self, generator):
        """Test that duplicate questions are filtered."""

        # Mark a question as asked
        generator._asked_questions.add("security audit api key")

        # Similar question should be detected as duplicate
        assert generator._is_duplicate("What is the API key for security audit?")

    def test_prioritize_questions(self, generator):
        """Test question prioritization by objective relevance."""
        from aragora.audit.exploration.session import Question

        questions = [
            Question(text="What is the database schema?", priority=0.5),
            Question(text="How does authentication work?", priority=0.5),
            Question(text="What color is the logo?", priority=0.5),
        ]

        prioritized = generator.prioritize_questions(
            questions,
            objective="Security authentication audit",
        )

        # Authentication question should be boosted
        auth_q = next(q for q in prioritized if "authentication" in q.text.lower())
        color_q = next(q for q in prioritized if "color" in q.text.lower())
        assert auth_q.priority > color_q.priority


class TestCodebaseAuditor:
    """Tests for CodebaseAuditor."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root."""
        return Path(__file__).parent.parent.parent.parent

    @pytest.fixture
    def auditor(self, project_root):
        """Create a codebase auditor."""
        from aragora.audit.codebase_auditor import CodebaseAuditor, CodebaseAuditConfig

        config = CodebaseAuditConfig(
            include_paths=["aragora/core.py"],  # Just one file for speed
            max_findings_per_cycle=5,
        )
        return CodebaseAuditor(root_path=project_root, config=config)

    def test_file_collection(self, auditor):
        """Test collecting files to audit."""
        files = auditor._collect_files()
        assert len(files) >= 1
        assert any("core.py" in str(f) for f in files)

    def test_security_pattern_detection(self, auditor):
        """Test detecting security patterns in code."""
        # Code with potential issues
        code = """
        api_key = "sk-abc123secretkey456"
        password = 'hardcoded_password'
        cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
        """

        findings = auditor._check_security_patterns(code, "test.py", "chunk1")
        assert len(findings) >= 1
        # Should find the potential secrets
        categories = [f.category for f in findings]
        assert "hardcoded_secret" in categories or "sql_injection" in categories

    def test_quality_pattern_detection(self, auditor):
        """Test detecting quality patterns.

        Note: The TODO/FIXME comments below are INTENTIONAL test fixture data
        used to verify the auditor correctly detects these comment patterns.
        They are not actual issues in the codebase.
        """
        code = """
        # TODO: Fix this later
        # FIXME: Security issue here
        def long_function():
            pass
        """

        findings = auditor._check_quality_patterns(code, "test.py", "chunk1")
        assert len(findings) >= 1
        assert any(f.category == "incomplete_code" for f in findings)

    def test_findings_to_proposals(self, auditor):
        """Test converting findings to proposals."""
        from aragora.audit.document_auditor import AuditFinding, AuditType, FindingSeverity

        findings = [
            AuditFinding(
                audit_type=AuditType.SECURITY,
                category="hardcoded_secret",
                title="API Key exposed",
                description="Found hardcoded API key",
                severity=FindingSeverity.HIGH,
                confidence=0.9,
                session_id="test",
            ),
            AuditFinding(
                audit_type=AuditType.SECURITY,
                category="hardcoded_secret",
                title="Password exposed",
                description="Found hardcoded password",
                severity=FindingSeverity.HIGH,
                confidence=0.8,
                session_id="test",
            ),
        ]

        proposals = auditor.findings_to_proposals(findings)
        assert len(proposals) >= 1
        # Should be grouped by category
        assert proposals[0].severity == FindingSeverity.HIGH


class TestDocumentExplorerIntegration:
    """Integration tests for DocumentExplorer."""

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    def test_explorer_creation(self):
        """Test creating a document explorer."""
        from aragora.audit.exploration.explorer import (
            DocumentExplorer,
            ExplorerConfig,
        )
        from aragora.audit.exploration.agents import ExplorationAgent

        # Create with mock agent
        class MockAgent(ExplorationAgent):
            def __init__(self):
                self.name = "mock"
                self.model = "mock"
                self.role = "explorer"
                self.config = None

            async def generate(self, prompt, context=None):
                return '{"summary": "test"}'

        config = ExplorerConfig(max_iterations=3, enable_verification=False)
        explorer = DocumentExplorer(
            agents=[MockAgent()],
            config=config,
        )

        assert explorer.config.max_iterations == 3
        assert len(explorer.agents) == 1

    def test_default_document_loader(self, project_root):
        """Test the default document loader."""
        from aragora.audit.exploration.explorer import DocumentExplorer
        from aragora.audit.exploration.agents import ExplorationAgent

        class MockAgent(ExplorationAgent):
            def __init__(self):
                self.name = "mock"
                self.model = "mock"
                self.role = "explorer"
                self.config = None

            async def generate(self, prompt, context=None):
                return '{"summary": "test"}'

        explorer = DocumentExplorer(agents=[MockAgent()])

        # Should load existing file
        content = explorer.document_loader(str(project_root / "CLAUDE.md"))
        assert len(content) > 0
        assert "aragora" in content.lower()

        # Should raise for non-existent file
        with pytest.raises(FileNotFoundError):
            explorer.document_loader("nonexistent_file.txt")


class TestExplorationAgentPatterns:
    """Tests for ExplorationAgent prompts and parsing."""

    def test_json_parsing(self):
        """Test JSON response parsing."""
        from aragora.audit.exploration.agents import ExplorationAgent

        class MockAgent(ExplorationAgent):
            def __init__(self):
                self.name = "mock"
                self.model = "mock"
                self.role = "explorer"
                self.config = None

            async def generate(self, prompt, context=None):
                return '{"key": "value"}'

        agent = MockAgent()

        # Test clean JSON
        result = agent._parse_json_response('{"key": "value"}')
        assert result["key"] == "value"

        # Test JSON in markdown code block
        result = agent._parse_json_response('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

        # Test JSON with surrounding text
        result = agent._parse_json_response('Here is the result: {"key": "value"} done')
        assert result["key"] == "value"

        # Test JSON array
        result = agent._parse_json_response('[{"a": 1}, {"b": 2}]')
        assert len(result) == 2
