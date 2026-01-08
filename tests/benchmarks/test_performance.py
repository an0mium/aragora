"""
Performance benchmarks for critical Aragora subsystems.

Tests performance of:
- Consensus detection algorithms
- Memory tier queries
- ELO rating calculations
- Handler response times

Run with: pytest tests/benchmarks/test_performance.py -v --benchmark-only
Or without benchmark plugin: pytest tests/benchmarks/test_performance.py::TestTimingBaselines -v
"""

import random
import string
import tempfile
import time
from pathlib import Path

import pytest

# Check if pytest-benchmark is available
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Skip benchmark tests if plugin not available
benchmark_skip = pytest.mark.skipif(
    not BENCHMARK_AVAILABLE,
    reason="pytest-benchmark not installed"
)


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for benchmark databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@benchmark_skip
class TestEloPerformance:
    """Benchmark ELO system operations."""

    @pytest.fixture
    def elo_system(self, temp_db_dir):
        """Create ELO system for benchmarks."""
        from aragora.ranking.elo import EloSystem
        return EloSystem(str(temp_db_dir / "elo_benchmark.db"))

    def test_record_match_performance(self, benchmark, elo_system):
        """Benchmark single match recording."""
        def record_single_match():
            agent_a = f"agent-{random.randint(1, 100)}"
            agent_b = f"agent-{random.randint(1, 100)}"
            scores = {agent_a: random.random(), agent_b: 1 - random.random()}
            elo_system.record_match(agent_a, agent_b, scores, "benchmark")

        benchmark(record_single_match)

    def test_get_rating_performance(self, benchmark, elo_system):
        """Benchmark rating retrieval."""
        # Setup: create some agents
        for i in range(100):
            elo_system.record_match(f"agent-{i}", f"agent-{i+1}", {f"agent-{i}": 1.0, f"agent-{i+1}": 0.0}, "setup")

        def get_single_rating():
            agent = f"agent-{random.randint(0, 100)}"
            return elo_system.get_rating(agent)

        benchmark(get_single_rating)

    def test_leaderboard_performance(self, benchmark, elo_system):
        """Benchmark leaderboard generation."""
        # Setup: create many agents
        for i in range(200):
            elo_system.record_match(f"agent-{i}", f"agent-{i+1}", {f"agent-{i}": random.random(), f"agent-{i+1}": random.random()}, "setup")

        def get_leaderboard():
            return elo_system.get_leaderboard(limit=50)

        benchmark(get_leaderboard)


@benchmark_skip
class TestMemoryPerformance:
    """Benchmark memory subsystem operations."""

    @pytest.fixture
    def critique_store(self, temp_db_dir):
        """Create critique store for benchmarks."""
        from aragora.memory.store import CritiqueStore
        return CritiqueStore(str(temp_db_dir / "memory_benchmark.db"))

    def test_pattern_retrieval_performance(self, benchmark, critique_store):
        """Benchmark pattern retrieval."""
        # Setup: store some patterns
        from aragora.core import Critique
        for i in range(100):
            critique = Critique(
                agent=f"critic-{i}",
                target_agent=f"target-{i}",
                target_content=f"Content {i}",
                issues=[f"Issue {i}"],
                suggestions=[f"Suggestion {i}"],
                severity=random.random(),
                reasoning=f"Reasoning {i}"
            )
            critique_store.store_pattern(critique, f"Fix {i}")

        def retrieve_patterns():
            return critique_store.retrieve_patterns(issue_type="general", limit=20)

        benchmark(retrieve_patterns)

    def test_reputation_lookup_performance(self, benchmark, critique_store):
        """Benchmark reputation lookup."""
        # Setup: create reputations via pattern storage
        from aragora.core import Critique
        for i in range(50):
            critique = Critique(
                agent=f"critic-{i % 10}",
                target_agent=f"target-{i % 5}",
                target_content=f"Content",
                issues=["Issue"],
                suggestions=["Suggestion"],
                severity=0.5,
                reasoning="Reasoning"
            )
            critique_store.store_pattern(critique, "Fix")

        def get_reputation():
            agent = f"critic-{random.randint(0, 9)}"
            return critique_store.get_reputation(agent)

        benchmark(get_reputation)


@benchmark_skip
class TestConsensusPerformance:
    """Benchmark consensus detection operations."""

    def test_majority_vote_calculation(self, benchmark):
        """Benchmark majority vote calculation."""
        from aragora.core import Vote

        def calculate_majority():
            votes = []
            agents = [f"agent-{i}" for i in range(10)]
            choices = ["option_a", "option_b", "option_c"]

            for agent in agents:
                votes.append(Vote(
                    agent=agent,
                    choice=random.choice(choices),
                    reasoning="Test reasoning",
                    confidence=random.random(),
                    continue_debate=False
                ))

            # Count votes
            vote_counts = {}
            for vote in votes:
                vote_counts[vote.choice] = vote_counts.get(vote.choice, 0) + 1

            # Find majority
            max_votes = max(vote_counts.values())
            return [c for c, v in vote_counts.items() if v == max_votes]

        benchmark(calculate_majority)

    def test_weighted_vote_calculation(self, benchmark):
        """Benchmark weighted vote calculation."""
        from aragora.core import Vote

        def calculate_weighted():
            votes = []
            agents = [f"agent-{i}" for i in range(20)]
            choices = ["option_a", "option_b", "option_c"]
            weights = {agent: random.uniform(0.5, 2.0) for agent in agents}

            for agent in agents:
                votes.append(Vote(
                    agent=agent,
                    choice=random.choice(choices),
                    reasoning="Test reasoning",
                    confidence=random.random(),
                    continue_debate=False
                ))

            # Weighted count
            weighted_counts = {}
            for vote in votes:
                weight = weights.get(vote.agent, 1.0) * vote.confidence
                weighted_counts[vote.choice] = weighted_counts.get(vote.choice, 0) + weight

            return max(weighted_counts.items(), key=lambda x: x[1])

        benchmark(calculate_weighted)


@benchmark_skip
class TestAuthPerformance:
    """Benchmark authentication operations."""

    def test_token_generation_performance(self, benchmark):
        """Benchmark token generation."""
        from aragora.server.auth import AuthConfig

        config = AuthConfig()
        config.api_token = "benchmark-secret-key"
        config.enabled = True

        def generate_token():
            loop_id = ''.join(random.choices(string.ascii_lowercase, k=10))
            return config.generate_token(loop_id, expires_in=3600)

        benchmark(generate_token)

    def test_token_validation_performance(self, benchmark):
        """Benchmark token validation."""
        from aragora.server.auth import AuthConfig

        config = AuthConfig()
        config.api_token = "benchmark-secret-key"
        config.enabled = True

        # Pre-generate tokens
        tokens = [config.generate_token(f"loop-{i}", 3600) for i in range(100)]

        def validate_token():
            token = random.choice(tokens)
            loop_id = f"loop-{random.randint(0, 99)}"
            return config.validate_token(token, loop_id)

        benchmark(validate_token)

    def test_rate_limit_check_performance(self, benchmark):
        """Benchmark rate limit checking."""
        from aragora.server.auth import AuthConfig

        config = AuthConfig()
        config.rate_limit_per_minute = 1000  # High limit for benchmark

        def check_rate_limit():
            token = f"token-{random.randint(0, 100)}"
            return config.check_rate_limit(token)

        benchmark(check_rate_limit)


@benchmark_skip
class TestHandlerPerformance:
    """Benchmark handler operations."""

    @pytest.fixture
    def handler_context(self, temp_db_dir):
        """Create handler context for benchmarks."""
        from aragora.ranking.elo import EloSystem
        from aragora.memory.store import CritiqueStore

        return {
            "storage": None,
            "elo_system": EloSystem(str(temp_db_dir / "handler_elo.db")),
            "nomic_dir": temp_db_dir,
            "critique_store": CritiqueStore(str(temp_db_dir / "handler_memory.db")),
        }

    def test_health_check_performance(self, benchmark, handler_context):
        """Benchmark health check endpoint."""
        from aragora.server.handlers import SystemHandler

        handler = SystemHandler(handler_context)

        def health_check():
            return handler.handle("/api/health", {}, None)

        benchmark(health_check)

    def test_leaderboard_handler_performance(self, benchmark, handler_context):
        """Benchmark leaderboard handler."""
        from aragora.server.handlers import AgentsHandler

        # Setup: add some agents
        elo = handler_context["elo_system"]
        for i in range(100):
            elo.record_match(f"agent-{i}", f"agent-{i+1}", {f"agent-{i}": random.random(), f"agent-{i+1}": random.random()}, "bench")

        handler = AgentsHandler(handler_context)

        def get_leaderboard():
            return handler.handle("/api/leaderboard", {"limit": "20"}, None)

        benchmark(get_leaderboard)


@benchmark_skip
class TestDatabasePerformance:
    """Benchmark database operations."""

    def test_sqlite_write_performance(self, benchmark, temp_db_dir):
        """Benchmark raw SQLite write performance."""
        import sqlite3

        db_path = temp_db_dir / "raw_bench.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()

        counter = [0]

        def write_row():
            counter[0] += 1
            conn.execute("INSERT INTO test (data) VALUES (?)", (f"data-{counter[0]}",))
            conn.commit()

        benchmark(write_row)
        conn.close()

    def test_sqlite_read_performance(self, benchmark, temp_db_dir):
        """Benchmark raw SQLite read performance."""
        import sqlite3

        db_path = temp_db_dir / "raw_read_bench.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)")

        # Setup: insert data
        for i in range(1000):
            conn.execute("INSERT INTO test (data) VALUES (?)", (f"data-{i}",))
        conn.commit()

        def read_rows():
            cursor = conn.execute("SELECT * FROM test WHERE id BETWEEN ? AND ?",
                                  (random.randint(1, 500), random.randint(501, 1000)))
            return cursor.fetchall()

        benchmark(read_rows)
        conn.close()


@benchmark_skip
class TestStringOperations:
    """Benchmark string operations used in processing."""

    def test_json_serialization(self, benchmark):
        """Benchmark JSON serialization."""
        import json

        data = {
            "debate_id": "test-debate-123",
            "rounds": [
                {"round": i, "messages": [{"agent": f"agent-{j}", "content": f"Message {i}-{j}"} for j in range(5)]}
                for i in range(10)
            ],
            "metadata": {"topic": "Test topic", "created_at": "2026-01-08T00:00:00Z"}
        }

        def serialize():
            return json.dumps(data)

        benchmark(serialize)

    def test_json_deserialization(self, benchmark):
        """Benchmark JSON deserialization."""
        import json

        json_str = json.dumps({
            "debate_id": "test-debate-123",
            "rounds": [
                {"round": i, "messages": [{"agent": f"agent-{j}", "content": f"Message {i}-{j}"} for j in range(5)]}
                for i in range(10)
            ],
            "metadata": {"topic": "Test topic", "created_at": "2026-01-08T00:00:00Z"}
        })

        def deserialize():
            return json.loads(json_str)

        benchmark(deserialize)


# Timing utilities for non-benchmark comparisons
class TestTimingBaselines:
    """Establish timing baselines for operations."""

    def test_elo_batch_operations_timing(self, temp_db_dir):
        """Measure batch ELO operation timing."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(str(temp_db_dir / "timing_elo.db"))

        # Time 1000 match recordings
        start = time.perf_counter()
        for i in range(1000):
            elo.record_match(f"a-{i % 50}", f"b-{i % 50}", {f"a-{i % 50}": 0.6, f"b-{i % 50}": 0.4}, "timing")
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 5 seconds for 1000 matches)
        assert elapsed < 5.0, f"1000 matches took {elapsed:.2f}s (expected < 5s)"
        print(f"\n1000 match recordings: {elapsed:.3f}s ({elapsed/1000*1000:.2f}ms per match)")

    def test_handler_batch_timing(self, temp_db_dir):
        """Measure batch handler operation timing."""
        from aragora.server.handlers import SystemHandler
        from aragora.ranking.elo import EloSystem

        ctx = {
            "storage": None,
            "elo_system": EloSystem(str(temp_db_dir / "handler_timing.db")),
            "nomic_dir": temp_db_dir,
        }
        handler = SystemHandler(ctx)

        # Time 1000 health checks
        start = time.perf_counter()
        for _ in range(1000):
            handler.handle("/api/health", {}, None)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 2 seconds for 1000 requests)
        assert elapsed < 2.0, f"1000 health checks took {elapsed:.2f}s (expected < 2s)"
        print(f"\n1000 health checks: {elapsed:.3f}s ({elapsed/1000*1000:.2f}ms per request)")
