#!/usr/bin/env python3
"""
Performance profiling for Aragora.

Identifies bottlenecks in:
- Debate orchestration
- Database queries
- Memory operations
- API response times

Usage:
    python diagnostics/performance_profile.py [--full]
"""

import asyncio
import cProfile
import pstats
import io
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


class PerformanceProfiler:
    """Profile various Aragora subsystems."""

    def __init__(self, nomic_dir: str = ".nomic"):
        self.nomic_dir = Path(nomic_dir)
        self.results: dict = {}

    def profile_database_queries(self) -> dict:
        """Profile common database query patterns."""
        results = {"queries": []}

        # Find all SQLite databases
        db_files = list(self.nomic_dir.glob("**/*.db"))

        for db_path in db_files:
            if not db_path.exists():
                continue

            try:
                conn = sqlite3.connect(str(db_path), timeout=5.0)
                cursor = conn.cursor()

                # Get table info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                for table in tables:
                    start = time.perf_counter()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    elapsed = (time.perf_counter() - start) * 1000

                    results["queries"].append({
                        "db": db_path.name,
                        "table": table,
                        "rows": count,
                        "count_ms": round(elapsed, 2),
                    })

                conn.close()
            except Exception as e:
                results["queries"].append({
                    "db": db_path.name,
                    "error": str(e),
                })

        return results

    def profile_import_times(self) -> dict:
        """Profile module import times."""
        results = {"imports": []}

        modules = [
            "aragora.debate.orchestrator",
            "aragora.ranking.elo",
            "aragora.memory.store",
            "aragora.server.stream",
            "aragora.agents.api_agents",
            "aragora.reasoning.belief",
            "aragora.verification.formal",
        ]

        for module in modules:
            start = time.perf_counter()
            try:
                __import__(module)
                elapsed = (time.perf_counter() - start) * 1000
                results["imports"].append({
                    "module": module,
                    "time_ms": round(elapsed, 2),
                })
            except Exception as e:
                results["imports"].append({
                    "module": module,
                    "error": str(e),
                })

        return results

    def profile_arena_creation(self) -> dict:
        """Profile Arena initialization time."""
        results = {}

        try:
            from aragora.core import Environment, Agent
            from aragora.debate.orchestrator import Arena, DebateProtocol

            # Mock agent for profiling
            class MockAgent(Agent):
                async def generate(self, prompt: str, context=None) -> str:
                    return "Mock response"
                async def critique(self, proposal: str, task: str, context=None):
                    return None
                async def vote(self, proposals: dict, task: str):
                    return None

            env = Environment(task="Profile test task")
            agents = [MockAgent("test1", "mock"), MockAgent("test2", "mock")]
            protocol = DebateProtocol(rounds=1)

            start = time.perf_counter()
            arena = Arena(environment=env, agents=agents, protocol=protocol)
            elapsed = (time.perf_counter() - start) * 1000

            results["arena_init_ms"] = round(elapsed, 2)
            results["arena_attributes"] = len(dir(arena))

        except Exception as e:
            results["error"] = str(e)

        return results

    def profile_elo_operations(self) -> dict:
        """Profile ELO system operations."""
        results = {}

        try:
            from aragora.ranking.elo import EloSystem
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = f"{tmpdir}/profile_elo.db"

                # Init
                start = time.perf_counter()
                elo = EloSystem(db_path=db_path)
                results["init_ms"] = round((time.perf_counter() - start) * 1000, 2)

                # Record 100 matches
                start = time.perf_counter()
                for i in range(100):
                    elo.record_match(f"agent_{i % 5}", f"agent_{(i+1) % 5}", f"agent_{i % 5}")
                results["100_matches_ms"] = round((time.perf_counter() - start) * 1000, 2)

                # Get ratings
                start = time.perf_counter()
                for i in range(5):
                    elo.get_rating(f"agent_{i}")
                results["5_ratings_ms"] = round((time.perf_counter() - start) * 1000, 2)

                # Get leaderboard
                start = time.perf_counter()
                elo.get_leaderboard(limit=10)
                results["leaderboard_ms"] = round((time.perf_counter() - start) * 1000, 2)

        except Exception as e:
            results["error"] = str(e)

        return results

    def profile_memory_operations(self) -> dict:
        """Profile memory store operations."""
        results = {}

        try:
            from aragora.memory.store import CritiqueStore
            from aragora.core import Critique
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = f"{tmpdir}/profile_memory.db"

                # Init
                start = time.perf_counter()
                store = CritiqueStore(db_path=db_path)
                results["init_ms"] = round((time.perf_counter() - start) * 1000, 2)

                # Store 50 patterns
                start = time.perf_counter()
                for i in range(50):
                    critique = Critique(
                        agent=f"agent_{i % 3}",
                        target_agent=f"agent_{(i+1) % 3}",
                        target_content=f"Content {i}",
                        issues=[f"Issue {i}"],
                        suggestions=[f"Suggestion {i}"],
                        severity=0.5,
                        reasoning=f"Reasoning {i}",
                    )
                    store.store_pattern(critique, f"Fix {i}")
                results["50_patterns_ms"] = round((time.perf_counter() - start) * 1000, 2)

                # Retrieve patterns
                start = time.perf_counter()
                patterns = store.retrieve_patterns("test query", limit=10)
                results["retrieve_ms"] = round((time.perf_counter() - start) * 1000, 2)

        except Exception as e:
            results["error"] = str(e)

        return results

    async def run_profile(self, full: bool = False) -> dict:
        """Run all profiling tests."""
        print("=" * 60)
        print("ARAGORA PERFORMANCE PROFILE")
        print(f"Time: {datetime.now().isoformat()}")
        print("=" * 60)

        # Import times
        print("\n[1/5] Profiling import times...")
        self.results["imports"] = self.profile_import_times()

        # Arena creation
        print("[2/5] Profiling Arena creation...")
        self.results["arena"] = self.profile_arena_creation()

        # ELO operations
        print("[3/5] Profiling ELO operations...")
        self.results["elo"] = self.profile_elo_operations()

        # Memory operations
        print("[4/5] Profiling memory operations...")
        self.results["memory"] = self.profile_memory_operations()

        # Database queries
        print("[5/5] Profiling database queries...")
        self.results["database"] = self.profile_database_queries()

        # Summary
        print("\n" + "=" * 60)
        print("PROFILE SUMMARY")
        print("=" * 60)

        # Import times
        print("\n## Import Times (ms)")
        for imp in self.results.get("imports", {}).get("imports", []):
            if "error" not in imp:
                print(f"  {imp['module']}: {imp['time_ms']}ms")

        # Arena
        print("\n## Arena Initialization")
        arena = self.results.get("arena", {})
        if "arena_init_ms" in arena:
            print(f"  Init time: {arena['arena_init_ms']}ms")

        # ELO
        print("\n## ELO Operations")
        elo = self.results.get("elo", {})
        for key in ["init_ms", "100_matches_ms", "5_ratings_ms", "leaderboard_ms"]:
            if key in elo:
                print(f"  {key}: {elo[key]}ms")

        # Memory
        print("\n## Memory Store Operations")
        memory = self.results.get("memory", {})
        for key in ["init_ms", "50_patterns_ms", "retrieve_ms"]:
            if key in memory:
                print(f"  {key}: {memory[key]}ms")

        # Database
        print("\n## Database Tables")
        db = self.results.get("database", {})
        for query in db.get("queries", [])[:10]:
            if "error" not in query:
                print(f"  {query['db']}/{query['table']}: {query['rows']} rows ({query['count_ms']}ms)")

        print("\n" + "=" * 60)

        return self.results


async def main():
    import sys
    full = "--full" in sys.argv

    profiler = PerformanceProfiler()
    await profiler.run_profile(full=full)


if __name__ == "__main__":
    asyncio.run(main())
