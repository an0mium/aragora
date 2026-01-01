"""
Prompt evolution system.

Enables agents to improve their system prompts based on successful patterns
observed in debates. Implements self-improvement through pattern mining
and prompt refinement.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import sqlite3

from aagora.core import Agent, DebateResult, Critique
from aagora.memory.store import CritiqueStore, Pattern


class EvolutionStrategy(Enum):
    """Strategies for prompt evolution."""

    APPEND = "append"  # Add new instructions to existing prompt
    REPLACE = "replace"  # Replace sections of the prompt
    REFINE = "refine"  # Use LLM to refine the prompt
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class PromptVersion:
    """A version of an agent's prompt."""

    version: int
    prompt: str
    created_at: str
    performance_score: float = 0.0
    debates_count: int = 0
    consensus_rate: float = 0.0
    metadata: dict = field(default_factory=dict)


class PromptEvolver:
    """
    Evolves agent prompts based on successful debate patterns.

    The evolver:
    1. Mines winning patterns from successful debates
    2. Extracts effective critique and response strategies
    3. Updates agent system prompts to incorporate learnings
    4. Tracks prompt versions and their performance
    """

    def __init__(
        self,
        db_path: str = "aagora_evolution.db",
        critique_store: CritiqueStore = None,
        strategy: EvolutionStrategy = EvolutionStrategy.APPEND,
    ):
        self.db_path = Path(db_path)
        self.critique_store = critique_store
        self.strategy = strategy
        self._init_db()

    def _init_db(self):
        """Initialize evolution database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prompt versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                performance_score REAL DEFAULT 0.0,
                debates_count INTEGER DEFAULT 0,
                consensus_rate REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, version)
            )
        """)

        # Extracted patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_text TEXT NOT NULL,
                source_debate_id TEXT,
                effectiveness_score REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Evolution history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                from_version INTEGER,
                to_version INTEGER,
                strategy TEXT,
                patterns_applied TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def extract_winning_patterns(
        self,
        debates: list[DebateResult],
        min_confidence: float = 0.6,
    ) -> list[dict]:
        """
        Extract patterns from successful debates.

        Returns patterns that led to high-confidence consensus.
        """
        patterns = []

        for debate in debates:
            if not debate.consensus_reached or debate.confidence < min_confidence:
                continue

            # Extract critique patterns
            for critique in debate.critiques:
                if critique.severity < 0.7:  # Lower severity = issue was addressed
                    for issue in critique.issues:
                        patterns.append({
                            "type": "issue_identification",
                            "text": issue,
                            "severity": critique.severity,
                            "source_debate": debate.id,
                        })
                    for suggestion in critique.suggestions:
                        patterns.append({
                            "type": "improvement_suggestion",
                            "text": suggestion,
                            "severity": critique.severity,
                            "source_debate": debate.id,
                        })

            # Extract response patterns from final answer
            if debate.final_answer:
                # Look for structural patterns
                if "```" in debate.final_answer:
                    patterns.append({
                        "type": "includes_code",
                        "text": "Include code examples in responses",
                        "source_debate": debate.id,
                    })
                if any(marker in debate.final_answer.lower() for marker in ["step 1", "first,", "1.", "1)"]):
                    patterns.append({
                        "type": "structured_response",
                        "text": "Use numbered steps or structured format",
                        "source_debate": debate.id,
                    })

        return patterns

    def store_patterns(self, patterns: list[dict]):
        """Store extracted patterns in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for pattern in patterns:
            cursor.execute(
                """
                INSERT INTO extracted_patterns (pattern_type, pattern_text, source_debate_id)
                VALUES (?, ?, ?)
            """,
                (pattern["type"], pattern["text"], pattern.get("source_debate")),
            )

        conn.commit()
        conn.close()

    def get_top_patterns(
        self,
        pattern_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get most effective patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if pattern_type:
            cursor.execute(
                """
                SELECT pattern_type, pattern_text, effectiveness_score, usage_count
                FROM extracted_patterns
                WHERE pattern_type = ?
                ORDER BY effectiveness_score DESC, usage_count DESC
                LIMIT ?
            """,
                (pattern_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT pattern_type, pattern_text, effectiveness_score, usage_count
                FROM extracted_patterns
                ORDER BY effectiveness_score DESC, usage_count DESC
                LIMIT ?
            """,
                (limit,),
            )

        patterns = [
            {
                "type": row[0],
                "text": row[1],
                "effectiveness": row[2],
                "usage_count": row[3],
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return patterns

    def get_prompt_version(self, agent_name: str, version: int = None) -> Optional[PromptVersion]:
        """Get a specific prompt version or the latest."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if version is not None:
            cursor.execute(
                """
                SELECT version, prompt, performance_score, debates_count, consensus_rate, metadata, created_at
                FROM prompt_versions
                WHERE agent_name = ? AND version = ?
            """,
                (agent_name, version),
            )
        else:
            cursor.execute(
                """
                SELECT version, prompt, performance_score, debates_count, consensus_rate, metadata, created_at
                FROM prompt_versions
                WHERE agent_name = ?
                ORDER BY version DESC
                LIMIT 1
            """,
                (agent_name,),
            )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return PromptVersion(
            version=row[0],
            prompt=row[1],
            performance_score=row[2],
            debates_count=row[3],
            consensus_rate=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
            created_at=row[6],
        )

    def save_prompt_version(self, agent_name: str, prompt: str, metadata: dict = None) -> int:
        """Save a new prompt version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next version number
        cursor.execute(
            "SELECT MAX(version) FROM prompt_versions WHERE agent_name = ?",
            (agent_name,),
        )
        row = cursor.fetchone()
        next_version = (row[0] or 0) + 1

        cursor.execute(
            """
            INSERT INTO prompt_versions (agent_name, version, prompt, metadata)
            VALUES (?, ?, ?, ?)
        """,
            (agent_name, next_version, prompt, json.dumps(metadata or {})),
        )

        conn.commit()
        conn.close()

        return next_version

    def evolve_prompt(
        self,
        agent: Agent,
        patterns: list[dict] = None,
        strategy: EvolutionStrategy = None,
    ) -> str:
        """
        Evolve an agent's prompt based on patterns.

        Returns the new prompt.
        """
        strategy = strategy or self.strategy
        patterns = patterns or self.get_top_patterns(limit=5)

        current_prompt = agent.system_prompt or ""

        if strategy == EvolutionStrategy.APPEND:
            return self._evolve_append(current_prompt, patterns)
        elif strategy == EvolutionStrategy.REPLACE:
            return self._evolve_replace(current_prompt, patterns)
        elif strategy == EvolutionStrategy.REFINE:
            return self._evolve_refine(current_prompt, patterns)
        elif strategy == EvolutionStrategy.HYBRID:
            # Try append first, then refine if prompt gets too long
            new_prompt = self._evolve_append(current_prompt, patterns)
            if len(new_prompt) > 2000:
                return self._evolve_refine(current_prompt, patterns)
            return new_prompt
        else:
            return current_prompt

    def _evolve_append(self, current_prompt: str, patterns: list[dict]) -> str:
        """Append new learnings to prompt."""
        learnings = []

        for pattern in patterns:
            if pattern["type"] == "issue_identification":
                learnings.append(f"- Watch for: {pattern['text']}")
            elif pattern["type"] == "improvement_suggestion":
                learnings.append(f"- Consider: {pattern['text']}")
            elif pattern["type"] == "structured_response":
                learnings.append(f"- {pattern['text']}")
            elif pattern["type"] == "includes_code":
                learnings.append(f"- {pattern['text']}")

        if not learnings:
            return current_prompt

        learnings_section = "\n\nLearned patterns from successful debates:\n" + "\n".join(learnings)

        return current_prompt + learnings_section

    def _evolve_replace(self, current_prompt: str, patterns: list[dict]) -> str:
        """Replace sections of the prompt with improved versions."""
        # Simple replacement: update the learnings section if it exists
        if "Learned patterns from successful debates:" in current_prompt:
            # Remove old learnings section
            parts = current_prompt.split("Learned patterns from successful debates:")
            current_prompt = parts[0].strip()

        # Add new learnings
        return self._evolve_append(current_prompt, patterns)

    def _evolve_refine(self, current_prompt: str, patterns: list[dict]) -> str:
        """
        Use LLM to refine the prompt (placeholder for future implementation).

        This would call an LLM to synthesize the prompt and patterns
        into a more coherent, refined prompt.
        """
        # For now, fall back to append
        # TODO: Implement LLM-based refinement
        return self._evolve_append(current_prompt, patterns)

    def apply_evolution(self, agent: Agent, patterns: list[dict] = None) -> str:
        """
        Apply evolution to an agent and save the new version.

        Returns the new prompt.
        """
        new_prompt = self.evolve_prompt(agent, patterns)

        # Save the new version
        version = self.save_prompt_version(
            agent_name=agent.name,
            prompt=new_prompt,
            metadata={
                "strategy": self.strategy.value,
                "patterns_count": len(patterns) if patterns else 0,
                "previous_prompt_length": len(agent.system_prompt or ""),
                "new_prompt_length": len(new_prompt),
            },
        )

        # Update the agent
        agent.set_system_prompt(new_prompt)

        # Record evolution history
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evolution_history (agent_name, from_version, to_version, strategy, patterns_applied)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                agent.name,
                version - 1 if version > 1 else None,
                version,
                self.strategy.value,
                json.dumps([p["text"] for p in (patterns or [])[:5]]),
            ),
        )
        conn.commit()
        conn.close()

        return new_prompt

    def get_evolution_history(self, agent_name: str, limit: int = 10) -> list[dict]:
        """Get evolution history for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT from_version, to_version, strategy, patterns_applied, created_at
            FROM evolution_history
            WHERE agent_name = ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (agent_name, limit),
        )

        history = [
            {
                "from_version": row[0],
                "to_version": row[1],
                "strategy": row[2],
                "patterns": json.loads(row[3]) if row[3] else [],
                "created_at": row[4],
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return history

    def update_performance(
        self,
        agent_name: str,
        version: int,
        debate_result: DebateResult,
    ):
        """Update performance metrics for a prompt version."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute(
            """
            SELECT debates_count, consensus_rate
            FROM prompt_versions
            WHERE agent_name = ? AND version = ?
        """,
            (agent_name, version),
        )
        row = cursor.fetchone()

        if row:
            current_count = row[0]
            current_rate = row[1]

            new_count = current_count + 1
            # Running average of consensus rate
            new_rate = (current_rate * current_count + (1 if debate_result.consensus_reached else 0)) / new_count
            new_score = debate_result.confidence if debate_result.consensus_reached else 0

            cursor.execute(
                """
                UPDATE prompt_versions
                SET debates_count = ?, consensus_rate = ?, performance_score = ?
                WHERE agent_name = ? AND version = ?
            """,
                (new_count, new_rate, new_score, agent_name, version),
            )

            conn.commit()

        conn.close()
