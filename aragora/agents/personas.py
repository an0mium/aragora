"""
Agent Personas with evolving specialization.

Inspired by Project Sid's emergent specialization, this module provides:
- Defined personality traits and expertise areas
- Specialization scores that evolve based on performance
- Persona-aware prompting for more focused critiques
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

from aragora.config import DB_PERSONAS_PATH, DB_TIMEOUT_SECONDS
from aragora.insights.database import InsightsDatabase
from aragora.storage.schema import SchemaManager
from aragora.utils.json_helpers import safe_json_loads

# Schema version for PersonaManager migrations
PERSONA_SCHEMA_VERSION = 1


# Predefined expertise domains
EXPERTISE_DOMAINS = [
    "security",
    "performance",
    "architecture",
    "testing",
    "error_handling",
    "concurrency",
    "api_design",
    "database",
    "frontend",
    "devops",
    "documentation",
    "code_style",
]

# Predefined personality traits
PERSONALITY_TRAITS = [
    "thorough",      # Catches many issues
    "pragmatic",     # Focuses on practical solutions
    "innovative",    # Suggests creative alternatives
    "conservative",  # Prefers proven approaches
    "diplomatic",    # Balances criticism with praise
    "direct",        # Gets straight to the point
    "collaborative", # Builds on others' ideas
    "contrarian",    # Challenges assumptions
]


@dataclass
class Persona:
    """An agent's persona with traits and expertise."""

    agent_name: str
    description: str = ""
    traits: list[str] = field(default_factory=list)
    expertise: dict[str, float] = field(default_factory=dict)  # domain -> score 0-1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def top_expertise(self) -> list[tuple[str, float]]:
        """Get top 3 expertise areas."""
        sorted_exp = sorted(self.expertise.items(), key=lambda x: x[1], reverse=True)
        return sorted_exp[:3]

    @property
    def trait_string(self) -> str:
        """Get traits as comma-separated string."""
        return ", ".join(self.traits) if self.traits else "balanced"

    def to_prompt_context(self) -> str:
        """Generate prompt context from persona."""
        parts = []

        if self.description:
            parts.append(f"Your role: {self.description}")

        if self.traits:
            parts.append(f"Your approach: {self.trait_string}")

        if self.expertise:
            top = self.top_expertise
            if top:
                exp_str = ", ".join([f"{domain} ({score:.0%})" for domain, score in top])
                parts.append(f"Your expertise areas: {exp_str}")

        return "\n".join(parts) if parts else ""


class PersonaManager:
    """
    Manages agent personas with evolving specialization.

    Tracks expertise areas and personality traits that develop
    based on agent performance in debates.
    """

    def __init__(self, db_path: str = DB_PERSONAS_PATH):
        self.db_path = Path(db_path)
        self.db = InsightsDatabase(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with guaranteed cleanup."""
        with self.db.connection() as conn:
            yield conn

    def _init_db(self):
        """Initialize database schema using SchemaManager."""
        with self._get_connection() as conn:
            # Use SchemaManager for version tracking and migrations
            manager = SchemaManager(
                conn, "personas", current_version=PERSONA_SCHEMA_VERSION
            )

            # Initial schema (v1)
            initial_schema = """
                -- Personas table
                CREATE TABLE IF NOT EXISTS personas (
                    agent_name TEXT PRIMARY KEY,
                    description TEXT,
                    traits TEXT,
                    expertise TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Performance history for learning
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    debate_id TEXT,
                    domain TEXT,
                    action TEXT,
                    success INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """

            manager.ensure_schema(initial_schema=initial_schema)

    def get_persona(self, agent_name: str) -> Persona | None:
        """Get persona for an agent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT agent_name, description, traits, expertise, created_at, updated_at FROM personas WHERE agent_name = ?",
                (agent_name,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return Persona(
                agent_name=row[0],
                description=row[1] or "",
                traits=safe_json_loads(row[2], []),
                expertise=safe_json_loads(row[3], {}),
                created_at=row[4],
                updated_at=row[5],
            )

    def create_persona(
        self,
        agent_name: str,
        description: str = "",
        traits: list[str] | None = None,
        expertise: dict[str, float] | None = None,
    ) -> Persona:
        """Create or update a persona for an agent."""
        now = datetime.now().isoformat()
        traits = traits or []
        expertise = expertise or {}

        # Validate traits
        traits = [t for t in traits if t in PERSONALITY_TRAITS]

        # Validate and normalize expertise
        expertise = {
            k: max(0.0, min(1.0, v))
            for k, v in expertise.items()
            if k in EXPERTISE_DOMAINS
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO personas (agent_name, description, traits, expertise, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_name) DO UPDATE SET
                    description = excluded.description,
                    traits = excluded.traits,
                    expertise = excluded.expertise,
                    updated_at = excluded.updated_at
                """,
                (agent_name, description, json.dumps(traits), json.dumps(expertise), now, now),
            )

            conn.commit()

        return Persona(
            agent_name=agent_name,
            description=description,
            traits=traits,
            expertise=expertise,
            created_at=now,
            updated_at=now,
        )

    def record_performance(
        self,
        agent_name: str,
        domain: str,
        success: bool,
        action: str = "critique",
        debate_id: str | None = None,
    ):
        """
        Record a performance event to update expertise.

        Args:
            agent_name: Name of the agent
            domain: Expertise domain (e.g., "security", "performance")
            success: Whether the action was successful
            action: Type of action (critique, proposal, etc.)
            debate_id: Optional debate ID
        """
        if domain not in EXPERTISE_DOMAINS:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO performance_history (agent_name, debate_id, domain, action, success)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_name, debate_id, domain, action, 1 if success else 0),
            )

            conn.commit()

        # Update expertise based on performance
        self._update_expertise(agent_name, domain)

    def _update_expertise(self, agent_name: str, domain: str):
        """Update expertise score based on recent performance."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get recent performance in this domain (last 50 events)
            cursor.execute(
                """
                SELECT success FROM performance_history
                WHERE agent_name = ? AND domain = ?
                ORDER BY created_at DESC
                LIMIT 50
                """,
                (agent_name, domain),
            )
            rows = cursor.fetchall()

            if not rows:
                return

            # Calculate success rate with recency weighting
            total_weight = 0.0
            weighted_success = 0.0
            for i, (success,) in enumerate(rows):
                weight = 0.95 ** i  # Exponential decay
                total_weight += weight
                weighted_success += weight * success

            new_score = weighted_success / total_weight if total_weight > 0 else 0.5

            # Get current persona
            cursor.execute("SELECT expertise FROM personas WHERE agent_name = ?", (agent_name,))
            row = cursor.fetchone()

            if row:
                expertise = safe_json_loads(row[0], {})
            else:
                expertise = {}

            # Smooth update (blend old and new)
            old_score = expertise.get(domain, 0.5)
            expertise[domain] = 0.7 * new_score + 0.3 * old_score

            # Update persona
            cursor.execute(
                """
                INSERT INTO personas (agent_name, expertise, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_name) DO UPDATE SET
                    expertise = excluded.expertise,
                    updated_at = excluded.updated_at
                """,
                (agent_name, json.dumps(expertise), datetime.now().isoformat()),
            )

            conn.commit()

    def infer_traits(self, agent_name: str) -> list[str]:
        """
        Infer personality traits from performance patterns.

        Returns suggested traits based on observed behavior.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get performance stats
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successes,
                    COUNT(DISTINCT domain) as domains_covered
                FROM performance_history
                WHERE agent_name = ?
                """,
                (agent_name,),
            )
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return []

            total, successes, domains = row
            success_rate = successes / total if total > 0 else 0

            traits = []

            # Infer traits from patterns
            if domains >= 5:
                traits.append("thorough")  # Covers many domains

            if success_rate > 0.7:
                traits.append("pragmatic")  # High success rate

            if domains <= 2 and total >= 10:
                traits.append("conservative")  # Focuses on few areas

            return traits

    def get_all_personas(self) -> list[Persona]:
        """Get all personas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT agent_name, description, traits, expertise, created_at, updated_at FROM personas"
            )
            rows = cursor.fetchall()

            return [
                Persona(
                    agent_name=row[0],
                    description=row[1] or "",
                    traits=safe_json_loads(row[2], []),
                    expertise=safe_json_loads(row[3], {}),
                    created_at=row[4],
                    updated_at=row[5],
                )
                for row in rows
            ]

    def get_performance_summary(self, agent_name: str) -> dict:
        """Get performance summary for an agent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    domain,
                    COUNT(*) as total,
                    SUM(success) as successes
                FROM performance_history
                WHERE agent_name = ?
                GROUP BY domain
                ORDER BY total DESC
                """,
                (agent_name,),
            )
            rows = cursor.fetchall()

            return {
                row[0]: {
                    "total": row[1],
                    "successes": row[2],
                    "rate": row[2] / row[1] if row[1] > 0 else 0,
                }
                for row in rows
            }


# Default personas for common agent types
DEFAULT_PERSONAS = {
    "claude": Persona(
        agent_name="claude",
        description="Thoughtful analyzer focused on correctness and safety",
        traits=["thorough", "diplomatic", "conservative"],
        expertise={"security": 0.8, "error_handling": 0.7, "documentation": 0.6},
    ),
    "codex": Persona(
        agent_name="codex",
        description="Pragmatic coder focused on working solutions",
        traits=["pragmatic", "direct", "innovative"],
        expertise={"architecture": 0.7, "performance": 0.6, "api_design": 0.6},
    ),
    "gemini": Persona(
        agent_name="gemini",
        description="Versatile assistant with broad knowledge",
        traits=["collaborative", "thorough"],
        expertise={"testing": 0.6, "documentation": 0.6, "code_style": 0.5},
    ),
    "grok": Persona(
        agent_name="grok",
        description="Bold thinker willing to challenge conventions",
        traits=["contrarian", "innovative", "direct"],
        expertise={"architecture": 0.6, "performance": 0.5},
    ),
    "qwen": Persona(
        agent_name="qwen",
        description="Detail-oriented with strong technical depth",
        traits=["thorough", "pragmatic"],
        expertise={"concurrency": 0.6, "database": 0.6, "performance": 0.5},
    ),
    "deepseek": Persona(
        agent_name="deepseek",
        description="Efficient problem solver with cost-conscious approach",
        traits=["pragmatic", "direct"],
        expertise={"architecture": 0.6, "api_design": 0.5},
    ),
}


def get_or_create_persona(manager: PersonaManager, agent_name: str) -> Persona:
    """Get existing persona or create from defaults."""
    persona = manager.get_persona(agent_name)

    if persona:
        return persona

    # Check for default
    base_name = agent_name.split("_")[0].lower()  # e.g., "claude_critic" -> "claude"
    if base_name in DEFAULT_PERSONAS:
        default = DEFAULT_PERSONAS[base_name]
        return manager.create_persona(
            agent_name=agent_name,
            description=default.description,
            traits=default.traits.copy(),
            expertise=default.expertise.copy(),
        )

    # Create empty persona
    return manager.create_persona(agent_name=agent_name)
