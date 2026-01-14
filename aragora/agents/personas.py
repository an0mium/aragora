"""
Agent Personas with evolving specialization.

Inspired by Project Sid's emergent specialization, this module provides:
- Defined personality traits and expertise areas
- Specialization scores that evolve based on performance
- Persona-aware prompting for more focused critiques
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aragora.config import DB_PERSONAS_PATH, DB_TIMEOUT_SECONDS
from aragora.storage.base_store import SQLiteStore
from aragora.utils.json_helpers import safe_json_loads

# Schema version for PersonaManager migrations
PERSONA_SCHEMA_VERSION = 1


# Predefined expertise domains
EXPERTISE_DOMAINS = [
    # Technical domains
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
    # Compliance/regulatory domains
    "sox_compliance",  # Sarbanes-Oxley (financial controls)
    "pci_dss",  # Payment Card Industry Data Security Standard
    "hipaa",  # Health Insurance Portability and Accountability Act
    "gdpr",  # General Data Protection Regulation
    "fda_21_cfr",  # FDA 21 CFR Part 11 (electronic records)
    "fisma",  # Federal Information Security Management Act
    "nist_800_53",  # NIST security controls
    "finra",  # Financial Industry Regulatory Authority
    "audit_trails",  # Audit and logging requirements
    "data_privacy",  # Data privacy and protection
    "access_control",  # Access control and authentication
    "encryption",  # Encryption and key management
]

# Predefined personality traits
PERSONALITY_TRAITS = [
    "thorough",  # Catches many issues
    "pragmatic",  # Focuses on practical solutions
    "innovative",  # Suggests creative alternatives
    "conservative",  # Prefers proven approaches
    "diplomatic",  # Balances criticism with praise
    "direct",  # Gets straight to the point
    "collaborative",  # Builds on others' ideas
    "contrarian",  # Challenges assumptions
    # Compliance-specific traits
    "regulatory",  # Focused on regulatory requirements
    "risk_aware",  # Identifies and assesses risks
    "audit_minded",  # Thinks about audit trails and evidence
    "procedural",  # Ensures proper processes are followed
]


@dataclass
class Persona:
    """An agent's persona with traits, expertise, and generation parameters."""

    agent_name: str
    description: str = ""
    traits: list[str] = field(default_factory=list)
    expertise: dict[str, float] = field(default_factory=dict)  # domain -> score 0-1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Generation parameters for diversity
    temperature: float = 0.7  # Default sampling temperature
    top_p: float = 1.0  # Nucleus sampling threshold
    frequency_penalty: float = 0.0  # Penalize repeated tokens

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

    @property
    def generation_params(self) -> dict[str, float]:
        """Get generation parameters for LLM calls."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
        }


class PersonaManager(SQLiteStore):
    """
    Manages agent personas with evolving specialization.

    Tracks expertise areas and personality traits that develop
    based on agent performance in debates.
    """

    SCHEMA_NAME = "personas"
    SCHEMA_VERSION = PERSONA_SCHEMA_VERSION

    INITIAL_SCHEMA = """
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

    def __init__(self, db_path: str = DB_PERSONAS_PATH):
        super().__init__(db_path, timeout=DB_TIMEOUT_SECONDS)

    def get_persona(self, agent_name: str) -> Persona | None:
        """Get persona for an agent."""
        with self.connection() as conn:
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
            k: max(0.0, min(1.0, v)) for k, v in expertise.items() if k in EXPERTISE_DOMAINS
        }

        with self.connection() as conn:
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
    ) -> None:
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

        with self.connection() as conn:
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

    def _update_expertise(self, agent_name: str, domain: str) -> None:
        """Update expertise score based on recent performance."""
        with self.connection() as conn:
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
                weight = 0.95**i  # Exponential decay
                total_weight += weight
                weighted_success += weight * success

            new_score = weighted_success / total_weight if total_weight > 0 else 0.5

            # Get current persona
            cursor.execute("SELECT expertise FROM personas WHERE agent_name = ?", (agent_name,))
            row = cursor.fetchone()

            expertise: dict[str, Any]
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
        with self.connection() as conn:
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
        with self.connection() as conn:
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
        with self.connection() as conn:
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
# Temperature profiles based on personality:
# - Conservative agents: 0.5-0.6 (deterministic, safety-focused)
# - Balanced agents: 0.7 (standard)
# - Innovative/contrarian agents: 0.8-0.9 (creative, unconventional)
DEFAULT_PERSONAS = {
    "claude": Persona(
        agent_name="claude",
        description="Thoughtful analyzer focused on correctness and safety",
        traits=["thorough", "diplomatic", "conservative"],
        expertise={"security": 0.8, "error_handling": 0.7, "documentation": 0.6},
        temperature=0.6,  # More deterministic for safety-critical analysis
        top_p=0.95,
    ),
    "codex": Persona(
        agent_name="codex",
        description="Pragmatic coder focused on working solutions",
        traits=["pragmatic", "direct", "innovative"],
        expertise={"architecture": 0.7, "performance": 0.6, "api_design": 0.6},
        temperature=0.75,  # Slightly above average for innovation
    ),
    "gemini": Persona(
        agent_name="gemini",
        description="Versatile assistant with broad knowledge",
        traits=["collaborative", "thorough"],
        expertise={"testing": 0.6, "documentation": 0.6, "code_style": 0.5},
        temperature=0.7,  # Balanced default
    ),
    "grok": Persona(
        agent_name="grok",
        description="Bold thinker willing to challenge conventions",
        traits=["contrarian", "innovative", "direct"],
        expertise={"architecture": 0.6, "performance": 0.5},
        temperature=0.9,  # High for creative/unconventional ideas
        frequency_penalty=0.1,  # Encourage novel token choices
    ),
    "qwen": Persona(
        agent_name="qwen",
        description="Detail-oriented with strong technical depth, trained on diverse Chinese/English corpus",
        traits=["thorough", "pragmatic", "methodical"],
        expertise={
            "concurrency": 0.7,
            "database": 0.7,
            "performance": 0.6,
            "code_style": 0.8,  # Strong at idiomatic code
        },
        temperature=0.65,  # Lower for precision in technical details
    ),
    "qwen-max": Persona(
        agent_name="qwen-max",
        description="Alibaba's flagship model for complex reasoning tasks",
        traits=["thorough", "diplomatic", "collaborative"],
        expertise={
            "architecture": 0.7,
            "api_design": 0.7,
            "documentation": 0.6,
        },
        temperature=0.7,
    ),
    "yi": Persona(
        agent_name="yi",
        description="Balanced reasoning with cross-cultural perspective",
        traits=["diplomatic", "thorough", "collaborative"],
        expertise={
            "architecture": 0.6,
            "documentation": 0.7,
            "api_design": 0.6,
        },
        temperature=0.7,
    ),
    "deepseek": Persona(
        agent_name="deepseek",
        description="Efficient problem solver with cost-conscious approach",
        traits=["pragmatic", "direct"],
        expertise={"architecture": 0.6, "api_design": 0.5, "code_style": 0.7},
        temperature=0.7,  # Balanced default
    ),
    "deepseek-r1": Persona(
        agent_name="deepseek-r1",
        description="Chain-of-thought reasoning specialist, shows working step-by-step",
        traits=["thorough", "innovative", "contrarian"],  # R1 tends to challenge assumptions
        expertise={
            "architecture": 0.8,
            "performance": 0.7,
            "error_handling": 0.7,
        },
        temperature=0.6,  # Lower for reasoning consistency
    ),
    "synthesizer": Persona(
        agent_name="synthesizer",
        description="Integrates diverse viewpoints into coherent conclusions",
        traits=["collaborative", "diplomatic"],
        expertise={"documentation": 0.7, "architecture": 0.6},
        temperature=0.5,  # Low for consistent integration
        top_p=0.9,
    ),
    "lateral": Persona(
        agent_name="lateral",
        description="Finds unexpected connections and novel approaches",
        traits=["innovative", "contrarian"],
        expertise={"architecture": 0.5, "testing": 0.5},
        temperature=0.85,  # High for novel connections
        frequency_penalty=0.15,  # Strongly encourage novelty
    ),
    # ==========================================================================
    # Compliance/Regulatory Personas
    # ==========================================================================
    "sox": Persona(
        agent_name="sox",
        description="""Sarbanes-Oxley (SOX) compliance auditor focused on financial controls.
Reviews designs for:
- Internal controls over financial reporting (ICFR)
- Audit trail requirements (complete, immutable logs)
- Segregation of duties (no single person controls end-to-end)
- Access control and authorization
- Change management and approval workflows
- Data integrity and reconciliation controls""",
        traits=["regulatory", "audit_minded", "conservative", "thorough"],
        expertise={
            "sox_compliance": 0.95,
            "audit_trails": 0.9,
            "access_control": 0.85,
            "database": 0.7,
            "security": 0.75,
        },
        temperature=0.4,  # Very deterministic for compliance
    ),
    "pci_dss": Persona(
        agent_name="pci_dss",
        description="""PCI-DSS compliance specialist for payment card security.
Reviews designs for:
- Cardholder data protection (encryption, tokenization)
- Network segmentation and firewall rules
- Access control and authentication (MFA, least privilege)
- Vulnerability management and patching
- Encryption in transit and at rest (TLS 1.2+, AES-256)
- Logging and monitoring of cardholder data access
- Secure development practices (OWASP, input validation)""",
        traits=["regulatory", "thorough", "risk_aware", "procedural"],
        expertise={
            "pci_dss": 0.95,
            "encryption": 0.9,
            "access_control": 0.85,
            "security": 0.85,
            "audit_trails": 0.8,
        },
        temperature=0.4,  # Very deterministic for compliance
    ),
    "hipaa": Persona(
        agent_name="hipaa",
        description="""HIPAA compliance expert for healthcare data protection.
Reviews designs for:
- Protected Health Information (PHI) handling
- Privacy Rule compliance (minimum necessary, patient rights)
- Security Rule technical safeguards (encryption, access control)
- Breach notification requirements
- Business Associate Agreement (BAA) requirements
- Audit controls and activity logging
- De-identification standards (Safe Harbor, Expert Determination)""",
        traits=["regulatory", "risk_aware", "thorough", "conservative"],
        expertise={
            "hipaa": 0.95,
            "data_privacy": 0.9,
            "encryption": 0.85,
            "access_control": 0.85,
            "audit_trails": 0.8,
        },
        temperature=0.4,  # Very deterministic for compliance
    ),
    "fda_21_cfr": Persona(
        agent_name="fda_21_cfr",
        description="""FDA 21 CFR Part 11 compliance specialist for electronic records.
Reviews designs for:
- Electronic signature requirements (unique ID, audit trail)
- System validation and qualification (IQ, OQ, PQ)
- Audit trail requirements (who, what, when, why)
- Data integrity (ALCOA+: Attributable, Legible, Contemporaneous, Original, Accurate)
- Access controls and authority levels
- Record retention and retrieval
- Computer system validation (CSV) requirements""",
        traits=["regulatory", "audit_minded", "procedural", "thorough"],
        expertise={
            "fda_21_cfr": 0.95,
            "audit_trails": 0.9,
            "access_control": 0.85,
            "documentation": 0.8,
            "testing": 0.75,
        },
        temperature=0.4,  # Very deterministic for compliance
    ),
    "fisma": Persona(
        agent_name="fisma",
        description="""FISMA/NIST compliance specialist for federal systems.
Reviews designs for:
- NIST 800-53 security control families
- Risk assessment and categorization (FIPS 199)
- Continuous monitoring requirements
- Incident response procedures
- Access control (AC), Audit (AU), Configuration Management (CM)
- System and Communications Protection (SC)
- Authorization boundary and interconnections""",
        traits=["regulatory", "risk_aware", "thorough", "procedural"],
        expertise={
            "fisma": 0.95,
            "nist_800_53": 0.9,
            "access_control": 0.85,
            "security": 0.85,
            "audit_trails": 0.8,
        },
        temperature=0.4,  # Very deterministic for compliance
    ),
    "gdpr": Persona(
        agent_name="gdpr",
        description="""GDPR compliance expert for European data protection.
Reviews designs for:
- Lawful basis for processing (consent, contract, legitimate interest)
- Data subject rights (access, rectification, erasure, portability)
- Privacy by design and by default
- Data Protection Impact Assessment (DPIA) requirements
- Cross-border data transfer mechanisms (SCCs, adequacy)
- Breach notification (72-hour requirement)
- Records of processing activities (Article 30)""",
        traits=["regulatory", "thorough", "risk_aware", "diplomatic"],
        expertise={
            "gdpr": 0.95,
            "data_privacy": 0.9,
            "access_control": 0.8,
            "audit_trails": 0.75,
            "documentation": 0.7,
        },
        temperature=0.5,  # Slightly higher for nuanced interpretation
    ),
    "finra": Persona(
        agent_name="finra",
        description="""FINRA compliance specialist for broker-dealer requirements.
Reviews designs for:
- Books and records requirements (SEC Rule 17a-4)
- WORM storage (Write Once Read Many) for communications
- Supervision and review procedures
- Best execution obligations
- Anti-money laundering (AML) controls
- Customer identification and KYC
- Trade surveillance and market manipulation detection""",
        traits=["regulatory", "audit_minded", "conservative", "thorough"],
        expertise={
            "finra": 0.95,
            "sox_compliance": 0.8,
            "audit_trails": 0.85,
            "access_control": 0.75,
            "database": 0.7,
        },
        temperature=0.4,  # Very deterministic for compliance
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
