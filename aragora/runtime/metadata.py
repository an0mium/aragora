"""
Debate Metadata - Run configuration for reproducibility.

Stores all configuration needed to reproduce a debate:
- Model identifiers and versions
- Prompt hashes
- Sampling parameters
- Random seeds
- Environment info

Note: LLM outputs are inherently non-deterministic even with
temperature=0. "Reproducibility" here means same configuration,
not identical outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import hashlib
import json
import platform


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str                    # e.g., "gpt-4-turbo", "claude-3-opus"
    provider: str                    # e.g., "openai", "anthropic", "local"
    version: Optional[str] = None   # Model version if known

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096

    # Context settings
    system_prompt_hash: Optional[str] = None
    context_window: int = 8192

    # Cost info
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    def __post_init__(self):
        if self.system_prompt_hash is None:
            self.system_prompt_hash = ""

    @classmethod
    def from_agent(cls, agent) -> "ModelConfig":
        """Create config from an Agent instance."""
        system_hash = ""
        if hasattr(agent, "system_prompt") and agent.system_prompt:
            system_hash = hashlib.sha256(
                agent.system_prompt.encode()
            ).hexdigest()[:16]

        return cls(
            model_id=agent.model,
            provider=cls._infer_provider(agent.model),
            system_prompt_hash=system_hash,
        )

    @staticmethod
    def _infer_provider(model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "codex" in model_lower:
            return "openai-codex"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "local"
        else:
            return "unknown"

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "version": self.version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "system_prompt_hash": self.system_prompt_hash,
            "context_window": self.context_window,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
        }


@dataclass
class DebateMetadata:
    """
    Complete metadata for a debate run.

    Contains all configuration needed for reproducibility context
    (not deterministic reproduction, as LLMs are non-deterministic).
    """

    # Identity
    debate_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Task
    task: str = ""
    task_hash: str = ""  # SHA-256 of task for quick comparison
    context: str = ""
    context_hash: str = ""

    # Protocol
    protocol_type: str = "standard"  # standard, graph, tournament
    max_rounds: int = 3
    consensus_method: str = "majority"  # majority, unanimous, judge
    consensus_threshold: float = 0.7

    # Models
    agent_configs: list[ModelConfig] = field(default_factory=list)

    # Randomness
    random_seed: Optional[int] = None  # For reproducibility context

    # Environment
    aragora_version: str = "0.07"
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform_info: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    # Additional context
    tags: list[str] = field(default_factory=list)
    custom_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.task and not self.task_hash:
            self.task_hash = hashlib.sha256(self.task.encode()).hexdigest()[:16]
        if self.context and not self.context_hash:
            self.context_hash = hashlib.sha256(self.context.encode()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        """Hash of all configuration for quick comparison."""
        config = {
            "task_hash": self.task_hash,
            "protocol": self.protocol_type,
            "rounds": self.max_rounds,
            "consensus": self.consensus_method,
            "agents": [c.to_dict() for c in self.agent_configs],
        }
        return hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "created_at": self.created_at,
            "task": self.task,
            "task_hash": self.task_hash,
            "context": self.context[:500] if self.context else "",
            "context_hash": self.context_hash,
            "protocol": {
                "type": self.protocol_type,
                "max_rounds": self.max_rounds,
                "consensus_method": self.consensus_method,
                "consensus_threshold": self.consensus_threshold,
            },
            "agents": [c.to_dict() for c in self.agent_configs],
            "random_seed": self.random_seed,
            "environment": {
                "aragora_version": self.aragora_version,
                "python_version": self.python_version,
                "platform": self.platform_info,
            },
            "config_hash": self.config_hash,
            "tags": self.tags,
            "custom": self.custom_metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "DebateMetadata":
        """Deserialize from dictionary."""
        protocol = data.get("protocol", {})
        env = data.get("environment", {})

        agent_configs = []
        for ac in data.get("agents", []):
            agent_configs.append(ModelConfig(
                model_id=ac["model_id"],
                provider=ac["provider"],
                version=ac.get("version"),
                temperature=ac.get("temperature", 0.7),
                top_p=ac.get("top_p", 1.0),
                max_tokens=ac.get("max_tokens", 4096),
                system_prompt_hash=ac.get("system_prompt_hash"),
                context_window=ac.get("context_window", 8192),
            ))

        return cls(
            debate_id=data["debate_id"],
            created_at=data.get("created_at", ""),
            task=data.get("task", ""),
            task_hash=data.get("task_hash", ""),
            context=data.get("context", ""),
            context_hash=data.get("context_hash", ""),
            protocol_type=protocol.get("type", "standard"),
            max_rounds=protocol.get("max_rounds", 3),
            consensus_method=protocol.get("consensus_method", "majority"),
            consensus_threshold=protocol.get("consensus_threshold", 0.7),
            agent_configs=agent_configs,
            random_seed=data.get("random_seed"),
            aragora_version=env.get("aragora_version", "0.07"),
            python_version=env.get("python_version", ""),
            platform_info=env.get("platform", ""),
            tags=data.get("tags", []),
            custom_metadata=data.get("custom", {}),
        )

    @classmethod
    def from_arena(cls, arena, debate_id: str) -> "DebateMetadata":
        """Create metadata from an Arena instance."""
        agent_configs = []
        for agent in arena.agents:
            agent_configs.append(ModelConfig.from_agent(agent))

        return cls(
            debate_id=debate_id,
            task=arena.env.task,
            context=arena.env.context,
            protocol_type="standard",
            max_rounds=arena.protocol.rounds,
            consensus_method=arena.protocol.consensus,
            agent_configs=agent_configs,
        )

    def is_similar_config(self, other: "DebateMetadata") -> bool:
        """Check if two debates have similar configuration."""
        return self.config_hash == other.config_hash

    def diff(self, other: "DebateMetadata") -> dict[str, dict[str, Any]]:
        """Get differences between two metadata configs."""
        diffs: dict[str, dict[str, Any]] = {}

        if self.task_hash != other.task_hash:
            diffs["task"] = {"self": self.task_hash, "other": other.task_hash}

        if self.protocol_type != other.protocol_type:
            diffs["protocol_type"] = {"self": self.protocol_type, "other": other.protocol_type}

        if self.max_rounds != other.max_rounds:
            diffs["max_rounds"] = {"self": self.max_rounds, "other": other.max_rounds}

        if len(self.agent_configs) != len(other.agent_configs):
            diffs["agent_count"] = {"self": len(self.agent_configs), "other": len(other.agent_configs)}

        return diffs


class MetadataStore:
    """
    Persistent storage for debate metadata.

    Enables querying past debates by configuration.
    """

    def __init__(self, db_path: str = "aragora_metadata.db"):
        import sqlite3
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS debate_metadata (
                debate_id TEXT PRIMARY KEY,
                config_hash TEXT,
                task_hash TEXT,
                created_at TEXT,
                metadata_json TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_config_hash
            ON debate_metadata(config_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_hash
            ON debate_metadata(task_hash)
        """)

        conn.commit()
        conn.close()

    def store(self, metadata: DebateMetadata):
        """Store debate metadata."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO debate_metadata
            (debate_id, config_hash, task_hash, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metadata.debate_id,
            metadata.config_hash,
            metadata.task_hash,
            metadata.created_at,
            metadata.to_json(),
        ))

        conn.commit()
        conn.close()

    def get(self, debate_id: str) -> Optional[DebateMetadata]:
        """Retrieve metadata by debate ID."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT metadata_json FROM debate_metadata WHERE debate_id = ?",
            (debate_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return DebateMetadata.from_dict(json.loads(row[0]))
        return None

    def find_similar(self, metadata: DebateMetadata, limit: int = 10) -> list[DebateMetadata]:
        """Find debates with similar configuration."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT metadata_json FROM debate_metadata
            WHERE config_hash = ? AND debate_id != ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (metadata.config_hash, metadata.debate_id, limit))

        results = []
        for row in cursor.fetchall():
            results.append(DebateMetadata.from_dict(json.loads(row[0])))

        conn.close()
        return results

    def find_by_task(self, task_hash: str, limit: int = 10) -> list[DebateMetadata]:
        """Find debates for the same task."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT metadata_json FROM debate_metadata
            WHERE task_hash = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (task_hash, limit))

        results = []
        for row in cursor.fetchall():
            results.append(DebateMetadata.from_dict(json.loads(row[0])))

        conn.close()
        return results
