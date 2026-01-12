"""
Tests for aragora/runtime/metadata.py - Debate reproducibility metadata.

This module tracks configuration for reproducibility:
- Model identifiers and versions
- Prompt hashes
- Sampling parameters
- Environment info
"""

import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aragora.runtime.metadata import (
    DebateMetadata,
    MetadataStore,
    ModelConfig,
)


class TestModelConfig(unittest.TestCase):
    """Tests for ModelConfig dataclass."""

    def test_basic_creation(self):
        """Should create config with required fields."""
        config = ModelConfig(model_id="gpt-4", provider="openai")
        self.assertEqual(config.model_id, "gpt-4")
        self.assertEqual(config.provider, "openai")

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ModelConfig(model_id="test", provider="test")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.context_window, 8192)

    def test_system_prompt_hash_defaults_to_empty(self):
        """system_prompt_hash should default to empty string."""
        config = ModelConfig(model_id="test", provider="test")
        self.assertEqual(config.system_prompt_hash, "")

    def test_custom_values(self):
        """Should accept custom values."""
        config = ModelConfig(
            model_id="claude-3-opus",
            provider="anthropic",
            version="2024-01",
            temperature=0.5,
            max_tokens=8192,
        )
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 8192)
        self.assertEqual(config.version, "2024-01")


class TestModelConfigProviderInference(unittest.TestCase):
    """Tests for provider inference from model name."""

    def test_infer_openai_from_gpt(self):
        """Should infer openai from GPT models."""
        self.assertEqual(ModelConfig._infer_provider("gpt-4"), "openai")
        self.assertEqual(ModelConfig._infer_provider("gpt-3.5-turbo"), "openai")
        self.assertEqual(ModelConfig._infer_provider("GPT-4-turbo"), "openai")

    def test_infer_anthropic_from_claude(self):
        """Should infer anthropic from Claude models."""
        self.assertEqual(ModelConfig._infer_provider("claude-3-opus"), "anthropic")
        self.assertEqual(ModelConfig._infer_provider("claude-2.1"), "anthropic")
        self.assertEqual(ModelConfig._infer_provider("Claude-3-Sonnet"), "anthropic")

    def test_infer_google_from_gemini(self):
        """Should infer google from Gemini models."""
        self.assertEqual(ModelConfig._infer_provider("gemini-pro"), "google")
        self.assertEqual(ModelConfig._infer_provider("Gemini-1.5-Flash"), "google")

    def test_infer_codex(self):
        """Should identify Codex as openai-codex."""
        self.assertEqual(ModelConfig._infer_provider("codex"), "openai-codex")
        self.assertEqual(ModelConfig._infer_provider("codex-davinci"), "openai-codex")

    def test_infer_local_models(self):
        """Should identify local models."""
        self.assertEqual(ModelConfig._infer_provider("llama-2-70b"), "local")
        self.assertEqual(ModelConfig._infer_provider("mistral-7b"), "local")

    def test_infer_unknown(self):
        """Should return unknown for unrecognized models."""
        self.assertEqual(ModelConfig._infer_provider("random-model"), "unknown")
        self.assertEqual(ModelConfig._infer_provider("custom-llm-v1"), "unknown")


class TestModelConfigFromAgent(unittest.TestCase):
    """Tests for creating ModelConfig from Agent instances."""

    def test_from_agent_basic(self):
        """Should extract model from agent."""
        agent = MagicMock()
        agent.model = "gpt-4-turbo"
        agent.system_prompt = None

        config = ModelConfig.from_agent(agent)
        self.assertEqual(config.model_id, "gpt-4-turbo")
        self.assertEqual(config.provider, "openai")

    def test_from_agent_with_system_prompt(self):
        """Should hash system prompt."""
        agent = MagicMock()
        agent.model = "claude-3-opus"
        agent.system_prompt = "You are a helpful assistant."

        config = ModelConfig.from_agent(agent)

        # Hash should be first 16 chars of SHA-256
        expected_hash = hashlib.sha256(
            "You are a helpful assistant.".encode()
        ).hexdigest()[:16]
        self.assertEqual(config.system_prompt_hash, expected_hash)

    def test_from_agent_no_system_prompt_attr(self):
        """Should handle agents without system_prompt attr."""
        agent = MagicMock(spec=[])  # No system_prompt attr
        agent.model = "gpt-4"

        config = ModelConfig.from_agent(agent)
        self.assertEqual(config.system_prompt_hash, "")


class TestModelConfigSerialization(unittest.TestCase):
    """Tests for ModelConfig serialization."""

    def test_to_dict(self):
        """Should serialize all fields to dict."""
        config = ModelConfig(
            model_id="gpt-4",
            provider="openai",
            version="0613",
            temperature=0.5,
        )

        d = config.to_dict()
        self.assertEqual(d["model_id"], "gpt-4")
        self.assertEqual(d["provider"], "openai")
        self.assertEqual(d["version"], "0613")
        self.assertEqual(d["temperature"], 0.5)
        self.assertIn("max_tokens", d)
        self.assertIn("context_window", d)


class TestDebateMetadata(unittest.TestCase):
    """Tests for DebateMetadata dataclass."""

    def test_basic_creation(self):
        """Should create metadata with required fields."""
        meta = DebateMetadata(debate_id="debate-123")
        self.assertEqual(meta.debate_id, "debate-123")
        self.assertIsNotNone(meta.created_at)

    def test_default_protocol(self):
        """Should have default protocol settings."""
        meta = DebateMetadata(debate_id="test")
        self.assertEqual(meta.protocol_type, "standard")
        self.assertEqual(meta.max_rounds, 3)
        self.assertEqual(meta.consensus_method, "majority")
        self.assertEqual(meta.consensus_threshold, 0.7)

    def test_environment_defaults(self):
        """Should capture environment info."""
        meta = DebateMetadata(debate_id="test")
        self.assertEqual(meta.aragora_version, "0.8.0")
        self.assertIn(".", meta.python_version)  # e.g., "3.10.13"
        self.assertTrue(len(meta.platform_info) > 0)


class TestDebateMetadataHashing(unittest.TestCase):
    """Tests for task and config hash generation."""

    def test_task_hash_generated(self):
        """Should generate task hash on creation."""
        meta = DebateMetadata(
            debate_id="test",
            task="Implement a rate limiter"
        )
        self.assertTrue(len(meta.task_hash) == 16)  # First 16 chars of SHA-256

    def test_task_hash_consistent(self):
        """Same task should produce same hash."""
        meta1 = DebateMetadata(debate_id="1", task="Test task")
        meta2 = DebateMetadata(debate_id="2", task="Test task")
        self.assertEqual(meta1.task_hash, meta2.task_hash)

    def test_task_hash_different_for_different_tasks(self):
        """Different tasks should produce different hashes."""
        meta1 = DebateMetadata(debate_id="1", task="Task A")
        meta2 = DebateMetadata(debate_id="2", task="Task B")
        self.assertNotEqual(meta1.task_hash, meta2.task_hash)

    def test_context_hash_generated(self):
        """Should generate context hash."""
        meta = DebateMetadata(
            debate_id="test",
            context="Some additional context"
        )
        self.assertTrue(len(meta.context_hash) == 16)

    def test_config_hash_includes_agents(self):
        """Config hash should include agent configurations."""
        agents = [
            ModelConfig(model_id="gpt-4", provider="openai"),
            ModelConfig(model_id="claude-3", provider="anthropic"),
        ]
        meta = DebateMetadata(
            debate_id="test",
            task="Test",
            agent_configs=agents,
        )
        self.assertTrue(len(meta.config_hash) == 16)

    def test_config_hash_different_for_different_agents(self):
        """Different agent configs should produce different config hashes."""
        meta1 = DebateMetadata(
            debate_id="1",
            task="Test",
            agent_configs=[ModelConfig(model_id="gpt-4", provider="openai")],
        )
        meta2 = DebateMetadata(
            debate_id="2",
            task="Test",
            agent_configs=[ModelConfig(model_id="claude-3", provider="anthropic")],
        )
        self.assertNotEqual(meta1.config_hash, meta2.config_hash)


class TestDebateMetadataSerialization(unittest.TestCase):
    """Tests for DebateMetadata serialization."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        agents = [ModelConfig(model_id="gpt-4", provider="openai")]
        meta = DebateMetadata(
            debate_id="debate-123",
            task="Test task",
            agent_configs=agents,
            tags=["test", "unit"],
        )

        d = meta.to_dict()
        self.assertEqual(d["debate_id"], "debate-123")
        self.assertEqual(d["task"], "Test task")
        self.assertEqual(len(d["agents"]), 1)
        self.assertEqual(d["tags"], ["test", "unit"])
        self.assertIn("protocol", d)
        self.assertIn("environment", d)

    def test_to_json(self):
        """Should serialize to JSON string."""
        meta = DebateMetadata(debate_id="test", task="Test")
        json_str = meta.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["debate_id"], "test")

    def test_context_truncated_in_dict(self):
        """Long context should be truncated in to_dict."""
        long_context = "x" * 1000
        meta = DebateMetadata(debate_id="test", context=long_context)

        d = meta.to_dict()
        self.assertEqual(len(d["context"]), 500)  # Truncated to 500

    def test_from_dict_roundtrip(self):
        """Should deserialize back to equivalent object."""
        agents = [ModelConfig(model_id="gpt-4", provider="openai", temperature=0.5)]
        original = DebateMetadata(
            debate_id="debate-123",
            task="Test task",
            max_rounds=5,
            consensus_method="unanimous",
            agent_configs=agents,
            random_seed=42,
            tags=["test"],
        )

        d = original.to_dict()
        restored = DebateMetadata.from_dict(d)

        self.assertEqual(restored.debate_id, original.debate_id)
        self.assertEqual(restored.task, original.task)
        self.assertEqual(restored.max_rounds, original.max_rounds)
        self.assertEqual(restored.consensus_method, original.consensus_method)
        self.assertEqual(restored.random_seed, original.random_seed)
        self.assertEqual(restored.tags, original.tags)
        self.assertEqual(len(restored.agent_configs), 1)
        self.assertEqual(restored.agent_configs[0].temperature, 0.5)


class TestDebateMetadataComparison(unittest.TestCase):
    """Tests for metadata comparison and diffing."""

    def test_is_similar_config_identical(self):
        """Identical configs should be similar."""
        agents = [ModelConfig(model_id="gpt-4", provider="openai")]
        meta1 = DebateMetadata(
            debate_id="1",
            task="Test",
            max_rounds=3,
            agent_configs=agents,
        )
        meta2 = DebateMetadata(
            debate_id="2",
            task="Test",
            max_rounds=3,
            agent_configs=agents,
        )
        self.assertTrue(meta1.is_similar_config(meta2))

    def test_is_similar_config_different(self):
        """Different configs should not be similar."""
        meta1 = DebateMetadata(debate_id="1", task="Task A", max_rounds=3)
        meta2 = DebateMetadata(debate_id="2", task="Task B", max_rounds=5)
        self.assertFalse(meta1.is_similar_config(meta2))

    def test_diff_empty_when_identical(self):
        """Diff should be empty for identical configs."""
        meta1 = DebateMetadata(debate_id="1", task="Test", max_rounds=3)
        meta2 = DebateMetadata(debate_id="2", task="Test", max_rounds=3)

        diffs = meta1.diff(meta2)
        self.assertEqual(len(diffs), 0)

    def test_diff_detects_task_change(self):
        """Diff should detect task changes."""
        meta1 = DebateMetadata(debate_id="1", task="Task A")
        meta2 = DebateMetadata(debate_id="2", task="Task B")

        diffs = meta1.diff(meta2)
        self.assertIn("task", diffs)
        self.assertNotEqual(diffs["task"]["self"], diffs["task"]["other"])

    def test_diff_detects_protocol_change(self):
        """Diff should detect protocol type changes."""
        meta1 = DebateMetadata(debate_id="1", task="Test", protocol_type="standard")
        meta2 = DebateMetadata(debate_id="2", task="Test", protocol_type="graph")

        diffs = meta1.diff(meta2)
        self.assertIn("protocol_type", diffs)

    def test_diff_detects_rounds_change(self):
        """Diff should detect max_rounds changes."""
        meta1 = DebateMetadata(debate_id="1", task="Test", max_rounds=3)
        meta2 = DebateMetadata(debate_id="2", task="Test", max_rounds=5)

        diffs = meta1.diff(meta2)
        self.assertIn("max_rounds", diffs)
        self.assertEqual(diffs["max_rounds"]["self"], 3)
        self.assertEqual(diffs["max_rounds"]["other"], 5)

    def test_diff_detects_agent_count_change(self):
        """Diff should detect agent count changes."""
        meta1 = DebateMetadata(
            debate_id="1",
            task="Test",
            agent_configs=[ModelConfig(model_id="gpt-4", provider="openai")],
        )
        meta2 = DebateMetadata(
            debate_id="2",
            task="Test",
            agent_configs=[
                ModelConfig(model_id="gpt-4", provider="openai"),
                ModelConfig(model_id="claude-3", provider="anthropic"),
            ],
        )

        diffs = meta1.diff(meta2)
        self.assertIn("agent_count", diffs)
        self.assertEqual(diffs["agent_count"]["self"], 1)
        self.assertEqual(diffs["agent_count"]["other"], 2)


class TestDebateMetadataFromArena(unittest.TestCase):
    """Tests for creating metadata from Arena instances."""

    def test_from_arena(self):
        """Should extract metadata from Arena."""
        # Mock arena
        arena = MagicMock()
        arena.env.task = "Design a cache system"
        arena.env.context = "For a web application"
        arena.protocol.rounds = 4
        arena.protocol.consensus = "majority"

        agent1 = MagicMock()
        agent1.model = "gpt-4"
        agent1.system_prompt = None

        agent2 = MagicMock()
        agent2.model = "claude-3-opus"
        agent2.system_prompt = "Be helpful"

        arena.agents = [agent1, agent2]

        meta = DebateMetadata.from_arena(arena, "debate-xyz")

        self.assertEqual(meta.debate_id, "debate-xyz")
        self.assertEqual(meta.task, "Design a cache system")
        self.assertEqual(meta.context, "For a web application")
        self.assertEqual(meta.max_rounds, 4)
        self.assertEqual(len(meta.agent_configs), 2)
        self.assertEqual(meta.agent_configs[0].model_id, "gpt-4")
        self.assertEqual(meta.agent_configs[1].model_id, "claude-3-opus")


class TestMetadataStore(unittest.TestCase):
    """Tests for MetadataStore persistence."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.db")
        self.store = MetadataStore(self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        try:
            os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_store_and_get(self):
        """Should store and retrieve metadata."""
        meta = DebateMetadata(
            debate_id="debate-123",
            task="Test task",
            max_rounds=5,
        )

        self.store.store(meta)
        retrieved = self.store.get("debate-123")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.debate_id, "debate-123")
        self.assertEqual(retrieved.task, "Test task")
        self.assertEqual(retrieved.max_rounds, 5)

    def test_get_nonexistent(self):
        """Should return None for nonexistent debate."""
        result = self.store.get("nonexistent-id")
        self.assertIsNone(result)

    def test_store_replaces_existing(self):
        """Should replace existing metadata with same ID."""
        meta1 = DebateMetadata(debate_id="debate-123", task="First task")
        meta2 = DebateMetadata(debate_id="debate-123", task="Updated task")

        self.store.store(meta1)
        self.store.store(meta2)

        retrieved = self.store.get("debate-123")
        self.assertEqual(retrieved.task, "Updated task")

    def test_find_similar(self):
        """Should find debates with similar configuration."""
        agents = [ModelConfig(model_id="gpt-4", provider="openai")]

        # Store multiple debates with same config
        meta1 = DebateMetadata(
            debate_id="debate-1",
            task="Same task",
            max_rounds=3,
            agent_configs=agents,
        )
        meta2 = DebateMetadata(
            debate_id="debate-2",
            task="Same task",
            max_rounds=3,
            agent_configs=agents,
        )
        meta3 = DebateMetadata(
            debate_id="debate-3",
            task="Different task",
            max_rounds=5,
        )

        self.store.store(meta1)
        self.store.store(meta2)
        self.store.store(meta3)

        similar = self.store.find_similar(meta1)
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].debate_id, "debate-2")

    def test_find_similar_excludes_self(self):
        """find_similar should not return the query debate itself."""
        meta = DebateMetadata(debate_id="debate-123", task="Test")
        self.store.store(meta)

        similar = self.store.find_similar(meta)
        self.assertEqual(len(similar), 0)

    def test_find_by_task(self):
        """Should find debates for the same task."""
        meta1 = DebateMetadata(debate_id="debate-1", task="Common task")
        meta2 = DebateMetadata(debate_id="debate-2", task="Common task")
        meta3 = DebateMetadata(debate_id="debate-3", task="Other task")

        self.store.store(meta1)
        self.store.store(meta2)
        self.store.store(meta3)

        # All have same task_hash
        results = self.store.find_by_task(meta1.task_hash)
        self.assertEqual(len(results), 2)

    def test_find_by_task_with_limit(self):
        """Should respect limit parameter."""
        for i in range(5):
            meta = DebateMetadata(debate_id=f"debate-{i}", task="Same task")
            self.store.store(meta)

        results = self.store.find_by_task(
            DebateMetadata(debate_id="query", task="Same task").task_hash,
            limit=3
        )
        self.assertEqual(len(results), 3)


class TestMetadataStoreEdgeCases(unittest.TestCase):
    """Edge case tests for MetadataStore."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.db")
        self.store = MetadataStore(self.db_path)

    def tearDown(self):
        try:
            os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_store_with_custom_metadata(self):
        """Should preserve custom metadata fields."""
        meta = DebateMetadata(
            debate_id="debate-123",
            task="Test",
            custom_metadata={"key": "value", "number": 42},
        )

        self.store.store(meta)
        retrieved = self.store.get("debate-123")

        self.assertEqual(retrieved.custom_metadata["key"], "value")
        self.assertEqual(retrieved.custom_metadata["number"], 42)

    def test_store_with_tags(self):
        """Should preserve tags."""
        meta = DebateMetadata(
            debate_id="debate-123",
            task="Test",
            tags=["important", "production", "v2"],
        )

        self.store.store(meta)
        retrieved = self.store.get("debate-123")

        self.assertEqual(retrieved.tags, ["important", "production", "v2"])

    def test_store_with_multiple_agents(self):
        """Should preserve all agent configurations."""
        agents = [
            ModelConfig(model_id="gpt-4", provider="openai", temperature=0.5),
            ModelConfig(model_id="claude-3", provider="anthropic", temperature=0.8),
            ModelConfig(model_id="gemini-pro", provider="google", temperature=0.6),
        ]
        meta = DebateMetadata(
            debate_id="debate-123",
            task="Test",
            agent_configs=agents,
        )

        self.store.store(meta)
        retrieved = self.store.get("debate-123")

        self.assertEqual(len(retrieved.agent_configs), 3)
        self.assertEqual(retrieved.agent_configs[0].temperature, 0.5)
        self.assertEqual(retrieved.agent_configs[1].temperature, 0.8)
        self.assertEqual(retrieved.agent_configs[2].temperature, 0.6)


class TestReproducibilityScenarios(unittest.TestCase):
    """Integration tests for reproducibility scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.db")
        self.store = MetadataStore(self.db_path)

    def tearDown(self):
        try:
            os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_find_previous_runs_of_same_experiment(self):
        """Should find previous runs with same config for comparison."""
        agents = [
            ModelConfig(model_id="gpt-4", provider="openai"),
            ModelConfig(model_id="claude-3", provider="anthropic"),
        ]

        # Run experiment multiple times
        for i in range(3):
            meta = DebateMetadata(
                debate_id=f"run-{i}",
                task="What is the best sorting algorithm?",
                max_rounds=3,
                agent_configs=agents,
            )
            self.store.store(meta)

        # Query for similar configs
        query = DebateMetadata(
            debate_id="run-3",  # New run
            task="What is the best sorting algorithm?",
            max_rounds=3,
            agent_configs=agents,
        )

        similar = self.store.find_similar(query, limit=10)
        self.assertEqual(len(similar), 3)  # All 3 previous runs

    def test_compare_configs_across_runs(self):
        """Should detect configuration drift across runs."""
        # First run with original config
        meta1 = DebateMetadata(
            debate_id="run-1",
            task="Test",
            max_rounds=3,
            agent_configs=[ModelConfig(model_id="gpt-4", provider="openai")],
        )

        # Second run with modified config
        meta2 = DebateMetadata(
            debate_id="run-2",
            task="Test",
            max_rounds=5,  # Changed
            agent_configs=[
                ModelConfig(model_id="gpt-4", provider="openai"),
                ModelConfig(model_id="claude-3", provider="anthropic"),  # Added
            ],
        )

        diffs = meta1.diff(meta2)

        # Should detect both changes
        self.assertIn("max_rounds", diffs)
        self.assertIn("agent_count", diffs)


if __name__ == "__main__":
    unittest.main()
