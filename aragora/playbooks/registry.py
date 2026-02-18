"""
Playbook Registry — discover, register, and load playbooks.

Provides:
- In-memory registry for playbook lookup
- YAML loading for file-based playbooks
- Auto-discovery of built-in playbooks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .models import Playbook

logger = logging.getLogger(__name__)

# Built-in playbook directory
BUILTIN_DIR = Path(__file__).parent / "builtin"


class PlaybookRegistry:
    """Registry for discovering and managing playbooks."""

    def __init__(self) -> None:
        self._playbooks: dict[str, Playbook] = {}
        self._loaded_builtins = False

    def register(self, playbook: Playbook) -> None:
        """Register a playbook."""
        self._playbooks[playbook.id] = playbook
        logger.debug("Registered playbook: %s", playbook.id)

    def get(self, playbook_id: str) -> Playbook | None:
        """Get a playbook by ID."""
        self._ensure_builtins()
        return self._playbooks.get(playbook_id)

    def list(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Playbook]:
        """List playbooks with optional filtering."""
        self._ensure_builtins()
        results = list(self._playbooks.values())

        if category:
            results = [p for p in results if p.category == category]
        if tags:
            tag_set = set(tags)
            results = [p for p in results if tag_set.intersection(p.tags)]

        return sorted(results, key=lambda p: p.name)

    def from_yaml(self, yaml_path: str | Path) -> Playbook:
        """Load a playbook from a YAML file and register it.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Loaded and registered Playbook

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        import yaml  # lazy import

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Playbook YAML not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid playbook YAML: expected mapping, got {type(data).__name__}")

        if "id" not in data:
            # Derive ID from filename
            data["id"] = path.stem

        playbook = Playbook.from_dict(data)
        self.register(playbook)
        return playbook

    def _ensure_builtins(self) -> None:
        """Load built-in playbooks if not already loaded."""
        if self._loaded_builtins:
            return
        self._loaded_builtins = True
        self._load_builtins()

    def _load_builtins(self) -> None:
        """Load all YAML playbooks from the builtin directory."""
        if not BUILTIN_DIR.exists():
            logger.debug("No builtin playbook directory found")
            return

        for yaml_file in sorted(BUILTIN_DIR.glob("*.yaml")):
            try:
                self.from_yaml(yaml_file)
                logger.debug("Loaded builtin playbook: %s", yaml_file.stem)
            except (ValueError, FileNotFoundError, ImportError) as e:
                logger.warning("Failed to load builtin playbook %s: %s", yaml_file.name, e)

    @property
    def count(self) -> int:
        """Number of registered playbooks."""
        self._ensure_builtins()
        return len(self._playbooks)


# Module-level singleton
_registry_singleton: PlaybookRegistry | None = None


def get_playbook_registry() -> PlaybookRegistry:
    """Get or create the module-level PlaybookRegistry singleton."""
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = PlaybookRegistry()
    return _registry_singleton


__all__ = ["PlaybookRegistry", "get_playbook_registry"]
