"""
Dynamic Skill Loader Module.

Provides utilities for dynamically discovering and loading skills from:
- Python modules
- Plugin directories
- Manifest files (YAML/JSON)

Usage:
    from aragora.skills import SkillLoader, get_skill_registry

    loader = SkillLoader()

    # Load from a module
    loader.load_module("aragora.skills.builtin.web_search")

    # Load all skills from a directory
    loader.load_directory("./custom_skills")

    # Auto-discover built-in skills
    loader.load_builtin_skills()
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillManifest
from .registry import SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """Error loading a skill."""

    pass


class SkillLoader:
    """
    Dynamic skill loader.

    Discovers and loads skills from various sources:
    - Python modules by import path
    - Python files by path
    - Directories containing skill modules
    - Manifest files (for declarative skills)
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        auto_register: bool = True,
    ):
        """
        Initialize the skill loader.

        Args:
            registry: SkillRegistry to register skills with (uses default if None)
            auto_register: If True, automatically register loaded skills
        """
        self._registry = registry or get_skill_registry()
        self._auto_register = auto_register
        self._loaded_modules: set[str] = set()

    def load_module(
        self,
        module_path: str,
        register: Optional[bool] = None,
    ) -> List[Skill]:
        """
        Load skills from a Python module.

        Args:
            module_path: Python import path (e.g., "aragora.skills.builtin.web_search")
            register: Whether to register (defaults to auto_register)

        Returns:
            List of loaded skills

        Raises:
            SkillLoadError: If module cannot be loaded
        """
        if module_path in self._loaded_modules:
            logger.debug(f"Module already loaded: {module_path}")
            return []

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise SkillLoadError(f"Failed to import module {module_path}: {e}") from e

        skills = self._extract_skills_from_module(module)
        self._loaded_modules.add(module_path)

        should_register = register if register is not None else self._auto_register
        if should_register:
            for skill in skills:
                self._registry.register(skill, replace=True)

        logger.info(f"Loaded {len(skills)} skills from {module_path}")
        return skills

    def load_file(
        self,
        file_path: str | Path,
        register: Optional[bool] = None,
    ) -> List[Skill]:
        """
        Load skills from a Python file.

        Args:
            file_path: Path to the Python file
            register: Whether to register (defaults to auto_register)

        Returns:
            List of loaded skills

        Raises:
            SkillLoadError: If file cannot be loaded
        """
        path = Path(file_path)
        if not path.exists():
            raise SkillLoadError(f"File not found: {path}")

        if not path.suffix == ".py":
            raise SkillLoadError(f"Not a Python file: {path}")

        # Generate a unique module name
        module_name = f"_skill_module_{path.stem}_{hash(str(path.absolute())) & 0xFFFFFFFF}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise SkillLoadError(f"Failed to create module spec for {path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise SkillLoadError(f"Failed to load file {path}: {e}") from e

        skills = self._extract_skills_from_module(module)

        should_register = register if register is not None else self._auto_register
        if should_register:
            for skill in skills:
                self._registry.register(skill, replace=True)

        logger.info(f"Loaded {len(skills)} skills from {path}")
        return skills

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        register: Optional[bool] = None,
    ) -> List[Skill]:
        """
        Load skills from all Python files in a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories
            register: Whether to register (defaults to auto_register)

        Returns:
            List of loaded skills
        """
        path = Path(directory)
        if not path.exists():
            raise SkillLoadError(f"Directory not found: {path}")

        if not path.is_dir():
            raise SkillLoadError(f"Not a directory: {path}")

        all_skills: List[Skill] = []
        pattern = "**/*.py" if recursive else "*.py"

        for file_path in path.glob(pattern):
            if file_path.name.startswith("_"):
                continue  # Skip private modules

            try:
                skills = self.load_file(file_path, register=register)
                all_skills.extend(skills)
            except SkillLoadError as e:
                logger.warning(f"Skipping file {file_path}: {e}")

        logger.info(f"Loaded {len(all_skills)} skills from directory {path}")
        return all_skills

    def load_builtin_skills(self, register: Optional[bool] = None) -> List[Skill]:
        """
        Load all built-in skills from aragora.skills.builtin.

        Args:
            register: Whether to register (defaults to auto_register)

        Returns:
            List of loaded skills
        """
        builtin_modules = [
            "aragora.skills.builtin.web_search",
            "aragora.skills.builtin.code_execution",
            "aragora.skills.builtin.knowledge_query",
            "aragora.skills.builtin.evidence_fetch",
        ]

        all_skills: List[Skill] = []
        for module_path in builtin_modules:
            try:
                skills = self.load_module(module_path, register=register)
                all_skills.extend(skills)
            except SkillLoadError as e:
                logger.debug(f"Built-in skill module not available: {module_path}: {e}")

        return all_skills

    def _extract_skills_from_module(self, module: Any) -> List[Skill]:
        """
        Extract Skill instances from a module.

        Looks for:
        1. A 'register_skills' function that returns skills
        2. A 'SKILLS' list/tuple of skill instances
        3. Any class that subclasses Skill
        """
        skills: List[Skill] = []

        # Check for register_skills function
        if hasattr(module, "register_skills"):
            try:
                result = module.register_skills()
                if isinstance(result, (list, tuple)):
                    skills.extend(result)
                elif isinstance(result, Skill):
                    skills.append(result)
            except Exception as e:
                logger.warning(f"register_skills() failed: {e}")

        # Check for SKILLS constant
        if hasattr(module, "SKILLS"):
            module_skills = module.SKILLS
            if isinstance(module_skills, (list, tuple)):
                skills.extend(s for s in module_skills if isinstance(s, Skill))

        # Find Skill subclasses
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Check if it's a Skill subclass (not Skill itself)
            if (
                isinstance(obj, type)
                and issubclass(obj, Skill)
                and obj is not Skill
                and not getattr(obj, "__abstract__", False)
            ):
                try:
                    # Try to instantiate
                    instance = obj()
                    if instance not in skills:
                        skills.append(instance)
                except Exception as e:
                    logger.debug(f"Could not instantiate {name}: {e}")

            # Check if it's already a Skill instance
            elif isinstance(obj, Skill):
                if obj not in skills:
                    skills.append(obj)

        return skills

    def load_from_manifest(
        self,
        manifest_path: str | Path,
        register: Optional[bool] = None,
    ) -> Optional[Skill]:
        """
        Load a skill from a manifest file (YAML or JSON).

        This is for declarative skills that don't need Python code.

        Args:
            manifest_path: Path to the manifest file
            register: Whether to register (defaults to auto_register)

        Returns:
            Loaded skill or None if failed
        """
        path = Path(manifest_path)
        if not path.exists():
            raise SkillLoadError(f"Manifest not found: {path}")

        try:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    with open(path) as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    raise SkillLoadError("PyYAML not installed for YAML manifests")
            elif path.suffix == ".json":
                import json

                with open(path) as f:
                    data = json.load(f)
            else:
                raise SkillLoadError(f"Unsupported manifest format: {path.suffix}")

            manifest = SkillManifest.from_dict(data)

            # Create a declarative skill wrapper
            skill = DeclarativeSkill(manifest)

            should_register = register if register is not None else self._auto_register
            if should_register:
                self._registry.register(skill, replace=True)

            logger.info(f"Loaded declarative skill from {path}: {manifest.name}")
            return skill

        except Exception as e:
            raise SkillLoadError(f"Failed to load manifest {path}: {e}") from e


class DeclarativeSkill(Skill):
    """
    A skill defined purely through a manifest.

    Used for simple skills that delegate to external services or
    when the implementation is provided by a plugin system.
    """

    def __init__(
        self,
        skill_manifest: SkillManifest,
        executor: Optional[Any] = None,
    ):
        """
        Initialize declarative skill.

        Args:
            skill_manifest: The skill manifest
            executor: Optional executor for the skill
        """
        self._manifest = skill_manifest
        self._executor = executor

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Any,
    ) -> Any:
        """Execute the skill."""
        from .base import SkillResult, SkillStatus

        if self._executor:
            return await self._executor(input_data, context)

        # Default implementation returns not implemented
        return SkillResult(
            status=SkillStatus.NOT_IMPLEMENTED,
            error_message=f"Skill '{self._manifest.name}' has no executor",
        )


# Convenience function
def load_skills(
    *sources: str | Path,
    registry: Optional[SkillRegistry] = None,
) -> List[Skill]:
    """
    Load skills from multiple sources.

    Args:
        *sources: Module paths, file paths, or directory paths
        registry: Optional registry to use

    Returns:
        List of all loaded skills
    """
    loader = SkillLoader(registry=registry, auto_register=True)
    all_skills: List[Skill] = []

    for source in sources:
        source_str = str(source)

        # Determine source type
        if os.path.exists(source_str):
            path = Path(source_str)
            if path.is_dir():
                skills = loader.load_directory(path)
            else:
                skills = loader.load_file(path)
        else:
            # Assume it's a module path
            skills = loader.load_module(source_str)

        all_skills.extend(skills)

    return all_skills
