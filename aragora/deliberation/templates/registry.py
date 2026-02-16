"""
Template Registry for managing deliberation templates.

Provides a centralized registry for template discovery, lookup,
and management. Templates can be:
- Built-in (defined in code)
- Loaded from YAML files
- Registered at runtime
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aragora.deliberation.templates.base import (
    DeliberationTemplate,
    TemplateCategory,
)
import builtins

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Central registry for deliberation templates.

    Supports:
    - Template registration and lookup
    - Filtering by category and tags
    - Loading from YAML files
    - Listing with pagination
    """

    def __init__(self) -> None:
        self._templates: dict[str, DeliberationTemplate] = {}
        self._initialized = False

    def register(self, template: DeliberationTemplate) -> None:
        """Register a template."""
        if template.name in self._templates:
            logger.warning(f"Overwriting existing template: {template.name}")
        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a template by name."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def get(self, name: str) -> DeliberationTemplate | None:
        """Get a template by name."""
        self._ensure_initialized()
        return self._templates.get(name)

    def list(
        self,
        category: TemplateCategory | None = None,
        tags: builtins.list[str] | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[DeliberationTemplate]:
        """
        List templates with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (OR matching)
            search: Search in name and description
            limit: Maximum results to return
            offset: Offset for pagination
        """
        self._ensure_initialized()
        templates = list(self._templates.values())

        # Filter by category
        if category:
            templates = [t for t in templates if t.category == category]

        # Filter by tags (OR matching)
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]

        # Search filter
        if search:
            search_lower = search.lower()
            templates = [
                t
                for t in templates
                if search_lower in t.name.lower() or search_lower in t.description.lower()
            ]

        # Sort by name
        templates.sort(key=lambda t: t.name)

        # Apply pagination
        return templates[offset : offset + limit]

    def count(
        self,
        category: TemplateCategory | None = None,
        tags: builtins.list[str] | None = None,
    ) -> int:
        """Count templates with optional filtering."""
        return len(self.list(category=category, tags=tags, limit=10000))

    def categories(self) -> dict[str, int]:
        """Get template counts by category."""
        self._ensure_initialized()
        counts: dict[str, int] = {}
        for template in self._templates.values():
            cat = template.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def load_from_yaml(self, yaml_path: Path) -> int:
        """
        Load templates from a YAML file.

        Returns the number of templates loaded.
        """
        try:
            import yaml

            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return 0

            templates = data.get("templates", [])
            if isinstance(data, list):
                templates = data

            count = 0
            for template_data in templates:
                try:
                    template = DeliberationTemplate.from_dict(template_data)
                    self.register(template)
                    count += 1
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to load template from YAML: {e}")

            return count
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML templates")
            return 0
        except (OSError, ValueError, Exception) as e:
            logger.error(f"Failed to load templates from {yaml_path}: {e}")
            return 0

    def load_from_directory(self, directory: Path) -> int:
        """
        Load all YAML templates from a directory.

        Returns the number of templates loaded.
        """
        count = 0
        for yaml_file in directory.glob("*.yaml"):
            count += self.load_from_yaml(yaml_file)
        for yaml_file in directory.glob("*.yml"):
            count += self.load_from_yaml(yaml_file)
        return count

    def recommend(
        self,
        question: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> builtins.list[DeliberationTemplate]:
        """Recommend templates for a given question.

        Uses TF-IDF cosine similarity when sklearn is available, falling
        back to keyword set intersection otherwise. If domain is provided,
        templates whose category matches the domain receive a score boost.

        Args:
            question: The user's question or topic
            domain: Optional domain hint (maps to TemplateCategory values)
            limit: Maximum results to return

        Returns:
            Top templates sorted by relevance score
        """
        self._ensure_initialized()
        if not question:
            return []

        try:
            results = self._recommend_tfidf(question, domain, limit)
            if results:
                return results
            # TF-IDF returned no results (zero similarity); fall back to keywords
            return self._recommend_keywords(question, domain, limit)
        except (ImportError, ValueError, TypeError):
            # Fallback to keyword matching if sklearn unavailable or any error
            return self._recommend_keywords(question, domain, limit)

    def _recommend_tfidf(
        self,
        question: str,
        domain: str | None,
        limit: int,
    ) -> builtins.list[DeliberationTemplate]:
        """Recommend templates using TF-IDF cosine similarity.

        Builds a TF-IDF matrix over all template text (name, description,
        tags, example_topics) and computes cosine similarity with the query.

        Raises:
            ImportError: If sklearn is not installed (triggers keyword fallback)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        templates = list(self._templates.values())
        if not templates:
            return []

        # Build corpus from template text
        corpus: builtins.list[str] = []
        for t in templates:
            text_parts = [t.name.replace("_", " "), t.description]
            text_parts.extend(t.tags)
            if t.example_topics:
                text_parts.extend(t.example_topics)
            corpus.append(" ".join(text_parts).lower())

        # Add query as last document
        corpus.append(question.lower())

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Compute similarity between query (last) and all templates
        query_vec = tfidf_matrix[-1]
        similarities = cosine_similarity(query_vec, tfidf_matrix[:-1]).flatten()

        # Apply domain boost
        scored: builtins.list[tuple[float, DeliberationTemplate]] = []
        for sim, template in zip(similarities, templates):
            score = float(sim)
            if domain and template.category.value == domain:
                score += 0.3  # domain boost
            if score > 0.15:  # minimum relevance threshold
                scored.append((score, template))

        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [t for _, t in scored[:limit]]

    def _recommend_keywords(
        self,
        question: str,
        domain: str | None,
        limit: int,
    ) -> builtins.list[DeliberationTemplate]:
        """Recommend templates using keyword set intersection.

        This is the fallback scoring method when sklearn is not available.
        Scores each template by keyword overlap with name, description,
        tags, and example_topics.
        """
        # Extract keywords from question
        keywords = set(question.lower().split())
        # Remove very short/common words and stop words that inflate irrelevant scores
        _STOP_WORDS = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "been", "being", "have", "has", "had", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall", "its",
            "not", "but", "they", "them", "their", "your", "our", "all", "any",
            "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "than", "too", "very", "also", "just", "about", "into",
            "over", "after", "before", "between", "under", "again", "then",
            "once", "here", "there", "when", "where", "how", "what", "which",
            "who", "whom", "why", "own", "same", "only",
        }
        keywords = {w for w in keywords if len(w) > 2 and w not in _STOP_WORDS}

        scored: builtins.list[tuple[float, DeliberationTemplate]] = []
        for template in self._templates.values():
            score = 0.0

            # Score against name words
            name_words = set(template.name.lower().replace("_", " ").split())
            score += len(keywords & name_words) * 3.0

            # Score against description words
            desc_words = set(template.description.lower().split())
            score += len(keywords & desc_words) * 2.0

            # Score against tags
            tag_words = {t.lower() for t in template.tags}
            score += len(keywords & tag_words) * 2.5

            # Score against example_topics
            if template.example_topics:
                topics_text = " ".join(template.example_topics).lower()
                topics_words = set(topics_text.split())
                score += len(keywords & topics_words) * 1.5

            # Domain boost
            if domain and template.category.value == domain:
                score += 5.0

            if score > 0:
                scored.append((score, template))

        # Sort by score descending, then by name for stability
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [t for _, t in scored[:limit]]

    def _ensure_initialized(self) -> None:
        """Ensure built-in templates are loaded."""
        if not self._initialized:
            from aragora.deliberation.templates.builtins import BUILTIN_TEMPLATES

            for template in BUILTIN_TEMPLATES.values():
                if template.name not in self._templates:
                    self._templates[template.name] = template
            self._initialized = True


# Global registry instance
_global_registry = TemplateRegistry()


def get_template(name: str) -> DeliberationTemplate | None:
    """Get a template from the global registry."""
    return _global_registry.get(name)


def list_templates(
    category: str | None = None,
    tags: list[str] | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DeliberationTemplate]:
    """List templates from the global registry."""
    cat = None
    if category:
        try:
            cat = TemplateCategory(category)
        except ValueError as e:
            logger.warning(f"Failed to parse category filter '{category}': {e}")
            # Keep cat as None to skip category filtering
    return _global_registry.list(category=cat, tags=tags, search=search, limit=limit, offset=offset)


def register_template(template: DeliberationTemplate) -> None:
    """Register a template in the global registry."""
    _global_registry.register(template)


def load_templates_from_yaml(path: Path) -> int:
    """Load templates from YAML into the global registry."""
    if path.is_dir():
        return _global_registry.load_from_directory(path)
    return _global_registry.load_from_yaml(path)


def match_templates(goal: str, limit: int = 5) -> list[DeliberationTemplate]:
    """Match deliberation templates against a goal string using TF-IDF similarity.

    Uses TF-IDF cosine similarity (when sklearn is available) or keyword
    overlap as fallback to score templates against the goal across
    name, description, tags, and example_topics. Returns the
    top matches sorted by relevance.

    Args:
        goal: The goal or question to match against
        limit: Maximum results to return

    Returns:
        List of matching templates sorted by relevance score
    """
    return _global_registry.recommend(question=goal, limit=limit)


def get_template_dict(name: str) -> dict[str, Any] | None:
    """Get a template as a dictionary."""
    template = get_template(name)
    return template.to_dict() if template else None
