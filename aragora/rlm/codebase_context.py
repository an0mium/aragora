"""
Codebase Context Builder - shared RLM-powered codebase context.

Provides a non-Nomic wrapper around NomicContextBuilder so other
systems (agents, workflows, handlers) can reuse the 10M-token
codebase context pipeline without importing nomic directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from aragora.nomic.context_builder import NomicContextBuilder


class CodebaseContextBuilder(NomicContextBuilder):
    """
    Shared codebase context builder using TRUE RLM + REPL when available.

    This wrapper exposes more general env vars so the pipeline can be
    enabled across the codebase without tying callers to Nomic settings.
    """

    def __init__(
        self,
        root_path: Path,
        max_context_bytes: int = 0,
        include_tests: Optional[bool] = None,
        knowledge_mound: Optional[Any] = None,
        full_corpus: Optional[bool] = None,
    ) -> None:
        env_max = (
            os.environ.get("ARAGORA_CODEBASE_MAX_CONTEXT_BYTES")
            or os.environ.get("ARAGORA_NOMIC_MAX_CONTEXT_BYTES")
            or os.environ.get("NOMIC_MAX_CONTEXT_BYTES")
            or os.environ.get("ARAGORA_RLM_MAX_CONTENT_BYTES")
        )
        if max_context_bytes == 0 and env_max:
            try:
                max_context_bytes = int(env_max)
            except ValueError:
                max_context_bytes = 0

        if include_tests is None:
            include_tests = os.environ.get("ARAGORA_CODEBASE_INCLUDE_TESTS", "1") == "1"

        if full_corpus is None:
            # Default to off outside Nomic to avoid heavy LLM calls unless explicitly enabled.
            full_corpus = os.environ.get("ARAGORA_CODEBASE_RLM_FULL_CORPUS", "0") == "1"

        super().__init__(
            aragora_path=root_path,
            max_context_bytes=max_context_bytes,
            include_tests=include_tests,
            knowledge_mound=knowledge_mound,
            full_corpus=full_corpus,
        )


__all__ = ["CodebaseContextBuilder"]
