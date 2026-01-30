"""
File Analysis Skill.

Provides capabilities to analyze file contents, including:
- Text file analysis (word count, line count, encoding)
- Code file analysis (language detection, complexity metrics)
- Document structure analysis
- Content summarization
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


# Map file extensions to programming languages
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript_react",
    ".tsx": "typescript_react",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".tex": "latex",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
}


@dataclass
class FileMetrics:
    """Metrics about a file."""

    file_path: str
    file_name: str
    extension: str
    size_bytes: int
    line_count: int
    word_count: int
    char_count: int
    language: str | None = None
    mime_type: str | None = None
    encoding: str = "utf-8"

    # Code-specific metrics
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0

    # Content analysis
    top_words: list[tuple[str, int]] = field(default_factory=list)
    complexity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "size_readable": self._format_size(self.size_bytes),
            "line_count": self.line_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "language": self.language,
            "mime_type": self.mime_type,
            "encoding": self.encoding,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "code_lines": self.code_lines,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "top_words": [{"word": w, "count": c} for w, c in self.top_words],
            "complexity_score": round(self.complexity_score, 2),
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size to human readable."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class FileAnalysisSkill(Skill):
    """
    Skill for analyzing file contents.

    Supports:
    - Basic file metrics (size, lines, words)
    - Code analysis (functions, classes, complexity)
    - Language detection
    - Content summary
    - Top word frequency
    """

    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        top_words_count: int = 20,
    ):
        """
        Initialize file analysis skill.

        Args:
            max_file_size: Maximum file size to analyze in bytes
            top_words_count: Number of top words to include
        """
        self._max_file_size = max_file_size
        self._top_words_count = top_words_count

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="file_analysis",
            version="1.0.0",
            description="Analyze file contents for metrics and structure",
            capabilities=[
                SkillCapability.READ_LOCAL,
            ],
            input_schema={
                "content": {
                    "type": "string",
                    "description": "File content to analyze (if not providing file_path)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to file to analyze (if not providing content)",
                },
                "file_name": {
                    "type": "string",
                    "description": "File name (for language detection when providing content)",
                },
                "analyze_code": {
                    "type": "boolean",
                    "description": "Perform code-specific analysis",
                    "default": True,
                },
                "include_top_words": {
                    "type": "boolean",
                    "description": "Include top word frequency",
                    "default": True,
                },
                "extract_structure": {
                    "type": "boolean",
                    "description": "Extract document/code structure",
                    "default": False,
                },
            },
            tags=["file", "analysis", "metrics", "code"],
            required_permissions=["files:read"],
            debate_compatible=True,
            max_execution_time_seconds=30.0,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute file analysis."""
        content = input_data.get("content")
        file_path = input_data.get("file_path")
        file_name = input_data.get("file_name", "unknown")
        analyze_code = input_data.get("analyze_code", True)
        include_top_words = input_data.get("include_top_words", True)
        extract_structure = input_data.get("extract_structure", False)

        if not content and not file_path:
            return SkillResult.create_failure(
                "Either 'content' or 'file_path' is required",
                error_code="missing_input",
            )

        try:
            # Read file if path provided
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    return SkillResult.create_failure(
                        f"File not found: {file_path}",
                        error_code="file_not_found",
                    )

                if path.stat().st_size > self._max_file_size:
                    return SkillResult.create_failure(
                        f"File too large (max {self._max_file_size} bytes)",
                        error_code="file_too_large",
                    )

                # Detect encoding
                encoding = self._detect_encoding(path)
                try:
                    content = path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    content = path.read_text(encoding="latin-1")
                    encoding = "latin-1"

                file_name = path.name
                size_bytes = path.stat().st_size
            else:
                encoding = "utf-8"
                size_bytes = len(content.encode("utf-8"))

            # Basic metrics
            lines = content.split("\n")
            words = content.split()
            extension = os.path.splitext(file_name)[1].lower()
            language = LANGUAGE_MAP.get(extension)
            mime_type, _ = mimetypes.guess_type(file_name)

            metrics = FileMetrics(
                file_path=file_path or "",
                file_name=file_name,
                extension=extension,
                size_bytes=size_bytes,
                line_count=len(lines),
                word_count=len(words),
                char_count=len(content),
                language=language,
                mime_type=mime_type,
                encoding=encoding,
            )

            # Code analysis
            if analyze_code and language:
                self._analyze_code(content, lines, metrics, language)

            # Top words
            if include_top_words:
                metrics.top_words = self._get_top_words(content)

            # Structure extraction
            structure = None
            if extract_structure:
                structure = self._extract_structure(content, language)

            result = {
                "metrics": metrics.to_dict(),
            }

            if structure:
                result["structure"] = structure

            return SkillResult.create_success(result)

        except Exception as e:
            logger.exception(f"File analysis failed: {e}")
            return SkillResult.create_failure(f"Analysis failed: {e}")

    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet

            with open(path, "rb") as f:
                raw = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw)
            return result.get("encoding", "utf-8") or "utf-8"
        except ImportError:
            return "utf-8"
        except Exception:
            return "utf-8"

    def _analyze_code(
        self,
        content: str,
        lines: list[str],
        metrics: FileMetrics,
        language: str,
    ) -> None:
        """Perform code-specific analysis."""
        # Count blank, comment, and code lines
        blank_count = 0
        comment_count = 0
        code_count = 0

        # Comment patterns by language
        single_comment = {
            "python": "#",
            "javascript": "//",
            "typescript": "//",
            "java": "//",
            "c": "//",
            "cpp": "//",
            "go": "//",
            "rust": "//",
            "ruby": "#",
            "shell": "#",
            "bash": "#",
        }

        comment_prefix = single_comment.get(language, "#")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_count += 1
            elif stripped.startswith(comment_prefix):
                comment_count += 1
            else:
                code_count += 1

        metrics.blank_lines = blank_count
        metrics.comment_lines = comment_count
        metrics.code_lines = code_count

        # Count functions/methods
        function_patterns = {
            "python": r"^\s*def\s+\w+\s*\(",
            "javascript": r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
            "typescript": r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
            "java": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+\w+\s*\([^)]*\)\s*\{",
            "go": r"func\s+(?:\([^)]+\)\s+)?\w+\s*\(",
            "rust": r"(?:pub\s+)?fn\s+\w+",
        }

        if language in function_patterns:
            metrics.function_count = len(
                re.findall(function_patterns[language], content, re.MULTILINE)
            )

        # Count classes
        class_patterns = {
            "python": r"^\s*class\s+\w+",
            "javascript": r"class\s+\w+",
            "typescript": r"class\s+\w+",
            "java": r"(?:public|private|protected)?\s*class\s+\w+",
            "rust": r"(?:pub\s+)?struct\s+\w+",
        }

        if language in class_patterns:
            metrics.class_count = len(re.findall(class_patterns[language], content, re.MULTILINE))

        # Count imports
        import_patterns = {
            "python": r"^\s*(?:import|from)\s+\w+",
            "javascript": r"^\s*import\s+",
            "typescript": r"^\s*import\s+",
            "java": r"^\s*import\s+",
            "go": r"^\s*import\s+",
            "rust": r"^\s*use\s+",
        }

        if language in import_patterns:
            metrics.import_count = len(re.findall(import_patterns[language], content, re.MULTILINE))

        # Calculate complexity score (simple heuristic)
        # Based on cyclomatic complexity indicators
        complexity_keywords = [
            "if",
            "else",
            "elif",
            "for",
            "while",
            "switch",
            "case",
            "try",
            "catch",
            "except",
            "finally",
            "and",
            "or",
            "&&",
            "||",
        ]

        complexity_count = sum(len(re.findall(rf"\b{kw}\b", content)) for kw in complexity_keywords)

        # Normalize by code lines
        if metrics.code_lines > 0:
            metrics.complexity_score = complexity_count / metrics.code_lines * 10
        else:
            metrics.complexity_score = 0

    def _get_top_words(self, content: str) -> list[tuple[str, int]]:
        """Get top words by frequency."""
        # Remove common programming syntax and split into words
        text = re.sub(r"[^a-zA-Z\s]", " ", content)
        words = text.lower().split()

        # Filter out very short words and common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            # Common code words
            "def",
            "class",
            "function",
            "return",
            "import",
            "from",
            "self",
            "true",
            "false",
            "null",
            "none",
            "var",
            "let",
            "const",
        }

        filtered = [w for w in words if len(w) > 2 and w not in stop_words]
        counter = Counter(filtered)
        return counter.most_common(self._top_words_count)

    def _extract_structure(self, content: str, language: str | None) -> dict[str, Any]:
        """Extract document/code structure."""
        structure: dict[str, Any] = {
            "sections": [],
            "functions": [],
            "classes": [],
        }

        if not language:
            # Extract markdown-style headers for non-code files
            headers = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
            for level, title in headers:
                structure["sections"].append({"level": len(level), "title": title.strip()})
            return structure

        # Extract Python structure
        if language == "python":
            # Functions
            for match in re.finditer(r"^\s*def\s+(\w+)\s*\(([^)]*)\)", content, re.MULTILINE):
                structure["functions"].append(
                    {
                        "name": match.group(1),
                        "params": match.group(2).strip(),
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

            # Classes
            for match in re.finditer(r"^\s*class\s+(\w+)(?:\(([^)]*)\))?:", content, re.MULTILINE):
                structure["classes"].append(
                    {
                        "name": match.group(1),
                        "bases": match.group(2).strip() if match.group(2) else "",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        # Extract JavaScript/TypeScript structure
        elif language in ("javascript", "typescript"):
            # Functions
            for match in re.finditer(
                r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function)",
                content,
            ):
                name = match.group(1) or match.group(2)
                structure["functions"].append(
                    {
                        "name": name,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

            # Classes
            for match in re.finditer(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", content):
                structure["classes"].append(
                    {
                        "name": match.group(1),
                        "extends": match.group(2) or "",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        return structure


# Skill instance for registration
SKILLS = [FileAnalysisSkill()]
