"""
Filter implementations for folder scanning.

Supports gitignore-style patterns with ** for recursive matching.
"""

import re
from functools import lru_cache
from typing import Optional


class PatternMatcher:
    """
    Gitignore-style pattern matching for files and directories.

    Supports:
    - * for single path component wildcard
    - ** for recursive directory matching
    - ? for single character
    - [abc] for character classes
    - ! prefix for negation (include despite earlier exclusion)
    """

    def __init__(
        self,
        exclude_patterns: list[str],
        include_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the pattern matcher.

        Args:
            exclude_patterns: Patterns for files/dirs to exclude
            include_patterns: If provided, only include files matching these patterns
        """
        self._exclude_patterns = exclude_patterns or []
        self._include_patterns = include_patterns or []

        # Separate negation patterns (those starting with !)
        self._negation_patterns = [p[1:] for p in self._exclude_patterns if p.startswith("!")]
        self._exclude_patterns = [p for p in self._exclude_patterns if not p.startswith("!")]

        # Pre-compile regex patterns for performance
        self._compiled_exclude = [self._compile_pattern(p) for p in self._exclude_patterns]
        self._compiled_include = [self._compile_pattern(p) for p in self._include_patterns]
        self._compiled_negation = [self._compile_pattern(p) for p in self._negation_patterns]

    @staticmethod
    @lru_cache(maxsize=1000)
    def _compile_pattern(pattern: str) -> re.Pattern:
        """
        Convert a gitignore-style pattern to a regex.

        Handles:
        - ** for recursive directory matching
        - * for single component matching
        - ? for single character
        - [abc] for character classes
        """
        # Escape special regex chars except our wildcards
        regex = ""
        i = 0
        while i < len(pattern):
            c = pattern[i]

            if c == "*":
                # Check for **
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    # ** matches any path (including /)
                    if i + 2 < len(pattern) and pattern[i + 2] == "/":
                        # **/ at start or middle - matches zero or more directories
                        regex += r"(?:.*/)?"
                        i += 3
                    else:
                        # ** matches everything
                        regex += r".*"
                        i += 2
                else:
                    # * matches anything except /
                    regex += r"[^/]*"
                    i += 1
            elif c == "?":
                # ? matches single char except /
                regex += r"[^/]"
                i += 1
            elif c == "[":
                # Character class - find the closing ]
                j = i + 1
                if j < len(pattern) and pattern[j] == "!":
                    j += 1
                if j < len(pattern) and pattern[j] == "]":
                    j += 1
                while j < len(pattern) and pattern[j] != "]":
                    j += 1
                if j < len(pattern):
                    # Include the character class as-is
                    regex += pattern[i : j + 1]
                    i = j + 1
                else:
                    # No closing ] - treat [ as literal
                    regex += re.escape(c)
                    i += 1
            else:
                # Escape other special regex characters
                regex += re.escape(c)
                i += 1

        # Anchor the pattern
        # If pattern starts with /, it's anchored to root
        # Otherwise, it can match anywhere in the path
        if regex.startswith(r"\/"):
            regex = "^" + regex[2:]  # Remove leading \/ and anchor
        else:
            regex = r"(?:^|/)" + regex

        # Pattern should match end or be followed by /
        if not regex.endswith(r".*"):
            regex += r"(?:$|/)"

        return re.compile(regex)

    def is_excluded(self, path: str) -> tuple[bool, Optional[str]]:
        """
        Check if a path should be excluded.

        Args:
            path: Relative path to check (uses / as separator)

        Returns:
            Tuple of (is_excluded, matched_pattern or None)
        """
        # Normalize path separators
        path = path.replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]

        # First check if path matches any include pattern
        # If include patterns are specified and path doesn't match any, exclude it
        if self._include_patterns:
            matches_include = False
            for pattern, compiled in zip(self._include_patterns, self._compiled_include):
                if compiled.search(path):
                    matches_include = True
                    break
            if not matches_include:
                return True, "No include pattern matched"

        # Check exclusion patterns
        for pattern, compiled in zip(self._exclude_patterns, self._compiled_exclude):
            if compiled.search(path):
                # Check if there's a negation that overrides
                negated = False
                for neg_pattern, neg_compiled in zip(
                    self._negation_patterns, self._compiled_negation
                ):
                    if neg_compiled.search(path):
                        negated = True
                        break

                if not negated:
                    return True, pattern

        return False, None

    def is_directory_excluded(self, path: str) -> tuple[bool, Optional[str]]:
        """
        Check if a directory should be excluded from traversal.

        This is optimized to check patterns that explicitly target directories.

        Args:
            path: Relative directory path

        Returns:
            Tuple of (is_excluded, matched_pattern or None)
        """
        # Ensure path ends with / for directory matching
        path = path.replace("\\", "/")
        if not path.endswith("/"):
            path += "/"
        if path.startswith("./"):
            path = path[2:]

        # Check patterns that look like directory patterns
        for pattern, compiled in zip(self._exclude_patterns, self._compiled_exclude):
            # Patterns ending with ** or / are directory patterns
            if pattern.endswith("**") or pattern.endswith("/"):
                if compiled.search(path):
                    return True, pattern
            # Also check patterns like **/node_modules/**
            elif "**" in pattern:
                if compiled.search(path):
                    return True, pattern

        return False, None


class SizeFilter:
    """Filter files based on size limits."""

    def __init__(
        self,
        max_file_size_bytes: int,
        max_total_size_bytes: int,
        max_file_count: int,
    ):
        """
        Initialize the size filter.

        Args:
            max_file_size_bytes: Maximum size for a single file
            max_total_size_bytes: Maximum total size for all files
            max_file_count: Maximum number of files
        """
        self.max_file_size_bytes = max_file_size_bytes
        self.max_total_size_bytes = max_total_size_bytes
        self.max_file_count = max_file_count

        self._current_total_size = 0
        self._current_file_count = 0

    def reset(self) -> None:
        """Reset the running totals."""
        self._current_total_size = 0
        self._current_file_count = 0

    def check_file(self, size_bytes: int) -> tuple[bool, Optional[str]]:
        """
        Check if a file should be excluded based on size.

        Args:
            size_bytes: Size of the file in bytes

        Returns:
            Tuple of (is_excluded, reason or None)
        """
        # Check individual file size
        if size_bytes > self.max_file_size_bytes:
            return (
                True,
                f"File size ({size_bytes} bytes) exceeds limit ({self.max_file_size_bytes} bytes)",
            )

        # Check if adding this file would exceed total size
        if self._current_total_size + size_bytes > self.max_total_size_bytes:
            return True, f"Would exceed total size limit ({self.max_total_size_bytes} bytes)"

        # Check file count
        if self._current_file_count >= self.max_file_count:
            return True, f"Would exceed file count limit ({self.max_file_count} files)"

        return False, None

    def accept_file(self, size_bytes: int) -> None:
        """
        Record that a file has been accepted.

        Call this after check_file returns (False, None) and the file
        is confirmed to be included.
        """
        self._current_total_size += size_bytes
        self._current_file_count += 1

    @property
    def current_total_size(self) -> int:
        """Get the current total size of accepted files."""
        return self._current_total_size

    @property
    def current_file_count(self) -> int:
        """Get the current count of accepted files."""
        return self._current_file_count

    @property
    def remaining_size(self) -> int:
        """Get remaining size budget in bytes."""
        return max(0, self.max_total_size_bytes - self._current_total_size)

    @property
    def remaining_count(self) -> int:
        """Get remaining file count budget."""
        return max(0, self.max_file_count - self._current_file_count)


def parse_size_string(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Supports: b, kb, mb, gb, tb (case insensitive)

    Examples:
        "100mb" -> 104857600
        "1.5gb" -> 1610612736
        "500KB" -> 512000

    Args:
        size_str: Size string like "100mb", "1.5gb"

    Returns:
        Size in bytes

    Raises:
        ValueError: If the format is invalid
    """
    size_str = size_str.strip().lower()

    # Extract number and unit
    match = re.match(r"^([\d.]+)\s*(b|kb|mb|gb|tb)?$", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '100mb', '1.5gb'")

    number = float(match.group(1))
    unit = match.group(2) or "b"

    multipliers = {
        "b": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
        "tb": 1024 * 1024 * 1024 * 1024,
    }

    return int(number * multipliers[unit])


def format_size_bytes(size_bytes: int) -> str:
    """
    Format bytes as a human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string like "1.5 MB", "500 KB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
