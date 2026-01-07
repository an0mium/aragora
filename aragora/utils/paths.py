"""
Path validation utilities for security.

Provides safe path operations to prevent directory traversal attacks.
"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class PathTraversalError(ValueError):
    """Raised when a path traversal attempt is detected."""
    pass


def safe_path(
    base_dir: Union[Path, str],
    user_path: Union[Path, str],
    *,
    allow_symlinks: bool = False,
    must_exist: bool = False,
) -> Path:
    """
    Safely join a user-provided path to a base directory.

    Validates that the resulting path stays within the base directory,
    preventing directory traversal attacks (e.g., "../../../etc/passwd").

    Args:
        base_dir: The base directory that paths must stay within
        user_path: The user-provided path component
        allow_symlinks: If False (default), reject symlinks that could escape
        must_exist: If True, raise error if path doesn't exist

    Returns:
        Resolved Path object within base_dir

    Raises:
        PathTraversalError: If the path would escape base_dir
        FileNotFoundError: If must_exist=True and path doesn't exist

    Examples:
        >>> safe_path("/data/replays", "debate-123")
        PosixPath('/data/replays/debate-123')

        >>> safe_path("/data", "../etc/passwd")
        PathTraversalError: Path traversal blocked: ../etc/passwd

        >>> safe_path("/data", "subdir/../other")  # Normalized, stays in base
        PosixPath('/data/other')
    """
    base = Path(base_dir).resolve()
    combined = (base / user_path).resolve()

    # Check if resolved path is within base directory
    try:
        combined.relative_to(base)
    except ValueError:
        logger.warning(f"Path traversal blocked: {user_path}")
        raise PathTraversalError(f"Path traversal blocked: {user_path}")

    # Check for symlinks if not allowed
    if not allow_symlinks:
        # Check each component in the path for symlinks
        current = base
        for part in Path(user_path).parts:
            if part in (".", ".."):
                continue
            current = current / part
            if current.exists() and current.is_symlink():
                target = current.resolve()
                try:
                    target.relative_to(base)
                except ValueError:
                    logger.warning(f"Symlink escapes base directory: {user_path}")
                    raise PathTraversalError(f"Symlink escapes base directory: {user_path}")

    if must_exist and not combined.exists():
        raise FileNotFoundError(f"Path does not exist: {combined}")

    return combined


def validate_path_component(component: str) -> str:
    """
    Validate a single path component (filename or directory name).

    Rejects components that could be used for traversal or are invalid.

    Args:
        component: A single path component (no slashes)

    Returns:
        The validated component

    Raises:
        PathTraversalError: If component is invalid or could enable traversal

    Examples:
        >>> validate_path_component("debate-123")
        'debate-123'

        >>> validate_path_component("..")
        PathTraversalError: Invalid path component: ..
    """
    # Reject empty or whitespace-only
    if not component or not component.strip():
        raise PathTraversalError("Invalid path component: empty or whitespace")

    # Reject path separators
    if "/" in component or "\\" in component:
        raise PathTraversalError(f"Invalid path component: contains separator: {component}")

    # Reject traversal patterns
    if component in (".", ".."):
        raise PathTraversalError(f"Invalid path component: {component}")

    # Reject null bytes (could truncate paths in some systems)
    if "\x00" in component:
        raise PathTraversalError("Invalid path component: contains null byte")

    return component


def is_safe_path(base_dir: Union[Path, str], user_path: Union[Path, str]) -> bool:
    """
    Check if a user-provided path stays within a base directory.

    Non-throwing version of safe_path() for conditional checks.

    Args:
        base_dir: The base directory that paths must stay within
        user_path: The user-provided path component

    Returns:
        True if path is safe, False otherwise
    """
    try:
        safe_path(base_dir, user_path)
        return True
    except (PathTraversalError, FileNotFoundError):
        return False
