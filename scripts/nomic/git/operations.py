"""
Git operations for nomic loop.

Provides safe git operations including stash management,
diff retrieval, and commit helpers.
"""

import subprocess
from pathlib import Path
from typing import Callable, List, Optional


def git_stash_create(repo_path: Path, log_func: Callable = print) -> Optional[str]:
    """
    Create a git stash for transactional safety.

    Args:
        repo_path: Path to the git repository
        log_func: Function to log messages

    Returns:
        Stash reference if created, None if nothing to stash
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            return None

        result = subprocess.run(
            ["git", "stash", "push", "-m", "nomic-implement-backup"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ref_result = subprocess.run(
                ["git", "stash", "list", "-1", "--format=%H"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            return ref_result.stdout.strip() or "stash@{0}"
    except Exception as e:
        log_func(f"Warning: Could not create stash: {e}")
    return None


def git_stash_pop(repo_path: Path, stash_ref: Optional[str], log_func: Callable = print) -> None:
    """
    Pop a stash to restore previous state.

    Args:
        repo_path: Path to the git repository
        stash_ref: Stash reference to pop
        log_func: Function to log messages
    """
    if not stash_ref:
        return
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "stash", "pop"],
            cwd=repo_path,
            capture_output=True,
        )
    except Exception as e:
        log_func(f"Warning: Could not pop stash: {e}")


def get_git_diff(repo_path: Path) -> str:
    """
    Get current git diff.

    Args:
        repo_path: Path to the git repository

    Returns:
        Diff summary string
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def get_git_diff_full(repo_path: Path) -> str:
    """
    Get full git diff content.

    Args:
        repo_path: Path to the git repository

    Returns:
        Full diff content
    """
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def get_git_changed_files(repo_path: Path) -> List[str]:
    """
    Get list of changed files from git.

    Args:
        repo_path: Path to the git repository

    Returns:
        List of changed file paths
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        return files
    except Exception:
        return []


def get_modified_files(repo_path: Path) -> List[str]:
    """
    Get list of modified (staged and unstaged) files.

    Args:
        repo_path: Path to the git repository

    Returns:
        List of modified file paths
    """
    try:
        # Get both staged and unstaged changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        files = []
        for line in result.stdout.split("\n"):
            if line.strip():
                # Format is "XY filename" where X=staged, Y=unstaged
                parts = line.split()
                if len(parts) >= 2:
                    files.append(parts[-1])
        return files
    except Exception:
        return []


def git_add_all(repo_path: Path, log_func: Callable = print) -> bool:
    """
    Stage all changes.

    Args:
        repo_path: Path to the git repository
        log_func: Function to log messages

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        log_func(f"Error staging changes: {e}")
        return False


def git_commit(repo_path: Path, message: str, log_func: Callable = print) -> Optional[str]:
    """
    Create a git commit.

    Args:
        repo_path: Path to the git repository
        message: Commit message
        log_func: Function to log messages

    Returns:
        Commit hash if successful, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Get the commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            return hash_result.stdout.strip()[:12]
    except Exception as e:
        log_func(f"Error committing: {e}")
    return None


def git_reset_hard(repo_path: Path, log_func: Callable = print) -> bool:
    """
    Hard reset to HEAD (discard all changes).

    Args:
        repo_path: Path to the git repository
        log_func: Function to log messages

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        log_func(f"Error resetting: {e}")
        return False


def selective_rollback(repo_path: Path, files: List[str], log_func: Callable = print) -> bool:
    """
    Rollback specific files to their last committed state.

    Args:
        repo_path: Path to the git repository
        files: List of file paths to rollback
        log_func: Function to log messages

    Returns:
        True if successful
    """
    if not files:
        return True

    try:
        for file_path in files:
            result = subprocess.run(
                ["git", "checkout", "HEAD", "--", file_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                log_func(f"Warning: Could not rollback {file_path}")
        return True
    except Exception as e:
        log_func(f"Error during selective rollback: {e}")
        return False


def preserve_failed_work(
    repo_path: Path, branch_name: str, log_func: Callable = print
) -> Optional[str]:
    """
    Preserve failed work in a separate branch before rollback.

    Args:
        repo_path: Path to the git repository
        branch_name: Name for the preservation branch
        log_func: Function to log messages

    Returns:
        Branch name if created, None otherwise
    """
    try:
        # Check if there are changes to preserve
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            return None

        # Create branch
        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log_func(f"Warning: Could not create branch {branch_name}")
            return None

        # Stage and commit
        subprocess.run(["git", "add", "-A"], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"[nomic] Preserving failed work: {branch_name}"],
            cwd=repo_path,
            capture_output=True,
        )

        # Return to original branch
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=repo_path,
            capture_output=True,
        )

        return branch_name

    except Exception as e:
        log_func(f"Error preserving work: {e}")
        return None
