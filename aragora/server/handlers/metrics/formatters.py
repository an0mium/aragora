"""
Formatting utilities for metrics display.
"""

from __future__ import annotations


def format_uptime(seconds: float) -> str:
    """Format uptime as human-readable string.

    Args:
        seconds: Uptime in seconds

    Returns:
        Human-readable string like "2d 5h 30m" or "45m 12s"
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_size(size_bytes: int) -> str:
    """Format size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string like "1.5 MB" or "256 KB"
    """
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024
    return f"{size_float:.1f} TB"
