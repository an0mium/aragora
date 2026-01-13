"""
Configuration constants and environment loading for nomic loop.

Environment variables for CI/automation support:
- NOMIC_AUTO_COMMIT: Skip interactive commit prompt (default OFF)
- NOMIC_AUTO_CONTINUE: Skip interactive cycle continuation prompt (default ON)
- NOMIC_MAX_CYCLE_SECONDS: Cycle-level hard timeout (default 2 hours)
- NOMIC_STALL_THRESHOLD: Stall detection threshold (default 30 minutes)
"""

import os
from pathlib import Path


# =============================================================================
# AUTOMATION FLAGS - Environment variables for CI/automation support
# =============================================================================

# Auto-commit: Skip interactive commit prompt (default OFF - requires explicit opt-in)
NOMIC_AUTO_COMMIT = os.environ.get("NOMIC_AUTO_COMMIT", "0") == "1"

# Auto-continue: Skip interactive cycle continuation prompt (default ON for loops)
NOMIC_AUTO_CONTINUE = os.environ.get("NOMIC_AUTO_CONTINUE", "1") == "1"

# Cycle-level hard timeout in seconds (default 2 hours)
NOMIC_MAX_CYCLE_SECONDS = int(os.environ.get("NOMIC_MAX_CYCLE_SECONDS", "7200"))

# Stall detection threshold in seconds (default 30 minutes)
NOMIC_STALL_THRESHOLD = int(os.environ.get("NOMIC_STALL_THRESHOLD", "1800"))

# Minimum time buffer before deadline to exit verify-fix loop (default 5 minutes)
NOMIC_FIX_DEADLINE_BUFFER = int(os.environ.get("NOMIC_FIX_DEADLINE_BUFFER", "300"))

# Time allocation per fix iteration in seconds (default 10 minutes)
# Used to estimate if there's time for another iteration
NOMIC_FIX_ITERATION_BUDGET = int(os.environ.get("NOMIC_FIX_ITERATION_BUDGET", "600"))

# Enable automatic checkpointing between phases (default ON)
NOMIC_AUTO_CHECKPOINT = os.environ.get("NOMIC_AUTO_CHECKPOINT", "1") == "1"


# =============================================================================
# INTEGRATION FLAGS - Enable/disable feature integrations
# =============================================================================

# Performance-based agent selection using ELO rankings
NOMIC_USE_PERFORMANCE_SELECTION = os.environ.get("NOMIC_USE_PERFORMANCE_SELECTION", "0") == "1"

# Trickster hollow consensus detection
NOMIC_TRICKSTER_ENABLED = os.environ.get("NOMIC_TRICKSTER_ENABLED", "0") == "1"
NOMIC_TRICKSTER_SENSITIVITY = float(os.environ.get("NOMIC_TRICKSTER_SENSITIVITY", "0.7"))

# Calibration tracking for prediction accuracy
NOMIC_CALIBRATION_ENABLED = os.environ.get("NOMIC_CALIBRATION_ENABLED", "1") == "1"

# Outcome tracking for consensus-to-implementation feedback
NOMIC_OUTCOME_TRACKING = os.environ.get("NOMIC_OUTCOME_TRACKING", "1") == "1"


# Default backup directory name
DEFAULT_BACKUP_DIR = ".nomic_backups"

# Default state file name
DEFAULT_STATE_FILE = ".nomic_state.json"


def load_dotenv(env_path: Path) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to the .env file
    """
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get a boolean environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        True if env var is "1", "true", "yes" (case insensitive)
    """
    value = os.environ.get(key, "").lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    return default


def get_env_int(key: str, default: int) -> int:
    """
    Get an integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value or default
    """
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default
