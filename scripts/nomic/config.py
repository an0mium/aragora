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


# =============================================================================
# TESTFIXER FLAGS - Automated test repair loop integration
# =============================================================================

# Enable TestFixer integration in the nomic loop (default OFF)
NOMIC_TESTFIXER_ENABLED = os.environ.get("NOMIC_TESTFIXER_ENABLED", "0") == "1"

# Test command to run inside TestFixer
NOMIC_TESTFIXER_TEST_COMMAND = os.environ.get(
    "NOMIC_TESTFIXER_TEST_COMMAND", "pytest tests/ -q --maxfail=1"
)

# Per-test run timeout (seconds)
NOMIC_TESTFIXER_TEST_TIMEOUT = int(os.environ.get("NOMIC_TESTFIXER_TEST_TIMEOUT", "600"))

# Max iterations for the TestFixer loop
NOMIC_TESTFIXER_MAX_ITERATIONS = int(os.environ.get("NOMIC_TESTFIXER_MAX_ITERATIONS", "5"))

# Max repeated same failure before stopping
NOMIC_TESTFIXER_MAX_SAME_FAILURE = int(os.environ.get("NOMIC_TESTFIXER_MAX_SAME_FAILURE", "3"))

# Confidence thresholds
NOMIC_TESTFIXER_MIN_CONFIDENCE = float(os.environ.get("NOMIC_TESTFIXER_MIN_CONFIDENCE", "0.5"))
NOMIC_TESTFIXER_MIN_AUTO_CONFIDENCE = float(
    os.environ.get("NOMIC_TESTFIXER_MIN_AUTO_CONFIDENCE", "0.7")
)

# Require consensus across generators
NOMIC_TESTFIXER_REQUIRE_CONSENSUS = os.environ.get("NOMIC_TESTFIXER_REQUIRE_CONSENSUS", "0") == "1"

# Require manual approval before applying fixes
NOMIC_TESTFIXER_REQUIRE_APPROVAL = os.environ.get("NOMIC_TESTFIXER_REQUIRE_APPROVAL", "0") == "1"

# Revert failed fixes
NOMIC_TESTFIXER_REVERT_ON_FAILURE = os.environ.get("NOMIC_TESTFIXER_REVERT_ON_FAILURE", "1") == "1"

# Stop on first successful fix
NOMIC_TESTFIXER_STOP_ON_FIRST_SUCCESS = (
    os.environ.get("NOMIC_TESTFIXER_STOP_ON_FIRST_SUCCESS", "0") == "1"
)

# Agents to use for fix generation (comma-separated)
NOMIC_TESTFIXER_AGENTS = os.environ.get("NOMIC_TESTFIXER_AGENTS", "codex,claude")

# LLM analyzer integration
NOMIC_TESTFIXER_USE_LLM_ANALYZER = os.environ.get("NOMIC_TESTFIXER_USE_LLM_ANALYZER", "0") == "1"
NOMIC_TESTFIXER_ANALYSIS_AGENTS = os.environ.get("NOMIC_TESTFIXER_ANALYSIS_AGENTS", "")
NOMIC_TESTFIXER_ANALYSIS_REQUIRE_CONSENSUS = (
    os.environ.get("NOMIC_TESTFIXER_ANALYSIS_REQUIRE_CONSENSUS", "0") == "1"
)
NOMIC_TESTFIXER_ANALYSIS_CONSENSUS_THRESHOLD = float(
    os.environ.get("NOMIC_TESTFIXER_ANALYSIS_CONSENSUS_THRESHOLD", "0.7")
)

# Arena validator integration
NOMIC_TESTFIXER_ARENA_VALIDATE = os.environ.get("NOMIC_TESTFIXER_ARENA_VALIDATE", "0") == "1"
NOMIC_TESTFIXER_ARENA_AGENTS = os.environ.get("NOMIC_TESTFIXER_ARENA_AGENTS", "")
NOMIC_TESTFIXER_ARENA_ROUNDS = int(os.environ.get("NOMIC_TESTFIXER_ARENA_ROUNDS", "2"))
NOMIC_TESTFIXER_ARENA_MIN_CONFIDENCE = float(
    os.environ.get("NOMIC_TESTFIXER_ARENA_MIN_CONFIDENCE", "0.6")
)
NOMIC_TESTFIXER_ARENA_REQUIRE_CONSENSUS = (
    os.environ.get("NOMIC_TESTFIXER_ARENA_REQUIRE_CONSENSUS", "0") == "1"
)
NOMIC_TESTFIXER_ARENA_CONSENSUS_THRESHOLD = float(
    os.environ.get("NOMIC_TESTFIXER_ARENA_CONSENSUS_THRESHOLD", "0.7")
)

# Red team validator integration
NOMIC_TESTFIXER_REDTEAM_VALIDATE = os.environ.get("NOMIC_TESTFIXER_REDTEAM_VALIDATE", "0") == "1"
NOMIC_TESTFIXER_REDTEAM_ATTACKERS = os.environ.get("NOMIC_TESTFIXER_REDTEAM_ATTACKERS", "")
NOMIC_TESTFIXER_REDTEAM_DEFENDER = os.environ.get("NOMIC_TESTFIXER_REDTEAM_DEFENDER", "")
NOMIC_TESTFIXER_REDTEAM_ROUNDS = int(os.environ.get("NOMIC_TESTFIXER_REDTEAM_ROUNDS", "2"))
NOMIC_TESTFIXER_REDTEAM_ATTACKS_PER_ROUND = int(
    os.environ.get("NOMIC_TESTFIXER_REDTEAM_ATTACKS_PER_ROUND", "3")
)
NOMIC_TESTFIXER_REDTEAM_MIN_ROBUSTNESS = float(
    os.environ.get("NOMIC_TESTFIXER_REDTEAM_MIN_ROBUSTNESS", "0.6")
)

# Pattern learning
NOMIC_TESTFIXER_PATTERN_LEARNING = os.environ.get("NOMIC_TESTFIXER_PATTERN_LEARNING", "0") == "1"
NOMIC_TESTFIXER_PATTERN_STORE = os.environ.get("NOMIC_TESTFIXER_PATTERN_STORE", "")
NOMIC_TESTFIXER_GENERATION_TIMEOUT = float(
    os.environ.get("NOMIC_TESTFIXER_GENERATION_TIMEOUT", "600")
)
NOMIC_TESTFIXER_CRITIQUE_TIMEOUT = float(os.environ.get("NOMIC_TESTFIXER_CRITIQUE_TIMEOUT", "300"))

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
