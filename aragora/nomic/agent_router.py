"""
Agent routing for autonomous orchestration.

Routes subtasks to appropriate agents based on domain, track, and task complexity.
"""

from __future__ import annotations

from aragora.nomic.task_decomposer import SubTask
from aragora.nomic.types import (
    AGENTS_WITH_CODING_HARNESS,
    DEFAULT_TRACK_CONFIGS,
    KILOCODE_PROVIDER_MAPPING,
    AgentAssignment,
    Track,
    TrackConfig,
)


class AgentRouter:
    """
    Routes subtasks to appropriate agents based on domain and track.

    Uses heuristics to determine:
    1. Which track owns a subtask (based on file patterns)
    2. Which agent type is best suited (based on task complexity)
    """

    def __init__(self, track_configs: dict[Track, TrackConfig] | None = None):
        self.track_configs = track_configs or DEFAULT_TRACK_CONFIGS
        self._file_to_track_cache: dict[str, Track] = {}

    def determine_track(self, subtask: SubTask) -> Track:
        """Determine which track should handle a subtask."""
        # Check file scope first
        for file_path in subtask.file_scope:
            track = self._file_to_track(file_path)
            if track:
                return track

        # Infer from task description
        description_lower = subtask.description.lower()
        title_lower = subtask.title.lower()
        combined = f"{title_lower} {description_lower}"

        # Track detection patterns
        patterns = {
            Track.SME: ["ui", "frontend", "user", "dashboard", "workspace", "admin"],
            Track.DEVELOPER: ["sdk", "api", "documentation", "docs", "client"],
            Track.SELF_HOSTED: [
                "docker",
                "deploy",
                "backup",
                "restore",
                "ops",
                "kubernetes",
            ],
            Track.QA: ["test", "e2e", "ci", "coverage", "quality", "playwright"],
            Track.CORE: ["debate", "consensus", "arena", "agent", "memory"],
            Track.SECURITY: [
                "security",
                "vuln",
                "auth",
                "encrypt",
                "secret",
                "owasp",
                "xss",
                "csrf",
                "injection",
            ],
        }

        for track, keywords in patterns.items():
            if any(kw in combined for kw in keywords):
                return track

        # Default to developer track for unclassified tasks
        return Track.DEVELOPER

    def _file_to_track(self, file_path: str) -> Track | None:
        """Map a file path to its owning track."""
        if file_path in self._file_to_track_cache:
            return self._file_to_track_cache[file_path]

        for track, config in self.track_configs.items():
            for folder in config.folders:
                if file_path.startswith(folder):
                    self._file_to_track_cache[file_path] = track
                    return track

        return None

    def select_agent_type(self, subtask: SubTask, track: Track) -> str:
        """Select the best agent type for a subtask."""
        config = self.track_configs.get(track, DEFAULT_TRACK_CONFIGS[Track.DEVELOPER])

        if not config.agent_types:
            return "claude"  # Default

        # High complexity -> Claude (better reasoning)
        if subtask.estimated_complexity == "high":
            return "claude"

        # Code generation -> prefer Codex
        if "implement" in subtask.title.lower() or "code" in subtask.description.lower():
            if "codex" in config.agent_types:
                return "codex"

        # Default to first preferred agent
        return config.agent_types[0]

    def get_coding_harness(
        self,
        agent_type: str,
        track: Track,
    ) -> dict[str, str] | None:
        """Determine the coding harness to use for an agent.

        For agents with native coding harnesses (claude, codex), returns None.
        For other agents (gemini, grok, etc.), returns KiloCode configuration
        if the track allows it.

        Args:
            agent_type: The selected agent type
            track: The development track

        Returns:
            None if agent has native harness, otherwise dict with:
            - harness: "kilocode"
            - provider_id: The KiloCode provider to use
            - mode: The KiloCode mode (code, architect, etc.)
        """
        # Agents with native coding harnesses don't need KiloCode
        if agent_type in AGENTS_WITH_CODING_HARNESS:
            return None

        # Check if track allows KiloCode harness
        config = self.track_configs.get(track, DEFAULT_TRACK_CONFIGS[Track.DEVELOPER])
        if not config.use_kilocode_harness:
            return None

        # Get KiloCode provider for this agent type
        provider_id = KILOCODE_PROVIDER_MAPPING.get(agent_type)
        if not provider_id:
            # No KiloCode mapping for this agent
            return None

        return {
            "harness": "kilocode",
            "provider_id": provider_id,
            "mode": "code",  # Use code mode for implementation tasks
        }

    def check_conflicts(
        self,
        subtask: SubTask,
        active_assignments: list[AgentAssignment],
    ) -> list[str]:
        """Check for potential conflicts with active assignments."""
        conflicts = []

        for assignment in active_assignments:
            if assignment.status != "running":
                continue

            # Check file overlap
            active_files = set(assignment.subtask.file_scope)
            new_files = set(subtask.file_scope)
            overlap = active_files & new_files

            if overlap:
                conflicts.append(f"File conflict with {assignment.subtask.id}: {overlap}")

            # Check track overlap (some tracks shouldn't run in parallel)
            if assignment.track == Track.CORE and self.determine_track(subtask) == Track.CORE:
                conflicts.append("Core track conflict: only one core task at a time")

        return conflicts
