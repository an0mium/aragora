"""
Resource isolation enforcement for Agent Fabric.

Provides resource monitoring and isolation enforcement for agents including
memory limits, CPU limits, filesystem sandboxing, and network egress control.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import IsolationConfig

logger = logging.getLogger(__name__)

# Optional psutil import for resource monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


@dataclass
class ResourceUsage:
    """Current resource usage for an agent."""

    agent_id: str
    timestamp: datetime
    memory_mb: float
    cpu_percent: float
    disk_mb: float | None = None
    network_bytes: int | None = None


@dataclass
class ResourceViolation:
    """Details of a resource limit violation."""

    resource_type: str  # "memory", "cpu", "disk", "network"
    limit: float
    actual: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""


class ResourceMonitor:
    """
    Monitor resource usage for agents.

    Tracks CPU, memory, disk, and network usage for registered agent processes.
    Requires psutil for full functionality; operates in degraded mode without it.
    """

    def __init__(self) -> None:
        self._process_cache: dict[str, int] = {}  # agent_id -> pid
        self._usage_history: dict[str, list[ResourceUsage]] = {}
        self._max_history_per_agent = 100

    def register_process(self, agent_id: str, pid: int) -> None:
        """Register a process ID for an agent."""
        self._process_cache[agent_id] = pid
        if agent_id not in self._usage_history:
            self._usage_history[agent_id] = []
        logger.debug(f"Registered process {pid} for agent {agent_id}")

    def unregister_process(self, agent_id: str) -> None:
        """Unregister an agent's process."""
        self._process_cache.pop(agent_id, None)
        logger.debug(f"Unregistered process for agent {agent_id}")

    def get_usage(self, agent_id: str) -> ResourceUsage | None:
        """
        Get current resource usage for an agent.

        Returns None if the agent is not registered or psutil is unavailable.
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, cannot get resource usage")
            return None

        pid = self._process_cache.get(agent_id)
        if pid is None:
            return None

        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)

            # Get disk usage if filesystem root is set
            disk_mb = None
            try:
                io_counters = process.io_counters()
                disk_mb = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)
            except (psutil.AccessDenied, AttributeError) as e:
                logger.debug("Metric collection failed: %s", e)

            # Get network usage
            network_bytes = None
            try:
                net_io = psutil.net_io_counters()
                network_bytes = net_io.bytes_sent + net_io.bytes_recv
            except (psutil.AccessDenied, AttributeError) as e:
                logger.debug("Metric collection failed: %s", e)

            usage = ResourceUsage(
                agent_id=agent_id,
                timestamp=datetime.now(timezone.utc),
                memory_mb=memory_info.rss / (1024 * 1024),
                cpu_percent=cpu_percent,
                disk_mb=disk_mb,
                network_bytes=network_bytes,
            )

            # Store in history
            self._record_usage(agent_id, usage)
            return usage

        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} for agent {agent_id} no longer exists")
            self.unregister_process(agent_id)
            return None
        except psutil.AccessDenied:
            logger.warning(f"Access denied to process {pid} for agent {agent_id}")
            return None

    def _record_usage(self, agent_id: str, usage: ResourceUsage) -> None:
        """Record usage in history, maintaining max size."""
        if agent_id not in self._usage_history:
            self._usage_history[agent_id] = []

        history = self._usage_history[agent_id]
        history.append(usage)

        # Trim to max size
        if len(history) > self._max_history_per_agent:
            self._usage_history[agent_id] = history[-self._max_history_per_agent :]

    def get_usage_history(self, agent_id: str) -> list[ResourceUsage]:
        """Get usage history for an agent."""
        return list(self._usage_history.get(agent_id, []))

    def check_limits(self, agent_id: str, config: IsolationConfig) -> list[str]:
        """
        Check if agent exceeds configured limits.

        Returns a list of violation messages. Empty list means no violations.
        """
        violations: list[str] = []
        usage = self.get_usage(agent_id)

        if usage is None:
            return violations

        # Check memory limit
        if usage.memory_mb > config.memory_mb:
            violations.append(
                f"Memory limit exceeded: {usage.memory_mb:.1f}MB > {config.memory_mb}MB"
            )

        # Check CPU limit (cpu_cores * 100 = max percent for single process)
        max_cpu_percent = config.cpu_cores * 100
        if usage.cpu_percent > max_cpu_percent:
            violations.append(
                f"CPU limit exceeded: {usage.cpu_percent:.1f}% > {max_cpu_percent:.1f}%"
            )

        return violations

    def check_memory_limit(
        self, agent_id: str, config: IsolationConfig
    ) -> ResourceViolation | None:
        """
        Check if agent exceeds memory limit.

        Returns ResourceViolation if limit exceeded, None otherwise.
        """
        usage = self.get_usage(agent_id)
        if usage is None:
            return None

        if usage.memory_mb > config.memory_mb:
            return ResourceViolation(
                resource_type="memory",
                limit=config.memory_mb,
                actual=usage.memory_mb,
                message=f"Memory limit exceeded: {usage.memory_mb:.1f}MB > {config.memory_mb}MB",
            )
        return None

    def check_cpu_limit(self, agent_id: str, config: IsolationConfig) -> ResourceViolation | None:
        """
        Check if agent exceeds CPU limit.

        Returns ResourceViolation if limit exceeded, None otherwise.
        """
        usage = self.get_usage(agent_id)
        if usage is None:
            return None

        max_cpu_percent = config.cpu_cores * 100
        if usage.cpu_percent > max_cpu_percent:
            return ResourceViolation(
                resource_type="cpu",
                limit=max_cpu_percent,
                actual=usage.cpu_percent,
                message=f"CPU limit exceeded: {usage.cpu_percent:.1f}% > {max_cpu_percent:.1f}%",
            )
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            "psutil_available": PSUTIL_AVAILABLE,
            "registered_agents": len(self._process_cache),
            "agents_with_history": len(self._usage_history),
            "total_history_entries": sum(len(h) for h in self._usage_history.values()),
        }


class IsolationEnforcer:
    """
    Enforce isolation policies for agents.

    Manages filesystem sandboxes and validates network egress against allowlists.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "aragora_sandboxes"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sandboxes: dict[str, Path] = {}
        self._resource_monitor = ResourceMonitor()

    @property
    def resource_monitor(self) -> ResourceMonitor:
        """Get the resource monitor instance."""
        return self._resource_monitor

    def create_sandbox(self, agent_id: str, config: IsolationConfig) -> Path:
        """
        Create isolated filesystem sandbox for agent.

        Creates a directory structure with restricted permissions for the agent
        to use as its working directory.
        """
        if agent_id in self._sandboxes:
            logger.warning(f"Sandbox already exists for agent {agent_id}")
            return self._sandboxes[agent_id]

        # Create sandbox directory
        sandbox_path = self.base_dir / agent_id
        sandbox_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (sandbox_path / "tmp").mkdir(exist_ok=True)
        (sandbox_path / "data").mkdir(exist_ok=True)
        (sandbox_path / "logs").mkdir(exist_ok=True)

        # Set restrictive permissions (owner only)
        try:
            os.chmod(sandbox_path, 0o700)
            for subdir in sandbox_path.iterdir():
                if subdir.is_dir():
                    os.chmod(subdir, 0o700)
        except OSError as e:
            logger.warning(f"Could not set permissions on sandbox: {e}")

        self._sandboxes[agent_id] = sandbox_path
        logger.info(f"Created sandbox for agent {agent_id} at {sandbox_path}")
        return sandbox_path

    def get_sandbox(self, agent_id: str) -> Path | None:
        """Get the sandbox path for an agent."""
        return self._sandboxes.get(agent_id)

    def cleanup_sandbox(self, agent_id: str) -> bool:
        """
        Clean up agent's sandbox.

        Removes the sandbox directory and all its contents.
        Returns True if cleanup succeeded, False otherwise.
        """
        sandbox_path = self._sandboxes.pop(agent_id, None)
        if sandbox_path is None:
            logger.debug(f"No sandbox to clean up for agent {agent_id}")
            return False

        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
                logger.info(f"Cleaned up sandbox for agent {agent_id}")
            return True
        except OSError as e:
            logger.error(f"Failed to clean up sandbox for agent {agent_id}: {e}")
            return False

    def validate_network_egress(self, url: str, config: IsolationConfig) -> bool:
        """
        Check if URL is allowed by egress policy.

        Validates the URL against the network_egress allowlist in the config.
        Empty allowlist means all egress is blocked.
        """
        if not config.network_egress:
            # No allowlist = no egress allowed
            logger.debug("Network egress blocked: no allowlist configured")
            return False

        try:
            parsed = urlparse(url)
            host = parsed.netloc or parsed.path.split("/")[0]

            # Remove port if present
            if ":" in host:
                host = host.split(":")[0]

            # Check against allowlist
            for allowed in config.network_egress:
                # Exact match
                if host == allowed:
                    return True
                # Wildcard subdomain match (e.g., *.example.com)
                if allowed.startswith("*.") and host.endswith(allowed[1:]):
                    return True
                # Domain suffix match (e.g., example.com matches api.example.com)
                if host.endswith("." + allowed):
                    return True

            logger.debug(f"Network egress blocked for {host}: not in allowlist")
            return False

        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {e}")
            return False

    def check_network_egress(self, url: str, config: IsolationConfig) -> bool:
        """
        Validate network egress against allowlist.

        Alias for validate_network_egress for API consistency.
        """
        return self.validate_network_egress(url, config)

    def validate_path_access(self, agent_id: str, path: Path, config: IsolationConfig) -> bool:
        """
        Validate that an agent can access a given path.

        Ensures the path is within the agent's sandbox or an allowed location.
        """
        sandbox = self._sandboxes.get(agent_id)
        if sandbox is None:
            # No sandbox = no path access
            return False

        # Resolve to absolute path
        try:
            resolved = path.resolve()
            sandbox_resolved = sandbox.resolve()

            # Check if path is within sandbox
            return resolved.is_relative_to(sandbox_resolved)
        except (ValueError, RuntimeError):
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get enforcer statistics."""
        return {
            "base_dir": str(self.base_dir),
            "active_sandboxes": len(self._sandboxes),
            "sandbox_agents": list(self._sandboxes.keys()),
            "monitor_stats": self._resource_monitor.get_stats(),
        }


# Convenience functions for common operations


def create_enforcer(base_dir: str | Path | None = None) -> IsolationEnforcer:
    """Create an IsolationEnforcer with the specified base directory."""
    path = Path(base_dir) if base_dir else None
    return IsolationEnforcer(base_dir=path)


def check_resource_limits(
    enforcer: IsolationEnforcer, agent_id: str, config: IsolationConfig
) -> list[str]:
    """Check resource limits for an agent using the enforcer's monitor."""
    return enforcer.resource_monitor.check_limits(agent_id, config)
