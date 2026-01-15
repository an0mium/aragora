"""
Simple observer for monitoring agent execution and system health.

Tracks agent attempts, completions, failures, and loop_id issues
for operational visibility.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional


class SimpleObserver:
    """Observer for monitoring agent execution and system health.

    Tracks:
    - Agent attempts and completions
    - Failure rates
    - Null byte occurrences in output
    - Loop ID issues in WebSocket connections

    Example
    -------
    >>> observer = SimpleObserver()
    >>> attempt_id = observer.record_agent_attempt("claude", timeout=30.0)
    >>> observer.record_agent_completion(attempt_id, output="Result")
    >>> print(observer.get_failure_rate())
    0.0
    """

    def __init__(self, log_file: str = "system_health.log") -> None:
        """Initialize the observer.

        Parameters
        ----------
        log_file : str
            Path to the log file for health events.
        """
        self._log_file = log_file
        self._metrics: Dict[str, Dict[str, Any]] = {}

        # Set up logger with consistent name
        self._logger = logging.getLogger("aragora_observer")
        # Only add handlers if none exist to avoid duplicates
        if not self._logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    @property
    def log_file(self) -> str:
        """Get the log file path.

        Returns
        -------
        str
            Path to the log file.
        """
        return self._log_file

    @property
    def metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics dictionary.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping attempt IDs to their metrics.
        """
        return self._metrics

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance.

        Returns
        -------
        logging.Logger
            Logger for health events.
        """
        return self._logger

    def record_agent_attempt(self, agent: str, timeout: float) -> str:
        """Record the start of an agent attempt.

        Parameters
        ----------
        agent : str
            Name of the agent being invoked.
        timeout : float
            Timeout value for this attempt.

        Returns
        -------
        str
            Unique identifier for this attempt.
        """
        attempt_id = str(uuid.uuid4())
        self._metrics[attempt_id] = {
            "agent": agent,
            "timeout": timeout,
            "start_time": time.time(),
            "status": "in_progress",
        }
        self._logger.info(f"Agent attempt started: {agent} (timeout={timeout}s)")
        return attempt_id

    def record_agent_completion(
        self,
        attempt_id: str,
        output: Optional[str],
        error: Optional[Exception] = None,
    ) -> None:
        """Record the completion of an agent attempt.

        Parameters
        ----------
        attempt_id : str
            The attempt ID returned from record_agent_attempt.
        output : Optional[str]
            The output produced by the agent.
        error : Optional[Exception]
            Any exception that occurred during execution.
        """
        if attempt_id not in self._metrics:
            self._logger.warning(f"Unknown attempt ID: {attempt_id}")
            return

        record = self._metrics[attempt_id]
        record["end_time"] = time.time()
        record["duration"] = record["end_time"] - record["start_time"]
        record["output_length"] = len(output) if output else 0
        record["has_null_bytes"] = "\x00" in output if output else False

        if record["has_null_bytes"]:
            self._logger.warning(f"Null bytes detected in output for agent {record['agent']}")

        if error:
            record["status"] = "failed"
            record["error"] = str(error)
            self._logger.error(f"Agent {record['agent']} failed: {error}")
        else:
            record["status"] = "success"
            self._logger.info(
                f"Agent {record['agent']} completed in {record['duration']:.2f}s"
            )

    def record_loop_id_issue(self, ws_id: str, present: bool, source: str) -> None:
        """Record a loop_id issue in WebSocket communication.

        Parameters
        ----------
        ws_id : str
            WebSocket connection identifier.
        present : bool
            Whether loop_id was present.
        source : str
            Source of the issue (e.g., "client", "server").
        """
        status = "present" if present else "missing"
        self._logger.info(
            f"Loop ID {status} for {ws_id} from {source}"
        )

    def get_failure_rate(self) -> float:
        """Calculate the current failure rate.

        Returns
        -------
        float
            Failure rate as a fraction (0.0 to 1.0).
        """
        completed = [m for m in self._metrics.values() if m["status"] != "in_progress"]
        if not completed:
            return 0.0
        failed = sum(1 for m in completed if m["status"] == "failed")
        return failed / len(completed)

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report.

        Returns
        -------
        Dict[str, Any]
            Report containing aggregated metrics and statistics.
            Returns {"error": "No data..."} if no completed attempts.
        """
        completed = [m for m in self._metrics.values() if m["status"] != "in_progress"]

        if not completed:
            return {"error": "No data available - no completed attempts"}

        failed = [m for m in completed if m["status"] == "failed"]
        null_byte_incidents = sum(1 for m in completed if m.get("has_null_bytes", False))

        # Count timeout incidents (duration > timeout)
        timeout_incidents = sum(
            1 for m in completed
            if m.get("duration", 0) > m.get("timeout", float("inf"))
        )

        return {
            "total_attempts": len(completed),
            "failed_attempts": len(failed),
            "failure_rate": self.get_failure_rate(),
            "null_byte_incidents": null_byte_incidents,
            "timeout_incidents": timeout_incidents,
        }


__all__ = ["SimpleObserver"]
