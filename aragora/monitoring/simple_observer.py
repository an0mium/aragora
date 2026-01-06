"""
Simple system observer for tracking agent failures and loop_id issues.

Provides basic monitoring without complex ML or learning.
"""
import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

class SimpleObserver:
    """
    Basic monitoring for agent attempts and failures.

    Tracks completion rates, null bytes, and loop_id issues.
    """
    def __init__(self, log_file: str = "system_health.log"):
        self.metrics = {}
        self.log_file = log_file
        self.logger = logging.getLogger('aragora_observer')
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def record_agent_attempt(self, agent_name: str, timeout_seconds: float) -> str:
        """Record start of agent generation attempt."""
        import uuid
        attempt_id = str(uuid.uuid4())
        self.metrics[attempt_id] = {
            'agent': agent_name,
            'start_time': time.time(),
            'timeout': timeout_seconds,
            'status': 'in_progress'
        }
        self.logger.info(f"Agent attempt: {agent_name}")
        return attempt_id

    def record_agent_completion(self, attempt_id: str, output: Any, error: Optional[Exception] = None):
        """Record completion of agent attempt."""
        if attempt_id not in self.metrics:
            return

        duration = time.time() - self.metrics[attempt_id]['start_time']
        output_str = str(output) if output else ""

        self.metrics[attempt_id].update({
            'end_time': time.time(),
            'duration': duration,
            'status': 'success' if not error else 'failed',
            'has_null_bytes': '\x00' in output_str,
            'output_length': len(output_str),
            'error': str(error) if error else None
        })

        if error:
            self.logger.error(f"Agent {self.metrics[attempt_id]['agent']} failed: {error}")
        if '\x00' in output_str:
            self.logger.warning(f"Null bytes in {self.metrics[attempt_id]['agent']} output")

    def record_loop_id_issue(self, ws_id: str, has_loop_id: bool, source: str):
        """Track loop_id routing issues."""
        status = "present" if has_loop_id else "missing"
        self.logger.info(f"WebSocket {ws_id}: loop_id {status} from {source}")

    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        completed = [m for m in self.metrics.values() if m.get('status') != 'in_progress']
        if not completed:
            return 0.0
        failed = [m for m in completed if m['status'] == 'failed']
        return len(failed) / len(completed)

    def get_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        completed = [m for m in self.metrics.values() if m.get('status') != 'in_progress']
        if not completed:
            return {"error": "No data available"}

        null_byte_count = sum(1 for m in completed if m.get('has_null_bytes', False))
        timeout_count = sum(1 for m in completed if m.get('duration', 0) > m.get('timeout', 999))

        return {
            "total_attempts": len(completed),
            "failure_rate": self.get_failure_rate(),
            "null_byte_incidents": null_byte_count,
            "timeout_incidents": timeout_count
        }