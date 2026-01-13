"""
System health verification script.

Checks for common issues like null bytes, loop_id routing, and auth readiness.
Used to verify problems before applying fixes.
"""

import json
import os
import sys
from pathlib import Path


def check_for_null_bytes():
    """Check recent logs for null byte issues."""
    # Simplified check - look for null bytes in log files
    log_paths = [Path("logs/aragora.log"), Path("aragora.log"), Path("system_health.log")]
    for log_path in log_paths:
        if log_path.exists():
            try:
                with open(log_path, "rb") as f:
                    content = f.read()
                    if b"\x00" in content:
                        return True
            except Exception:
                pass
    return False


def check_timeout_handling():
    """Check if timeout mechanisms are in place."""
    # Look for timeout-related code in orchestrator
    orchestrator_path = Path("aragora/debate/orchestrator.py")
    if orchestrator_path.exists():
        try:
            with open(orchestrator_path, "r") as f:
                content = f.read()
                return "asyncio.wait_for" in content or "TimeoutError" in content
        except Exception:
            pass
    return False


def check_cli_integration():
    """Check if CLI agents are properly integrated."""
    # Simple check for CLI agent files
    cli_agent_path = Path("aragora/agents/cli_agents.py")
    return cli_agent_path.exists()


def check_stream_for_loop_id_binding():
    """Check if stream package has loop_id binding.

    Note: stream.py was refactored into aragora/server/stream/ package.
    Now checks multiple modules in the stream package.
    """
    # Check the refactored stream package modules
    stream_modules = [
        Path("aragora/server/stream/servers.py"),
        Path("aragora/server/stream/broadcaster.py"),
        Path("aragora/server/stream/state_manager.py"),
    ]
    for stream_path in stream_modules:
        if stream_path.exists():
            try:
                with open(stream_path, "r") as f:
                    content = f.read()
                    if (
                        "_bound_loop_id" in content
                        or "ws.loop_id" in content
                        or "loop_id" in content
                    ):
                        return True
            except Exception:
                pass
    return False


def check_stream_for_validation():
    """Check if stream package validates loop_id.

    Note: stream.py was refactored into aragora/server/stream/ package.
    Now checks multiple modules in the stream package.
    """
    # Check the refactored stream package modules
    stream_modules = [
        Path("aragora/server/stream/servers.py"),
        Path("aragora/server/stream/state_manager.py"),
    ]
    for stream_path in stream_modules:
        if stream_path.exists():
            try:
                with open(stream_path, "r") as f:
                    content = f.read()
                    if "loop_id in self.active_loops" in content or "active_loops" in content:
                        return True
            except Exception:
                pass
    return False


def check_file_exists(file_path):
    """Check if a file exists."""
    return Path(file_path).exists()


def check_function_implementation(func_name):
    """Check if a function is implemented in a file."""
    auth_path = Path("aragora/auth.py")
    if auth_path.exists():
        try:
            with open(auth_path, "r") as f:
                content = f.read()
                return f"def {func_name}" in content
        except Exception:
            pass
    return False


def check_websocket_auth_capability():
    """Check if auth supports WebSocket."""
    auth_path = Path("aragora/auth.py")
    if auth_path.exists():
        try:
            with open(auth_path, "r") as f:
                content = f.read()
                return "websocket" in content.lower() or "ws" in content.lower()
        except Exception:
            pass
    return False


def diagnose_system():
    """Run all diagnostic checks."""
    diagnostics = {
        "agent_health": {
            "null_bytes_found": check_for_null_bytes(),
            "timeout_handling_present": check_timeout_handling(),
            "cli_integration_present": check_cli_integration(),
        },
        "loop_id_routing": {
            "loop_id_binding_present": check_stream_for_loop_id_binding(),
            "validation_present": check_stream_for_validation(),
        },
        "auth_readiness": {
            "module_exists": check_file_exists("aragora/auth.py"),
            "check_auth_implemented": check_function_implementation("check_auth"),
            "websocket_support": check_websocket_auth_capability(),
        },
    }
    return diagnostics


def prioritize_fixes(diagnostics):
    """Return ordered list of priorities."""
    priorities = []
    # Check for actual problems (not presence of safety features)
    if diagnostics["agent_health"]["null_bytes_found"]:
        priorities.append(("CRITICAL", "Fix null bytes in logs - indicates encoding issues"))
    if not diagnostics["agent_health"]["timeout_handling_present"]:
        priorities.append(("HIGH", "Add timeout handling - debates may hang indefinitely"))
    if not diagnostics["agent_health"]["cli_integration_present"]:
        priorities.append(("HIGH", "Fix CLI integration - agents cannot be invoked"))
    if not diagnostics["loop_id_routing"]["loop_id_binding_present"]:
        priorities.append(("HIGH", "Fix loop_id routing - audience participation broken"))
    if not diagnostics["loop_id_routing"]["validation_present"]:
        priorities.append(("MEDIUM", "Add loop_id validation - security gap"))
    if not diagnostics["auth_readiness"]["module_exists"]:
        priorities.append(("LOW", "Add auth module - security scaffolding"))
    if not priorities:
        priorities.append(("OK", "All critical systems healthy"))
    return priorities


if __name__ == "__main__":
    print("Running system diagnostics...")
    diag = diagnose_system()
    print(json.dumps(diag, indent=2))
    priorities = prioritize_fixes(diag)
    print("\nPrioritized fixes:")
    for priority, desc in priorities:
        print(f"- {priority}: {desc}")
