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
    log_paths = [
        Path("logs/aragora.log"),
        Path("aragora.log"),
        Path("system_health.log")
    ]
    for log_path in log_paths:
        if log_path.exists():
            try:
                with open(log_path, 'rb') as f:
                    content = f.read()
                    if b'\x00' in content:
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
            with open(orchestrator_path, 'r') as f:
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
    """Check if stream.py has loop_id binding."""
    stream_path = Path("aragora/server/stream.py")
    if stream_path.exists():
        try:
            with open(stream_path, 'r') as f:
                content = f.read()
                return "_bound_loop_id" in content or "ws.loop_id" in content
        except Exception:
            pass
    return False

def check_stream_for_validation():
    """Check if stream.py validates loop_id."""
    stream_path = Path("aragora/server/stream.py")
    if stream_path.exists():
        try:
            with open(stream_path, 'r') as f:
                content = f.read()
                return "loop_id in self.active_loops" in content
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
            with open(auth_path, 'r') as f:
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
            with open(auth_path, 'r') as f:
                content = f.read()
                return "websocket" in content.lower() or "ws" in content.lower()
        except Exception:
            pass
    return False

def diagnose_system():
    """Run all diagnostic checks."""
    diagnostics = {
        'agent_failures': {
            'null_bytes': check_for_null_bytes(),
            'timeouts': check_timeout_handling(),
            'cli_errors': check_cli_integration()
        },
        'loop_id_routing': {
            'missing_in_stream': not check_stream_for_loop_id_binding(),
            'validation_present': check_stream_for_validation()
        },
        'auth_readiness': {
            'module_exists': check_file_exists('aragora/auth.py'),
            'check_auth_implemented': check_function_implementation('check_auth'),
            'websocket_support': check_websocket_auth_capability()
        }
    }
    return diagnostics

def prioritize_fixes(diagnostics):
    """Return ordered list of priorities."""
    priorities = []
    if diagnostics['agent_failures']['null_bytes'] or diagnostics['agent_failures']['timeouts']:
        priorities.append(("CRITICAL", "Fix agent failure modes - debates cannot complete"))
    if diagnostics['loop_id_routing']['missing_in_stream']:
        priorities.append(("HIGH", "Fix loop_id routing - audience participation broken"))
    if (diagnostics['auth_readiness']['module_exists'] and
        diagnostics['auth_readiness']['check_auth_implemented']):
        priorities.append(("MEDIUM", "Wire authentication - security scaffolding"))
    return priorities

if __name__ == "__main__":
    print("Running system diagnostics...")
    diag = diagnose_system()
    print(json.dumps(diag, indent=2))
    priorities = prioritize_fixes(diag)
    print("\nPrioritized fixes:")
    for priority, desc in priorities:
        print(f"- {priority}: {desc}")