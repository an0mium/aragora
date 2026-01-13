"""
COMPREHENSIVE SYSTEM AUDIT
Establishes ground truth before any implementation.
Checks recent commits, sanitization usage, and fork infrastructure.
"""

import subprocess
import json


def audit_system():
    report = {
        "sanitization_defined": False,
        "sanitization_used": False,
        "fork_handler_present": False,
        "nomic_fork_support": False,
        "frontend_ready": False,
        "recent_commits": [],
    }

    # Check recent commits for existing fixes
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"], capture_output=True, text=True, check=True
        )
        recent = ["bb1cd6e", "09bc1d5", "f0c5950"]
        for commit in recent:
            if commit in result.stdout:
                report["recent_commits"].append(commit)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Git command failed: {e}")

    # Check orchestrator sanitization
    try:
        with open("aragora/debate/orchestrator.py", "r") as f:
            content = f.read()
            report["sanitization_defined"] = "def _sanitize" in content.lower()
            report["sanitization_used"] = "._sanitize" in content
    except FileNotFoundError:
        print("File not found: aragora/debate/orchestrator.py")
    except Exception as e:
        print(f"Error reading aragora/debate/orchestrator.py: {e}")

    # Check stream package for fork handler (refactored from stream.py)
    try:
        stream_modules = [
            "aragora/server/stream/servers.py",
            "aragora/server/stream/message_handlers.py",
        ]
        fork_handler_found = False
        for module_path in stream_modules:
            try:
                with open(module_path, "r") as f:
                    if "_handle_start_fork" in f.read():
                        fork_handler_found = True
                        break
            except FileNotFoundError:
                continue
        report["fork_handler_present"] = fork_handler_found
    except Exception as e:
        print(f"Error checking stream package: {e}")

    # Check frontend
    try:
        with open("frontend/src/components/ReplayBrowser.tsx", "r") as f:
            report["frontend_ready"] = "start_fork" in f.read()
    except FileNotFoundError:
        # It's okay if the frontend file doesn't exist, just report it.
        pass
    except Exception as e:
        print(f"Error reading frontend/src/components/ReplayBrowser.tsx: {e}")

    return report


if __name__ == "__main__":
    audit_report = audit_system()
    print(json.dumps(audit_report, indent=4))
