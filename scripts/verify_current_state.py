"""
IMMEDIATE SYSTEM STATE VERIFICATION
Run this BEFORE implementing any changes.
"""

import os
import subprocess
import sys


def check_orchestrator_sanitization():
    """Verify if sanitization already exists"""
    try:
        with open("aragora/debate/orchestrator.py", "r") as f:
            content = f.read()

        findings = []
        # Check for existing sanitization patterns
        if "sanitize" in content.lower():
            findings.append("Sanitization function exists")
        if "\\x00" in content and "replace" in content:
            findings.append("Null byte handling found")
        if "clean" in content.lower() and "output" in content.lower():
            findings.append("Output cleaning present")
        # Check for _sanitize_agent_output method specifically
        if "_sanitize_agent_output" in content:
            findings.append("_sanitize_agent_output method found")
            # Check if it's actually used
            if "self._sanitize_agent_output(" in content:
                findings.append("_sanitize_agent_output is used in code")

        return findings if findings else ["No explicit sanitization found"]
    except Exception as e:
        return [f"Cannot read orchestrator.py: {e}"]


def check_recent_fixes():
    """Check if recent commits addressed issues"""
    try:
        # Check last 5 commits
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"], capture_output=True, text=True, cwd="."
        )
        if "bb1cd6e" in result.stdout:
            return "Recent reliability commit found"
        return "No recent reliability commits"
    except:
        return "Git check failed"


def check_fork_bridge():
    """Check if fork bridge is implemented and integrated"""
    findings = []
    try:
        # Check if fork_handler.py exists
        if os.path.exists("aragora/server/fork_handler.py"):
            findings.append("fork_handler.py exists")
            with open("aragora/server/fork_handler.py", "r") as f:
                content = f.read()
                if "handle_start_fork" in content:
                    findings.append("ForkBridgeHandler.handle_start_fork method exists")
        else:
            findings.append("fork_handler.py missing")

        # Check if integrated in stream package (refactored from stream.py)
        stream_modules = [
            "aragora/server/stream/servers.py",
            "aragora/server/stream/message_handlers.py",
        ]
        fork_integrated = False
        for module_path in stream_modules:
            try:
                with open(module_path, "r") as f:
                    if "ForkBridgeHandler" in f.read():
                        fork_integrated = True
                        findings.append(f"ForkBridgeHandler integrated in {module_path}")
                        break
            except FileNotFoundError:
                continue
        if not fork_integrated:
            findings.append("ForkBridgeHandler not found in stream package")

    except Exception as e:
        findings.append(f"Cannot check fork bridge: {e}")

    return findings


def run_verification():
    """Run all checks and print results"""
    print("=" * 60)
    print("SYSTEM STATE VERIFICATION")
    print("=" * 60)

    sanitization = check_orchestrator_sanitization()
    recent = check_recent_fixes()
    fork_bridge = check_fork_bridge()

    print(f"\n1. SANITIZATION STATUS:")
    for finding in sanitization:
        print(f"   • {finding}")

    print(f"\n2. RECENT COMMITS:")
    print(f"   • {recent}")

    print(f"\n3. FORK BRIDGE STATUS:")
    for finding in fork_bridge:
        print(f"   • {finding}")

    print(f"\n4. RECOMMENDATION:")
    needs_sanitization = any("No explicit" in f for f in sanitization)
    fork_integrated = any("integrated" in f and "not" not in f for f in fork_bridge)

    if needs_sanitization:
        print("   ⚠️  SANITIZATION REQUIRED: Implement minimal sanitization")
    else:
        print("   ✅ SANITIZATION EXISTS: No additional sanitization needed")

    if not fork_integrated:
        print("   ⚠️  FORK BRIDGE INCOMPLETE: Integrate ForkBridgeHandler into stream.py")
    else:
        print("   ✅ FORK BRIDGE COMPLETE: No additional fork work needed")

    return sanitization, recent, fork_bridge


if __name__ == "__main__":
    run_verification()
