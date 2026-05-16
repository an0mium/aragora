"""Read-only inspectors for Codex Desktop local state.

This package surfaces Codex Desktop session/thread activity (rollout JSONL +
SQLite state) into aragora as a read-only operator tool. Nothing under this
package writes to ``~/.codex/`` or makes network/AI-key calls.
"""
