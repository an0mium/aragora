from __future__ import annotations

import argparse
import subprocess
import uuid


def _run(
    cmd: list[str],
    *,
    cwd: str,
) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "Claude session step failed.")


def _claude_cmd(
    *,
    claude_path: str,
    prompt: str,
    model: str | None,
    session_id: str,
    resume: bool,
    allow_dangerous: bool,
) -> list[str]:
    cmd = [claude_path, "-p", prompt]
    if resume:
        cmd.extend(["--resume", session_id])
    else:
        cmd.extend(["--session-id", session_id])
    if model:
        cmd.extend(["--model", model])
    if allow_dangerous:
        cmd.append("--dangerously-skip-permissions")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a multi-turn Claude session.")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--claude-path", default="claude")
    parser.add_argument("--model")
    parser.add_argument("--session-id")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    args = parser.parse_args()

    prompt_path = args.prompt_file
    try:
        with open(prompt_path, encoding="utf-8") as handle:
            base_prompt = handle.read().strip()
    except OSError as exc:
        raise SystemExit(f"Failed to read prompt file: {exc}") from exc

    session_id = args.session_id or str(uuid.uuid4())
    allow_dangerous = bool(args.dangerously_skip_permissions)

    steps = [
        "Phase 1: Read the task and context. Provide a short plan only. "
        "Do not edit files yet.\n\n"
        f"{base_prompt}",
        "Phase 2: Implement the changes now. Use tools, edit files, and keep scope tight.",
        "Phase 3: Run the expected tests. If failures occur, fix them.",
        "Phase 4: Commit changes with a clear message. Summarize what was done.",
    ]

    cwd = "."
    for index, step in enumerate(steps):
        cmd = _claude_cmd(
            claude_path=args.claude_path,
            prompt=step,
            model=args.model,
            session_id=session_id,
            resume=index > 0,
            allow_dangerous=allow_dangerous,
        )
        _run(cmd, cwd=cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
