from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"


def _run(
    args: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, env=env)


def _init_repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    _run(["git", "init", "--bare", str(origin)], cwd=tmp_path)

    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.name", "Test User"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    _run(["git", "remote", "add", "origin", str(origin)], cwd=repo)
    _run(["git", "push", "-u", "origin", "main"], cwd=repo)
    _run(["git", "switch", "-c", "codex/fix-open-pr-auth"], cwd=repo)
    return repo


def _stub_gh(path: Path, *, token_available: bool) -> None:
    token_block = "echo fake-token\nexit 0" if token_available else "exit 1"
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cmd="${1:-}"',
                'subcmd="${2:-}"',
                'if [[ "$cmd" == "auth" && "$subcmd" == "status" ]]; then',
                '  echo "api unavailable" >&2',
                "  exit 1",
                "fi",
                'if [[ "$cmd" == "auth" && "$subcmd" == "token" ]]; then',
                f"  {token_block}",
                "fi",
                'if [[ "$cmd" == "pr" && "$subcmd" == "list" ]]; then',
                '  if [[ " $* " == *" --jq "* ]]; then',
                '    echo ""',
                "  else",
                "    echo '[]'",
                "  fi",
                "  exit 0",
                "fi",
                'if [[ "$cmd" == "pr" && "$subcmd" == "create" ]]; then',
                '  echo "https://example.com/pr/123"',
                "  exit 0",
                "fi",
                'echo "unexpected gh invocation: $*" >&2',
                "exit 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _env_with_stub(tmp_path: Path, *, token_available: bool) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub_gh(bin_dir / "gh", token_available=token_available)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env


def test_open_pr_uses_cached_token_when_auth_status_is_unavailable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env = _env_with_stub(tmp_path, token_available=True)

    proc = _run(["bash", str(SCRIPT)], cwd=repo, env=env)

    assert proc.returncode == 0
    assert "https://example.com/pr/123" in proc.stdout
    assert "gh is not authenticated" not in proc.stderr


def test_open_pr_still_fails_when_no_gh_token_is_available(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env = _env_with_stub(tmp_path, token_available=False)

    proc = _run(["bash", str(SCRIPT)], cwd=repo, env=env)

    assert proc.returncode == 1
    assert "gh is not authenticated" in proc.stderr
