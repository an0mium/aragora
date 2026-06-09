from __future__ import annotations

import subprocess


def test_json_output_pipe_to_head_exits_without_broken_pipe() -> None:
    result = subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            "python3 scripts/analyze_openapi_coverage.py --json | head -5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "BrokenPipeError" not in result.stderr
    assert '"summary"' in result.stdout
