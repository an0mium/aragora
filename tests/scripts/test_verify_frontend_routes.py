from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_frontend_routes.sh"
EXPECTED_HEADER = "x-vercel-protection-bypass: token"


def _run_verify(
    tmp_path: Path,
    base_url: str,
    *routes: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _write_curl_stub(tmp_path)
    merged_env = os.environ.copy()
    merged_env["PATH"] = f"{bin_dir}:{merged_env['PATH']}"
    merged_env["VERIFY_FRONTEND_TEST_EXPECTED_HEADER"] = EXPECTED_HEADER
    if env:
        merged_env.update(env)

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), base_url, *routes],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_curl_stub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "curl"
    stub.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
headers = []
output_path = None

for index, arg in enumerate(args):
    if arg == "-H" and index + 1 < len(args):
        headers.append(args[index + 1])
    if arg == "-o" and index + 1 < len(args):
        output_path = args[index + 1]

expected_header = os.environ.get("VERIFY_FRONTEND_TEST_EXPECTED_HEADER")
if expected_header and expected_header not in headers:
    status = "401"
    body = "unauthorized"
else:
    status = "200"
    body = "<html><body>ok</body></html>"

if output_path is not None:
    Path(output_path).write_text(body, encoding="utf-8")

sys.stdout.write(status)
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def test_verify_frontend_routes_fails_on_401_by_default(tmp_path: Path) -> None:
    result = _run_verify(tmp_path, "https://aragora.test", "/")

    assert result.returncode != 0
    assert "status 401" in result.stdout


def test_verify_frontend_routes_uses_auth_header_when_configured(tmp_path: Path) -> None:
    result = _run_verify(
        tmp_path,
        "https://aragora.test",
        "/",
        env={"VERIFY_FRONTEND_AUTH_HEADER": EXPECTED_HEADER},
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_verify_frontend_routes_soft_fail_is_non_blocking(tmp_path: Path) -> None:
    result = _run_verify(
        tmp_path,
        "https://aragora.test",
        "/",
        env={
            "VERIFY_FRONTEND_SOFT_FAIL": "1",
            "VERIFY_FRONTEND_ANNOTATION_LEVEL": "warning",
        },
    )

    assert result.returncode == 0
    assert "::warning::Route check failed" in result.stdout
    assert "soft-fail mode" in result.stdout
