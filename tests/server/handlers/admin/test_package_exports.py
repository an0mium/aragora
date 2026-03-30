from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_admin_package_import_keeps_jwt_auth_lazy() -> None:
    result = _run(
        """
import sys
import aragora.server.handlers.admin as admin
print(admin.__name__)
print("aragora.billing.jwt_auth" in sys.modules)
"""
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-2] == "aragora.server.handlers.admin"
    assert lines[-1] == "False"


def test_admin_export_loads_jwt_auth_on_demand() -> None:
    result = _run(
        """
import sys
import aragora.server.handlers.admin as admin
print("before", "aragora.billing.jwt_auth" in sys.modules)
_ = admin.extract_user_from_request
print("after", "aragora.billing.jwt_auth" in sys.modules)
"""
    )

    assert result.returncode == 0, result.stderr
    assert "before False" in result.stdout
    assert "after True" in result.stdout
