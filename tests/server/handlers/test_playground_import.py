from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_playground_import_does_not_initialize_secrets_manager() -> None:
    """Importing the public playground handler should stay side-effect free."""
    script = """
import os
import aragora.config.secrets as secrets

os.environ.pop("ARAGORA_JWT_SECRET", None)
os.environ.pop("ARAGORA_JWT_SECRET_PREVIOUS", None)
os.environ["ARAGORA_USE_SECRETS_MANAGER"] = "true"

def fail(self):
    raise RuntimeError("aws_called_during_import")

secrets.SecretManager._load_from_aws = fail

import aragora.server.handlers.playground  # noqa: F401
print("ok")
"""

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().endswith("ok")
