"""Repository-local Python startup customizations.

Keep pytest runs isolated from user-site packages so local plugins installed
outside the repo do not change or break the test environment.
"""

from __future__ import annotations

import site
import sys


def _is_pytest_module_run() -> bool:
    orig_argv = getattr(sys, "orig_argv", sys.argv)
    return len(orig_argv) >= 3 and orig_argv[1] == "-m" and orig_argv[2] == "pytest"


if _is_pytest_module_run():
    user_site = site.getusersitepackages()
    while user_site in sys.path:
        sys.path.remove(user_site)
