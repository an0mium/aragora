"""Repository-local Python startup customizations.

Keep pytest runs isolated from user-site packages so local plugins installed
outside the repo do not change or break the test environment.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path


def _get_orig_argv() -> list[str]:
    return list(getattr(sys, "orig_argv", sys.argv))


def _command_name(argv: list[str]) -> str:
    if not argv:
        return ""
    name = Path(argv[0]).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _is_pytest_invocation(argv: list[str] | None = None) -> bool:
    orig_argv = list(argv) if argv is not None else _get_orig_argv()
    if len(orig_argv) >= 3 and orig_argv[1] == "-m" and orig_argv[2] == "pytest":
        return True
    command_name = _command_name(orig_argv)
    return command_name in {"pytest", "py.test"} or command_name.startswith("pytest")


def _remove_user_site(path: list[str] | None = None, user_site: str | None = None) -> None:
    target_path = sys.path if path is None else path
    target_user_site = site.getusersitepackages() if user_site is None else user_site
    while target_user_site in target_path:
        target_path.remove(target_user_site)


def _configure_pytest_startup(
    argv: list[str] | None = None,
    path: list[str] | None = None,
    user_site: str | None = None,
) -> None:
    orig_argv = list(argv) if argv is not None else _get_orig_argv()
    if not _is_pytest_invocation(orig_argv):
        return

    _remove_user_site(path=path, user_site=user_site)


_configure_pytest_startup()
