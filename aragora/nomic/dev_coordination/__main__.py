"""Enable ``python -m aragora.nomic.dev_coordination`` after the package split.

Before TCP-3 PR-A the module was a single ``dev_coordination.py`` file with
an ``if __name__ == "__main__": raise SystemExit(main())`` guard at its
tail.  Now that ``dev_coordination`` is a package, ``python -m`` looks for
``__main__.py`` here instead.  ``scripts/codex_session.sh`` relies on the
invocation verbatim.
"""

from __future__ import annotations

from aragora.nomic.dev_coordination.core import main

if __name__ == "__main__":
    raise SystemExit(main())
