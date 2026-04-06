from __future__ import annotations

import pytest

from aragora.swarm.boss_loop import BossLoop
from aragora.swarm.micro_decomposer import build_micro_work_orders


@pytest.mark.parametrize(
    "body",
    [
        "Touch `aragora/swarm/boss_loop.py` and `tests/swarm/test_boss_loop.py`.",
        "## Files\n- `aragora/swarm/boss_loop.py`\n- `tests/swarm/test_boss_loop.py`",
    ],
)
def test_issue_body_scope_hints_become_per_file_work_orders(tmp_path, body: str) -> None:
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    (tmp_path / "tests" / "swarm").mkdir(parents=True)
    (tmp_path / "aragora" / "swarm" / "boss_loop.py").write_text("def loop():\n    pass\n")
    (tmp_path / "tests" / "swarm" / "test_boss_loop.py").write_text("def test_loop():\n    pass\n")

    orders = build_micro_work_orders(
        goal=body,
        file_scope_hints=BossLoop._extract_file_scope_hints(body),
        repo_root=tmp_path,
    )

    file_scopes = [order["file_scope"] for order in orders]
    assert ["aragora/swarm/boss_loop.py"] in file_scopes
    assert ["tests/swarm/test_boss_loop.py"] in file_scopes
