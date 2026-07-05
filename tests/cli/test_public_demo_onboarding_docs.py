"""Guards for public first-run demo documentation.

These tests keep the README/quickstart from advertising an unreleased PyPI
path as if it were available in the current public package.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _section_between(text: str, start: str, end: str) -> str:
    assert start in text
    assert end in text
    after_start = text.split(start, 1)[1]
    return after_start.split(end, 1)[0]


def _line_containing(text: str, needle: str) -> str:
    matches = [line for line in text.splitlines() if needle in line]
    assert len(matches) == 1
    return matches[0]


def test_readme_does_not_claim_pypi_demo_offline_receipt_round_trip() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    pypi_try_it_now = _section_between(
        readme,
        "Current PyPI package:",
        "Current source checkout:",
    )
    pypi_table_row = _line_containing(readme, "Run the current PyPI zero-key demo")

    assert "pip install aragora && aragora demo --offline" not in readme
    assert re.search(r"pip install aragora\s*&&\s*aragora demo --offline", readme) is None
    assert re.search(r"pip install aragora\s+aragora demo --offline", readme) is None
    assert "--offline" not in pypi_table_row
    assert "--receipt" not in pypi_table_row
    assert "receipt verify" not in pypi_table_row
    assert "aragora verify" not in pypi_table_row
    assert "--offline" not in pypi_try_it_now
    assert "--receipt" not in pypi_try_it_now
    assert "receipt verify" not in pypi_try_it_now
    assert "aragora verify" not in pypi_try_it_now
    assert "Current source checkout:" in readme
    assert "PyPI `aragora` releases through" in readme


def test_quickstart_separates_pypi_demo_from_source_receipt_verification() -> None:
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text()
    pypi_section = _section_between(
        quickstart,
        "Current PyPI package:",
        "Current source checkout:",
    )
    source_section = quickstart.split("Current source checkout:", 1)[1]

    assert "Current PyPI package" in quickstart
    assert "Current source checkout" in quickstart
    assert "pip install aragora && aragora demo --offline" not in quickstart
    assert re.search(r"pip install aragora\s*&&\s*aragora demo --offline", quickstart) is None
    assert re.search(r"pip install aragora\s+aragora demo --offline", quickstart) is None
    assert "--offline" not in pypi_section
    assert "--receipt" not in pypi_section
    assert "receipt verify" not in pypi_section
    assert "aragora verify" not in pypi_section
    assert "aragora demo --offline --receipt aragora-demo-receipt.json" in source_section
    assert "aragora receipt verify aragora-demo-receipt.json" in source_section
    assert "PyPI `aragora` releases through" in quickstart
