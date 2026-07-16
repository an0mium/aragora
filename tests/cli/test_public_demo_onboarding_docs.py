"""Guards for public first-run demo documentation."""

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


def test_readme_advertises_current_pypi_receipt_round_trip() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    pypi_try_it_now = _section_between(
        readme,
        "Current PyPI package:",
        "Current source checkout:",
    )
    source_try_it_now = _section_between(
        readme,
        "Current source checkout:",
        "Live review with a provider key:",
    )
    live_review = _section_between(
        readme,
        "Live review with a provider key:",
        "## Core workflows",
    )
    pypi_table_row = _line_containing(readme, "Run the current PyPI zero-key receipt demo")

    assert "pip install aragora && aragora demo --offline" in readme
    assert "--offline" in pypi_table_row
    assert "--receipt" in pypi_table_row
    assert "receipt verify" in pypi_table_row
    assert "aragora demo --offline --receipt aragora-demo-receipt.json" in pypi_try_it_now
    assert "aragora receipt verify aragora-demo-receipt.json" in pypi_try_it_now
    assert "Current source checkout:" in readme
    assert "PyPI `aragora` 2.9.0 supports the explicit offline demo receipt round trip" in readme
    assert "aragora demo --offline --receipt aragora-demo-receipt.json" in source_try_it_now
    assert "aragora receipt verify aragora-demo-receipt.json" in source_try_it_now
    assert (
        "aragora receipt export aragora-demo-receipt.json --format odr -o receipt.odr.json"
        in source_try_it_now
    )
    assert "aragora receipt export" not in live_review


def test_quickstart_advertises_current_pypi_receipt_round_trip() -> None:
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text()
    pypi_section = _section_between(
        quickstart,
        "Current PyPI package:",
        "Current source checkout:",
    )
    source_section = quickstart.split("Current source checkout:", 1)[1]

    assert "Current PyPI package" in quickstart
    assert "Current source checkout" in quickstart
    assert "aragora demo --offline --receipt aragora-demo-receipt.json" in pypi_section
    assert "aragora receipt verify aragora-demo-receipt.json" in pypi_section
    assert "aragora demo --offline --receipt aragora-demo-receipt.json" in source_section
    assert "aragora receipt verify aragora-demo-receipt.json" in source_section
    assert (
        "PyPI `aragora` 2.9.0 supports the explicit offline demo receipt round trip" in quickstart
    )
