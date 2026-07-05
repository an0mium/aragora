"""Guards for public first-run demo documentation.

These tests keep the README/quickstart from advertising an unreleased PyPI
path as if it were available in the current public package.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_does_not_claim_pypi_demo_offline_receipt_round_trip() -> None:
    readme = (REPO_ROOT / "README.md").read_text()

    assert "pip install aragora && aragora demo --offline" not in readme
    assert "current source checkout" in readme
    assert "PyPI `aragora` package" in readme


def test_quickstart_separates_pypi_demo_from_source_receipt_verification() -> None:
    quickstart = (REPO_ROOT / "docs" / "quickstart.md").read_text()

    assert "Current PyPI package" in quickstart
    assert "Current source checkout" in quickstart
    assert "aragora receipt verify aragora-demo-receipt.json" in quickstart
