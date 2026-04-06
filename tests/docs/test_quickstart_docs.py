from pathlib import Path


def test_quickstart_guide_matches_cli_first_onboarding() -> None:
    content = Path("docs/quickstart.md").read_text(encoding="utf-8")

    assert "aragora quickstart --demo --no-browser" in content
    assert "aragora receipt inspect" in content
    assert "pip install aragora-debate" not in content
    assert "StyledMockAgent" not in content
    assert "from aragora_debate" not in content
