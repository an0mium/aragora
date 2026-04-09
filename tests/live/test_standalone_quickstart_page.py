from pathlib import Path


QUICKSTART_PAGE = Path("aragora/live/src/app/(standalone)/quickstart/page.tsx")


def test_standalone_quickstart_uses_one_truthful_python_surface() -> None:
    source = QUICKSTART_PAGE.read_text(encoding="utf-8")

    assert "pip install aragora-debate" in source
    assert "from aragora_debate import Arena, DebateConfig, StyledMockAgent" in source
    assert "from aragora_debate import Arena, DebateConfig, create_agent" in source
    assert "aragora-debate[anthropic]" in source
    assert "aragora-debate[openai]" in source
    assert "print(result.summary())" in source

    assert "from aragora import Arena, Environment, DebateProtocol" not in source
    assert "print(result.summary)" not in source
    assert "ConnectOpenRouterButton" not in source
