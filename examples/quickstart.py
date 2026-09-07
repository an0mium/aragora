"""Quickstart: run a multi-agent debate with zero API keys."""

import asyncio
import sys
from pathlib import Path

try:
    from aragora_debate import Arena, StyledMockAgent
except ModuleNotFoundError:
    debate_src = Path(__file__).resolve().parents[1] / "aragora-debate" / "src"
    if debate_src.is_dir():
        sys.path.insert(0, str(debate_src))
    from aragora_debate import Arena, StyledMockAgent

agents = [
    StyledMockAgent("analyst", style="supportive"),
    StyledMockAgent("critic", style="critical"),
    StyledMockAgent("pm", style="balanced"),
]
arena = Arena(question="Should we migrate to microservices?", agents=agents)
result = asyncio.run(arena.run())
print(result.receipt.to_markdown())
