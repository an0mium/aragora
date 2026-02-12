"""Quickstart: run a multi-agent debate with zero API keys."""
import asyncio
from aragora_debate.arena import Arena
from aragora_debate.styled_mock import StyledMockAgent

agents = [
    StyledMockAgent("analyst", style="supportive"),
    StyledMockAgent("critic", style="critical"),
    StyledMockAgent("pm", style="balanced"),
]
arena = Arena(question="Should we migrate to microservices?", agents=agents)
result = asyncio.run(arena.run())
print(result.receipt.to_markdown())
