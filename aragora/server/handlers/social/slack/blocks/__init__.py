"""
Slack Block Kit utilities package.

Provides block building functions for creating rich Slack messages:
- Starting blocks: Initial debate announcement
- Round update blocks: Progress updates during debates
- Agent response blocks: Individual agent responses
- Result blocks: Final debate results
- Gauntlet result blocks: Stress-test validation results
- Search result blocks: Search results display
"""

from aragora.server.handlers.social.slack.blocks.builders import (
    build_starting_blocks,
    build_round_update_blocks,
    build_agent_response_blocks,
    build_result_blocks,
    build_gauntlet_result_blocks,
    build_search_result_blocks,
    AGENT_EMOJIS,
)

__all__ = [
    "build_starting_blocks",
    "build_round_update_blocks",
    "build_agent_response_blocks",
    "build_result_blocks",
    "build_gauntlet_result_blocks",
    "build_search_result_blocks",
    "AGENT_EMOJIS",
]
