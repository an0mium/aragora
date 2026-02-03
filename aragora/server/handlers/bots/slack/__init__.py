"""
Slack Bot Handler Package.

This package provides bi-directional Slack integration for Aragora including:
- Slash commands (/aragora)
- Events API handling
- Interactive components (buttons, modals)
- OAuth installation flow
- RBAC permission enforcement

The package is organized into the following modules:
- handler.py: Main SlackHandler class and route registration
- events.py: Slack Events API webhook handling
- interactions.py: Interactive component callbacks
- commands.py: Slash command processing
- oauth.py: OAuth2 installation flow
- user_management.py: User/workspace operations
- blocks.py: Block Kit message builders
- signature.py: Request signature verification
- constants.py: Shared constants and validation
- state.py: Shared state management
- debates.py: Debate creation and management

For backward compatibility, all public symbols from the original slack.py
are re-exported from this package.
"""

from aragora.audit.unified import audit_data

# Import from submodules for re-export
from .blocks import (
    build_consensus_message_blocks,
    build_debate_message_blocks,
    build_debate_result_blocks,
    build_start_debate_modal,
    _build_start_debate_modal,
)
from .commands import handle_slack_commands
from .constants import (
    AGENT_DISPLAY_NAMES,
    COMMAND_PATTERN,
    MAX_CHANNEL_ID_LENGTH,
    MAX_COMMAND_LENGTH,
    MAX_TOPIC_LENGTH,
    MAX_USER_ID_LENGTH,
    PERM_SLACK_ADMIN,
    PERM_SLACK_COMMANDS_EXECUTE,
    PERM_SLACK_COMMANDS_READ,
    PERM_SLACK_DEBATES_CREATE,
    PERM_SLACK_INTERACTIVE,
    PERM_SLACK_VOTES_RECORD,
    RBAC_AVAILABLE,
    SLACK_BOT_TOKEN,
    SLACK_SIGNING_SECRET,
    TOPIC_PATTERN,
    AuthorizationContext,
    AuthorizationDecision,
    check_permission,
    validate_slack_channel_id,
    validate_slack_input,
    validate_slack_team_id,
    validate_slack_user_id,
    _validate_slack_channel_id,
    _validate_slack_input,
    _validate_slack_team_id,
    _validate_slack_user_id,
)
from .debates import (
    start_slack_debate,
    _start_slack_debate,
    _fallback_start_debate,
)
from .events import handle_slack_events
from .handler import SlackHandler, register_slack_routes
from .interactions import handle_slack_interactions
from .oauth import (
    handle_slack_oauth_callback,
    handle_slack_oauth_revoke,
    handle_slack_oauth_start,
)
from .signature import verify_slack_signature
from .state import (
    _active_debates,
    _user_votes,
    get_active_debates,
    get_debate_vote_counts,
    get_slack_integration,
    get_user_votes,
)
from .user_management import (
    build_auth_context_from_slack,
    check_user_permission,
    check_user_permission_or_admin,
    check_workspace_authorized,
    get_org_from_team,
    get_user_roles_from_slack,
)


# =============================================================================
# Backward Compatibility Exports
# =============================================================================
# These match the original slack.py __all__ list exactly
__all__ = [
    # Handler classes
    "SlackHandler",
    # Handler functions
    "handle_slack_events",
    "handle_slack_interactions",
    "handle_slack_commands",
    # Block Kit builders
    "build_debate_message_blocks",
    "build_consensus_message_blocks",
    "build_debate_result_blocks",
    # Utility functions
    "get_debate_vote_counts",
    "register_slack_routes",
    # RBAC Permission constants
    "PERM_SLACK_COMMANDS_READ",
    "PERM_SLACK_COMMANDS_EXECUTE",
    "PERM_SLACK_DEBATES_CREATE",
    "PERM_SLACK_VOTES_RECORD",
    "PERM_SLACK_INTERACTIVE",
    "PERM_SLACK_ADMIN",
    # Additional exports for full compatibility
    "verify_slack_signature",
    "get_slack_integration",
    "build_start_debate_modal",
    "_build_start_debate_modal",
    "start_slack_debate",
    "_start_slack_debate",
    "_fallback_start_debate",
    # State
    "_active_debates",
    "_user_votes",
    "get_active_debates",
    "get_user_votes",
    # Constants
    "AGENT_DISPLAY_NAMES",
    "COMMAND_PATTERN",
    "TOPIC_PATTERN",
    "MAX_TOPIC_LENGTH",
    "MAX_COMMAND_LENGTH",
    "MAX_USER_ID_LENGTH",
    "MAX_CHANNEL_ID_LENGTH",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "RBAC_AVAILABLE",
    "check_permission",
    "AuthorizationContext",
    "AuthorizationDecision",
    # Validation functions
    "validate_slack_input",
    "validate_slack_user_id",
    "validate_slack_channel_id",
    "validate_slack_team_id",
    "_validate_slack_input",
    "_validate_slack_user_id",
    "_validate_slack_channel_id",
    "_validate_slack_team_id",
    # User management
    "get_org_from_team",
    "get_user_roles_from_slack",
    "check_workspace_authorized",
    "build_auth_context_from_slack",
    "check_user_permission",
    "check_user_permission_or_admin",
    # OAuth
    "handle_slack_oauth_start",
    "handle_slack_oauth_callback",
    "handle_slack_oauth_revoke",
    # Audit logging
    "audit_data",
]


# =============================================================================
# Import Verification Comment
# =============================================================================
# To verify imports work correctly, run:
#   python -c "from aragora.server.handlers.bots.slack import SlackHandler, handle_slack_events; print('OK')"
# Or run the full verification:
#   python -c "from aragora.server.handlers.bots.slack import *; print('All imports successful')"
# Or verify backward compatibility with the original import path:
#   python -c "from aragora.server.handlers.bots import slack; print(f'SlackHandler: {slack.SlackHandler}')"
