"""
Platform-specific dock implementations.

Each dock implements the ChannelDock interface for a specific
messaging platform (Slack, Telegram, Discord, etc.).
"""

# Docks are imported lazily by the registry to avoid circular imports
# and to allow selective installation of platform dependencies.
