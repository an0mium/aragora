"""
Aragora config command - Manage configuration settings.

View and modify Aragora configuration from the command line.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE = ".aragora.yaml"
ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "OPENROUTER_API_KEY",
]

def find_config() -> Path | None:
    """Find the nearest .aragora.yaml config file."""
    current = Path.cwd()
    while current != current.parent:
        config_path = current / CONFIG_FILE
        if config_path.exists():
            return config_path
        current = current.parent
    return None

def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from file."""
    if config_path is None:
        config_path = find_config()

    if config_path is None or not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Could not load config: {e}")
        return {}

def save_config(config: dict, config_path: Path | None = None) -> bool:
    """Save configuration to file."""
    if config_path is None:
        config_path = find_config()

    if config_path is None:
        config_path = Path.cwd() / CONFIG_FILE

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except OSError as e:
        print(f"Error saving config: {e}")
        return False

def get_nested(config: dict, key: str) -> Any:
    """Get a nested config value using dot notation (e.g., 'debate.rounds')."""
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value

def set_nested(config: dict, key: str, value: Any) -> None:
    """Set a nested config value using dot notation."""
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def cmd_config(args) -> None:
    """Handle 'config' command."""
    action = getattr(args, "action", "show")

    if action == "show":
        _show_config(args)
    elif action == "get":
        _get_config(args)
    elif action == "set":
        _set_config(args)
    elif action == "env":
        _show_env()
    elif action == "path":
        _show_path()
    else:
        _show_config(args)

def _show_config(args) -> None:
    """Show all configuration."""
    config_path = find_config()

    if config_path is None:
        print("No .aragora.yaml found.")
        print("Run 'aragora init' to create one.")
        return

    config = load_config(config_path)

    print(f"\nConfiguration from: {config_path}\n")
    print("-" * 40)

    if not config:
        print("(empty)")
        return

    # Pretty print the config
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

def _get_config(args) -> None:
    """Get a specific configuration value."""
    key = getattr(args, "key", None)
    if not key:
        print("Usage: aragora config get <key>")
        print("Example: aragora config get debate.rounds")
        return

    config = load_config()
    value = get_nested(config, key)

    if value is None:
        print(f"{key}: (not set)")
    else:
        print(f"{key}: {value}")

def _set_config(args) -> None:
    """Set a configuration value."""
    key = getattr(args, "key", None)
    value = getattr(args, "value", None)

    if not key or value is None:
        print("Usage: aragora config set <key> <value>")
        print("Example: aragora config set debate.rounds 5")
        return

    config = load_config()

    # Try to parse value as int, float, or bool
    parsed_value: Any = value
    if value.lower() == "true":
        parsed_value = True
    elif value.lower() == "false":
        parsed_value = False
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value

    set_nested(config, key, parsed_value)

    if save_config(config):
        print(f"Set {key} = {parsed_value}")
    else:
        print("Failed to save configuration")

def _show_env() -> None:
    """Show environment variable status."""
    print("\nAPI Key Environment Variables:")
    print("-" * 40)

    for key in ENV_KEYS:
        value = os.environ.get(key)
        if value:
            # Mask the key, showing only first/last 4 chars
            if len(value) > 12:
                masked = value[:4] + "..." + value[-4:]
            else:
                masked = "****"
            print(f"  {key}: {masked}")
        else:
            print(f"  {key}: (not set)")

    print()

def _show_path() -> None:
    """Show config file path."""
    config_path = find_config()

    if config_path:
        print(f"Config file: {config_path}")
    else:
        print("No config file found.")
        print(f"Expected: {Path.cwd() / CONFIG_FILE}")
