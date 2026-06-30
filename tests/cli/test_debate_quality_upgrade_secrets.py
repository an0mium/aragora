"""Issue #8101 sub-bug 2: quality_upgrade secrets-path consistency.

The post-consensus quality_upgrade / concretization repair path must resolve
provider credentials through the same CLI key-store surface as the main
debate path (``_resolved_cli_provider_key``). Under strict-secrets mode the
main path created agents fine while the repair path re-discovered credentials
via the strict secrets manager and failed with SecretNotFoundError — an
inconsistent posture within one command.

These tests mock all provider boundaries; no live API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aragora.cli.commands.debate import _create_revision_agent


class TestCreateRevisionAgent:
    def test_passes_cli_resolved_key_to_create_agent(self) -> None:
        """Repair agents must receive the CLI-resolved provider key, exactly
        like the main debate path does."""
        fake_agent = MagicMock()
        with (
            patch(
                "aragora.cli.commands.debate.create_agent", return_value=fake_agent
            ) as create_mock,
            patch(
                "aragora.cli.commands.debate._resolved_cli_provider_key",
                return_value="cli-store-key",
            ) as resolve_mock,
        ):
            agent = _create_revision_agent(
                "mistral-api", name="quality_upgrade_mistral-api_1", model=None
            )

        assert agent is fake_agent
        resolve_mock.assert_called_once_with("mistral-api")
        kwargs = create_mock.call_args.kwargs
        assert kwargs["api_key"] == "cli-store-key"
        assert kwargs["name"] == "quality_upgrade_mistral-api_1"
        assert kwargs["role"] == "synthesizer"

    def test_unresolved_key_passes_none_not_strict_failure(self) -> None:
        """When the CLI store has no key, pass None through (agent decides),
        matching the main debate path. No strict-mode re-resolution."""
        with (
            patch(
                "aragora.cli.commands.debate.create_agent", return_value=MagicMock()
            ) as create_mock,
            patch(
                "aragora.cli.commands.debate._resolved_cli_provider_key",
                return_value=None,
            ),
        ):
            _create_revision_agent("grok", name="quality_upgrade_grok_2", model="grok-4")

        kwargs = create_mock.call_args.kwargs
        assert kwargs["api_key"] is None
        assert kwargs["model"] == "grok-4"
