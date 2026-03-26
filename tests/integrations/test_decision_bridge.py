from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.decision_bridge import DecisionBridge


class _CredentialProvider:
    def __init__(self, values: dict[str, str]):
        self._values = values

    async def get_credential(self, key: str) -> str | None:
        return self._values.get(key)

    async def set_credential(self, key: str, value: str) -> None:
        self._values[key] = value


def _plan(*, metadata: dict[str, str] | None = None, tasks: list[object] | None = None) -> object:
    return SimpleNamespace(
        id="plan-123",
        title="Bridge decision",
        metadata=metadata or {},
        implement_plan=SimpleNamespace(tasks=tasks or []),
    )


class TestDecisionBridgeConfig:
    @pytest.mark.asyncio
    async def test_jira_requires_real_config(self) -> None:
        bridge = DecisionBridge(default_targets=["jira"])

        with patch(
            "aragora.integrations.decision_bridge.get_credential_provider",
            return_value=_CredentialProvider({}),
        ):
            result = await bridge.handle_decision_plan(
                _plan(tasks=[SimpleNamespace(title="Ship bridge route")])
            )

        assert result.jira_issues == []
        assert result.errors == [
            "Decision bridge failed for jira: Jira bridge requires JIRA_BASE_URL and JIRA_PROJECT_KEY configuration"
        ]

    @pytest.mark.asyncio
    async def test_linear_requires_real_api_key(self) -> None:
        bridge = DecisionBridge(default_targets=["linear"])

        with patch(
            "aragora.integrations.decision_bridge.get_credential_provider",
            return_value=_CredentialProvider({}),
        ):
            result = await bridge.handle_decision_plan(
                _plan(tasks=[SimpleNamespace(title="Ship bridge route")])
            )

        assert result.linear_issues == []
        assert result.errors == [
            "Decision bridge failed for linear: Linear bridge requires LINEAR_API_KEY configuration"
        ]


class TestDecisionBridgeDispatch:
    @pytest.mark.asyncio
    async def test_jira_uses_configured_base_url_project_and_issue_endpoint(self) -> None:
        provider = _CredentialProvider(
            {
                "JIRA_BASE_URL": "https://example.atlassian.net",
                "JIRA_PROJECT_KEY": "ENG",
            }
        )
        connector = MagicMock()
        connector._api_request = AsyncMock(return_value={"id": "1001", "key": "ENG-1"})

        with (
            patch(
                "aragora.integrations.decision_bridge.get_credential_provider",
                return_value=provider,
            ),
            patch(
                "aragora.connectors.enterprise.collaboration.jira.JiraConnector",
                return_value=connector,
            ) as mock_connector_cls,
        ):
            result = await DecisionBridge(default_targets=["jira"]).handle_decision_plan(
                _plan(
                    tasks=[SimpleNamespace(title="Ship bridge route", description="Implement it")]
                )
            )

        assert result.errors == []
        assert result.jira_issues == [
            {"id": "1001", "key": "ENG-1", "summary": "Ship bridge route"}
        ]
        mock_connector_cls.assert_called_once_with(
            base_url="https://example.atlassian.net",
            projects=["ENG"],
        )
        connector._api_request.assert_awaited_once()
        call = connector._api_request.await_args
        assert call.args[0] == "/issue"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json_data"]["fields"]["project"] == {"key": "ENG"}

    @pytest.mark.asyncio
    async def test_linear_uses_configured_api_key_and_team(self) -> None:
        provider = _CredentialProvider(
            {
                "LINEAR_API_KEY": "lin_api_key_123",
                "LINEAR_TEAM_ID": "team-42",
            }
        )
        connector = MagicMock()
        connector.get_teams = AsyncMock(return_value=[SimpleNamespace(id="team-other")])
        connector.create_issue = AsyncMock(
            return_value=SimpleNamespace(id="issue-1", identifier="LIN-1")
        )

        with (
            patch(
                "aragora.integrations.decision_bridge.get_credential_provider",
                return_value=provider,
            ),
            patch(
                "aragora.connectors.enterprise.collaboration.linear.LinearConnector",
                return_value=connector,
            ) as mock_connector_cls,
        ):
            result = await DecisionBridge(default_targets=["linear"]).handle_decision_plan(
                _plan(tasks=[SimpleNamespace(title="Wire auth headers", description="Use auth")])
            )

        assert result.errors == []
        assert result.linear_issues == [
            {"id": "issue-1", "identifier": "LIN-1", "title": "Wire auth headers"}
        ]
        credentials = mock_connector_cls.call_args.kwargs["credentials"]
        assert credentials.api_key == "lin_api_key_123"
        assert credentials.base_url == "https://api.linear.app/graphql"
        connector.create_issue.assert_awaited_once_with(
            team_id="team-42",
            title="[Aragora] Wire auth headers",
            description="Auto-created from decision plan: Bridge decision\n\nUse auth",
        )
