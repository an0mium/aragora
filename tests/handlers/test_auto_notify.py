"""Tests for auto_notify=True default on server-created debates."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.orchestration.models import OrchestrationRequest


class TestOrchestrationRequestNotifyDefault:
    """OrchestrationRequest.notify defaults to True for server-created debates."""

    def test_notify_defaults_to_true(self):
        req = OrchestrationRequest(question="Test question")
        assert req.notify is True

    def test_notify_from_dict_defaults_to_true(self):
        req = OrchestrationRequest.from_dict({"question": "Test question"})
        assert req.notify is True

    def test_notify_opt_out_via_dict(self):
        req = OrchestrationRequest.from_dict(
            {"question": "Test question", "notify": False}
        )
        assert req.notify is False

    def test_notify_explicit_true_via_dict(self):
        req = OrchestrationRequest.from_dict(
            {"question": "Test question", "notify": True}
        )
        assert req.notify is True


class TestDeliberatePassesAutoNotify:
    """_handle_deliberate passes auto_notify through metadata."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.orchestration import OrchestrationHandler

        return OrchestrationHandler({})

    @pytest.fixture
    def mock_auth(self):
        from aragora.rbac.models import AuthorizationContext

        return AuthorizationContext(
            user_id="test-user",
            org_id="org-1",
            roles={"member"},
            permissions={
                "orchestration:deliberate:create",
                "debates:create",
            },
        )

    def test_metadata_includes_auto_notify_true(self, handler, mock_auth):
        """When notify=True (default), metadata includes auto_notify=True."""
        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-123")
        mock_manager.wait_for_outcome = AsyncMock(return_value=None)

        mock_coordinator = MagicMock()
        handler.ctx["control_plane_coordinator"] = mock_coordinator

        data = {"question": "Should we expand?"}

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(
                handler,
                "_select_agent_team",
                new_callable=AsyncMock,
                return_value=["agent-1"],
            ),
        ):
            handler._handle_deliberate(data, MagicMock(), mock_auth, sync=True)

        # Verify auto_notify was passed in metadata
        call_kwargs = mock_manager.submit_deliberation.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["auto_notify"] is True

    def test_metadata_includes_auto_notify_false_on_opt_out(self, handler, mock_auth):
        """When notify=False, metadata includes auto_notify=False."""
        mock_manager = MagicMock()
        mock_manager.submit_deliberation = AsyncMock(return_value="task-456")
        mock_manager.wait_for_outcome = AsyncMock(return_value=None)

        mock_coordinator = MagicMock()
        handler.ctx["control_plane_coordinator"] = mock_coordinator

        data = {"question": "Should we expand?", "notify": False}

        with (
            patch(
                "aragora.control_plane.deliberation.DeliberationManager",
                return_value=mock_manager,
            ),
            patch.object(
                handler,
                "_select_agent_team",
                new_callable=AsyncMock,
                return_value=["agent-1"],
            ),
        ):
            handler._handle_deliberate(data, MagicMock(), mock_auth, sync=True)

        call_kwargs = mock_manager.submit_deliberation.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata["auto_notify"] is False

    def test_fallback_path_passes_auto_notify(self, handler, mock_auth):
        """Without control plane coordinator, auto_notify is in DecisionRequest context."""
        # No coordinator in ctx (fallback path)
        handler.ctx.pop("control_plane_coordinator", None)

        mock_router = MagicMock()
        mock_result = MagicMock(success=True)
        mock_router.route = AsyncMock(return_value=mock_result)

        data = {"question": "Test fallback path"}

        with (
            patch(
                "aragora.core.decision.get_decision_router",
                return_value=mock_router,
            ),
            patch.object(
                handler,
                "_select_agent_team",
                new_callable=AsyncMock,
                return_value=["agent-1"],
            ),
        ):
            handler._handle_deliberate(data, MagicMock(), mock_auth, sync=True)

        # Verify the DecisionRequest was created with auto_notify in context metadata
        call_args = mock_router.route.call_args
        decision_request = call_args[0][0]
        assert decision_request.context.metadata["auto_notify"] is True
