import asyncio

from aragora.approvals import tokens as tokens_module
from aragora.approvals.interaction_router import ApprovalInteractionRouter
from aragora.approvals.tokens import encode_approval_action
from aragora.connectors.chat.models import (
    ChatChannel,
    ChatUser,
    InteractionType,
    UserInteraction,
    WebhookEvent,
)
from aragora.workflow.nodes import human_checkpoint as hc


class DummyConnector:
    def __init__(self) -> None:
        self.last_message = None

    async def respond_to_interaction(self, interaction, text, replace_original=False, **kwargs):
        self.last_message = text
        return True


def test_interaction_router_workflow_approval(monkeypatch):
    monkeypatch.setenv("ARAGORA_APPROVAL_ACTION_SECRET", "test-secret")
    monkeypatch.setenv("ARAGORA_CHAT_APPROVAL_REQUIRE_IDENTITY", "0")
    tokens_module._SECRET = None
    tokens_module._SECRET_INSECURE = False
    hc.clear_pending_approvals()

    request = hc.ApprovalRequest(
        id="apr_test",
        workflow_id="wf_test",
        step_id="step1",
        title="Test Approval",
        description="Please approve",
        checklist=[],
        timeout_seconds=3600,
        escalation_emails=[],
    )
    hc._pending_approvals[request.id] = request

    token = encode_approval_action(kind="workflow", target_id=request.id, action="approve")
    assert token

    interaction = UserInteraction(
        id="int1",
        interaction_type=InteractionType.BUTTON_CLICK,
        action_id="approval:approve",
        value=token,
        user=ChatUser(id="user-1", platform="slack", username="tester"),
        channel=ChatChannel(id="#test", platform="slack"),
        platform="slack",
    )
    event = WebhookEvent(platform="slack", event_type="interaction", interaction=interaction)

    connector = DummyConnector()
    router = ApprovalInteractionRouter()

    ok = asyncio.run(router.handle_interaction(event, connector))
    assert ok is True
    assert hc._pending_approvals[request.id].status == hc.ApprovalStatus.APPROVED
    assert connector.last_message is not None
