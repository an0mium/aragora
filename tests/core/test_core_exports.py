from aragora.core import (
    DebateProtocol,
    DecisionRequest,
    DecisionRouter,
    DecisionType,
    InputSource,
    RequestContext,
    ResponseChannel,
    get_decision_router,
)


def test_core_reexports_debate_protocol():
    from aragora.debate.protocol import DebateProtocol as DebateProtocolImpl

    assert DebateProtocol is DebateProtocolImpl


def test_core_reexports_decision_router_surface():
    from aragora.core.decision import (
        DecisionRequest as DecisionRequestImpl,
        DecisionRouter as DecisionRouterImpl,
        DecisionType as DecisionTypeImpl,
        InputSource as InputSourceImpl,
        RequestContext as RequestContextImpl,
        ResponseChannel as ResponseChannelImpl,
        get_decision_router as get_decision_router_impl,
    )

    assert DecisionRequest is DecisionRequestImpl
    assert DecisionRouter is DecisionRouterImpl
    assert DecisionType is DecisionTypeImpl
    assert InputSource is InputSourceImpl
    assert RequestContext is RequestContextImpl
    assert ResponseChannel is ResponseChannelImpl
    assert get_decision_router is get_decision_router_impl
