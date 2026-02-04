from aragora.knowledge.mound.api.router import DocumentFeatures, LaRARouter


def test_graph_hint_routing() -> None:
    router = LaRARouter()
    decision = router.route(
        "graph:node_123",
        DocumentFeatures(total_nodes=100),
        supports_rlm=False,
    )
    assert decision.route == "graph"
    assert decision.start_id == "node_123"


def test_rlm_hint_routing() -> None:
    router = LaRARouter()
    decision = router.route(
        "summarize the debate outcome",
        DocumentFeatures(total_nodes=500),
        supports_rlm=True,
    )
    assert decision.route == "rlm"


def test_keyword_routing_for_short_query() -> None:
    router = LaRARouter()
    decision = router.route(
        "hello",
        DocumentFeatures(total_nodes=10),
        supports_rlm=False,
    )
    assert decision.route == "keyword"


def test_long_context_routing() -> None:
    router = LaRARouter()
    decision = router.route(
        "this query is long enough to trigger long context routing",
        DocumentFeatures(total_nodes=2000),
        supports_rlm=False,
    )
    assert decision.route == "long_context"


def test_force_route() -> None:
    router = LaRARouter()
    decision = router.route(
        "whatever",
        DocumentFeatures(total_nodes=0),
        supports_rlm=False,
        force_route="semantic",
    )
    assert decision.route == "semantic"
