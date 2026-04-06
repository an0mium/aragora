from __future__ import annotations

from pathlib import Path
import re

from aragora.server.fastapi import create_app
from aragora.server.fastapi.routes.debates import CreateDebateRequest


REPO_ROOT = Path(__file__).resolve().parents[3]
ENDPOINT_SELECTOR_PATH = REPO_ROOT / "aragora/live/src/components/playground/EndpointSelector.tsx"


def _endpoint_selector_source() -> str:
    return ENDPOINT_SELECTOR_PATH.read_text()


def _extract_catalog_paths(source: str) -> list[str]:
    return re.findall(r"path:\s*'([^']+)'", source)


def _extract_create_debate_payload(source: str) -> dict[str, object]:
    match = re.search(
        r"method:\s*'POST',\s*path:\s*'/api/v2/debates'.*?body:\s*\{(?P<body>.*?)\n\s*\},",
        source,
        re.DOTALL,
    )
    assert match, "Could not locate the standalone playground create-debate example"
    body = match.group("body")
    assert "task:" not in body, "Standalone playground create-debate example still uses task"

    question_match = re.search(r"question:\s*'([^']+)'", body)
    rounds_match = re.search(r"rounds:\s*(\d+)", body)
    consensus_match = re.search(r"consensus:\s*'([^']+)'", body)
    agents_match = re.search(r"agents:\s*\[(?P<agents>[^\]]+)\]", body)

    assert question_match, "Create-debate example is missing question"
    assert rounds_match, "Create-debate example is missing rounds"
    assert consensus_match, "Create-debate example is missing consensus"
    assert agents_match, "Create-debate example is missing agents"

    agents = re.findall(r"'([^']+)'", agents_match.group("agents"))
    return {
        "question": question_match.group(1),
        "agents": agents,
        "rounds": int(rounds_match.group(1)),
        "consensus": consensus_match.group(1),
    }


def test_standalone_playground_catalog_only_advertises_live_fastapi_routes():
    source = _endpoint_selector_source()
    catalog_paths = _extract_catalog_paths(source)
    app = create_app()
    live_routes = {route.path for route in app.routes}

    missing = sorted(path for path in catalog_paths if path not in live_routes)

    assert not missing, (
        "Standalone playground advertises paths missing from create_app(): " + ", ".join(missing)
    )


def test_standalone_playground_create_example_matches_create_debate_request():
    source = _endpoint_selector_source()
    payload = _extract_create_debate_payload(source)

    parsed = CreateDebateRequest.model_validate(payload)

    assert parsed.question == payload["question"]
    assert parsed.agents == payload["agents"]
    assert parsed.rounds == payload["rounds"]
    assert parsed.consensus == payload["consensus"]
