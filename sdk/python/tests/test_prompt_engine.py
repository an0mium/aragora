"""Tests for prompt-engine namespace route mappings."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.prompt_engine import AsyncPromptEngineAPI, PromptEngineAPI


class TestPromptEngineRoutes:
    """Tests for /api/prompt-engine endpoints."""

    def test_prompt_engine_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            prompt_engine = PromptEngineAPI(client)

            prompt_engine.list_runs(status="spec_ready", limit=10)
            prompt_engine.get_run("run-1")
            prompt_engine.run("Build the thing", profile="autonomous")
            prompt_engine.decompose("Build the thing", context={"repo": "aragora"})
            prompt_engine.interrogate({"raw_prompt": "Build"}, depth="quick")
            prompt_engine.research({"raw_prompt": "Build"}, context={"repo": "aragora"})
            prompt_engine.specify({"raw_prompt": "Build"}, questions=[], research={"summary": "ok"})
            prompt_engine.validate({"title": "Spec"})

            expected_calls = [
                call(
                    "GET",
                    "/api/prompt-engine/runs",
                    params={"status": "spec_ready", "limit": 10},
                ),
                call("GET", "/api/prompt-engine/runs/run-1"),
                call(
                    "POST",
                    "/api/prompt-engine/run",
                    json={"prompt": "Build the thing", "profile": "autonomous"},
                ),
                call(
                    "POST",
                    "/api/prompt-engine/decompose",
                    json={"prompt": "Build the thing", "context": {"repo": "aragora"}},
                ),
                call(
                    "POST",
                    "/api/prompt-engine/interrogate",
                    json={"intent": {"raw_prompt": "Build"}, "depth": "quick"},
                ),
                call(
                    "POST",
                    "/api/prompt-engine/research",
                    json={"intent": {"raw_prompt": "Build"}, "context": {"repo": "aragora"}},
                ),
                call(
                    "POST",
                    "/api/prompt-engine/specify",
                    json={
                        "intent": {"raw_prompt": "Build"},
                        "questions": [],
                        "research": {"summary": "ok"},
                    },
                ),
                call(
                    "POST",
                    "/api/prompt-engine/validate",
                    json={"specification": {"title": "Spec"}},
                ),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == len(expected_calls)
            client.close()


class TestAsyncPromptEngineRoutes:
    """Async tests for /api/prompt-engine endpoints."""

    @pytest.mark.asyncio
    async def test_async_prompt_engine_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai", api_key="test-key"
            ) as client:
                prompt_engine = AsyncPromptEngineAPI(client)

                await prompt_engine.list_runs(offset=5)
                await prompt_engine.get_run("run-async")
                await prompt_engine.run("Build async")
                await prompt_engine.decompose("Build async")
                await prompt_engine.interrogate({"raw_prompt": "Build"})
                await prompt_engine.research({"raw_prompt": "Build"})
                await prompt_engine.specify({"raw_prompt": "Build"})
                await prompt_engine.validate({"title": "Spec"})

                expected_calls = [
                    call("GET", "/api/prompt-engine/runs", params={"offset": 5}),
                    call("GET", "/api/prompt-engine/runs/run-async"),
                    call("POST", "/api/prompt-engine/run", json={"prompt": "Build async"}),
                    call(
                        "POST",
                        "/api/prompt-engine/decompose",
                        json={"prompt": "Build async"},
                    ),
                    call(
                        "POST",
                        "/api/prompt-engine/interrogate",
                        json={"intent": {"raw_prompt": "Build"}},
                    ),
                    call(
                        "POST",
                        "/api/prompt-engine/research",
                        json={"intent": {"raw_prompt": "Build"}},
                    ),
                    call(
                        "POST",
                        "/api/prompt-engine/specify",
                        json={"intent": {"raw_prompt": "Build"}},
                    ),
                    call(
                        "POST",
                        "/api/prompt-engine/validate",
                        json={"specification": {"title": "Spec"}},
                    ),
                ]
                mock_request.assert_has_calls(expected_calls)
                assert mock_request.call_count == len(expected_calls)
