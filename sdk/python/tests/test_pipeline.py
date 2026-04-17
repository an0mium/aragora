"""Tests for the Pipeline namespace API."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestPipelineSync:
    """Synchronous pipeline endpoint tests."""

    def test_root_pipeline_routes(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.pipeline.list_pipelines()
            client.pipeline.approve_pipeline_transition(
                "pipe-1",
                "ideas",
                "goals",
                approved=False,
                comment="needs refinement",
            )

            mock_request.assert_has_calls(
                [
                    call("GET", "/api/v1/canvas/pipeline"),
                    call(
                        "POST",
                        "/api/v1/canvas/pipeline/approve-transition",
                        json={
                            "pipeline_id": "pipe-1",
                            "from_stage": "ideas",
                            "to_stage": "goals",
                            "approved": False,
                            "comment": "needs refinement",
                        },
                    ),
                ]
            )
            client.close()


class TestPipelineAsync:
    """Asynchronous pipeline endpoint tests."""

    @pytest.mark.asyncio
    async def test_root_pipeline_routes(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.pipeline.list_pipelines()
            await client.pipeline.approve_pipeline_transition(
                "pipe-1",
                "ideas",
                "goals",
                approved=False,
                comment="needs refinement",
            )

            mock_request.assert_has_calls(
                [
                    call("GET", "/api/v1/canvas/pipeline"),
                    call(
                        "POST",
                        "/api/v1/canvas/pipeline/approve-transition",
                        json={
                            "pipeline_id": "pipe-1",
                            "from_stage": "ideas",
                            "to_stage": "goals",
                            "approved": False,
                            "comment": "needs refinement",
                        },
                    ),
                ]
            )
            await client.close()
