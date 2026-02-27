"""Tests for Interrogation HTTP handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from aragora.server.handlers.interrogation import handler as handler_module
from aragora.server.handlers.interrogation.handler import InterrogationHandler


class TestInterrogationHandler:
    @pytest.fixture(autouse=True)
    def _clear_sessions(self):
        """Clear the module-level session store between tests."""
        handler_module._sessions.clear()
        yield
        handler_module._sessions.clear()

    @pytest.fixture
    def handler(self):
        return InterrogationHandler()

    # ------------------------------------------------------------------
    # POST /api/v1/interrogation/start
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_interrogation(self, handler):
        body = json.dumps({"prompt": "Make it better"}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        assert response.status == 200
        data = json.loads(response.body)
        assert "dimensions" in data["data"]
        assert "questions" in data["data"]
        assert "session_id" in data["data"]

    @pytest.mark.asyncio
    async def test_start_empty_prompt_returns_400(self, handler):
        body = json.dumps({"prompt": ""}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_prompt_returns_400(self, handler):
        body = json.dumps({}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, handler):
        body = b"not json"
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_start_returns_dimension_fields(self, handler):
        body = json.dumps({"prompt": "Make it better"}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        data = json.loads(response.body)["data"]
        for dim in data["dimensions"]:
            assert "name" in dim
            assert "description" in dim
            assert "vagueness_score" in dim

    @pytest.mark.asyncio
    async def test_start_returns_question_fields(self, handler):
        body = json.dumps({"prompt": "Make it better"}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_start(request)
        data = json.loads(response.body)["data"]
        for q in data["questions"]:
            assert "text" in q
            assert "why" in q
            assert "options" in q
            assert "context" in q
            assert "priority" in q

    # ------------------------------------------------------------------
    # POST /api/v1/interrogation/answer
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_answer_unknown_session_returns_404(self, handler):
        body = json.dumps(
            {
                "session_id": "nonexistent",
                "question": "test?",
                "answer": "yes",
            }
        ).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/answer")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_answer(request)
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_answer_missing_fields_returns_400(self, handler):
        # First start a session so we have a valid session_id
        start_body = json.dumps({"prompt": "Make it better"}).encode()
        start_req = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(start_req, "read", new_callable=AsyncMock, return_value=start_body):
            start_resp = await handler.handle_start(start_req)
        session_id = json.loads(start_resp.body)["data"]["session_id"]

        # Now answer without question/answer fields
        body = json.dumps({"session_id": session_id}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/answer")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_answer(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_answer_records_answer(self, handler):
        # Start a session
        start_body = json.dumps({"prompt": "Make it better"}).encode()
        start_req = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(start_req, "read", new_callable=AsyncMock, return_value=start_body):
            start_resp = await handler.handle_start(start_req)
        start_data = json.loads(start_resp.body)["data"]
        session_id = start_data["session_id"]
        first_q = start_data["questions"][0]["text"]

        # Answer the first question
        answer_body = json.dumps(
            {
                "session_id": session_id,
                "question": first_q,
                "answer": "Focus on latency",
            }
        ).encode()
        answer_req = make_mocked_request("POST", "/api/v1/interrogation/answer")
        with patch.object(answer_req, "read", new_callable=AsyncMock, return_value=answer_body):
            answer_resp = await handler.handle_answer(answer_req)
        assert answer_resp.status == 200
        answer_data = json.loads(answer_resp.body)["data"]
        assert answer_data["session_id"] == session_id
        assert answer_data["answered"] == 1

    # ------------------------------------------------------------------
    # POST /api/v1/interrogation/crystallize
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_crystallize_unknown_session_returns_404(self, handler):
        body = json.dumps({"session_id": "nonexistent"}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/crystallize")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_crystallize(request)
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_crystallize_returns_spec(self, handler):
        # Start a session
        start_body = json.dumps({"prompt": "Make it better"}).encode()
        start_req = make_mocked_request("POST", "/api/v1/interrogation/start")
        with patch.object(start_req, "read", new_callable=AsyncMock, return_value=start_body):
            start_resp = await handler.handle_start(start_req)
        session_id = json.loads(start_resp.body)["data"]["session_id"]

        # Crystallize
        body = json.dumps({"session_id": session_id}).encode()
        request = make_mocked_request("POST", "/api/v1/interrogation/crystallize")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_crystallize(request)
        assert response.status == 200
        data = json.loads(response.body)["data"]
        assert data["session_id"] == session_id
        assert "spec" in data
        spec = data["spec"]
        assert "problem_statement" in spec
        assert "requirements" in spec
        assert "non_requirements" in spec
        assert "success_criteria" in spec
        assert "risks" in spec
        assert "context_summary" in spec
        assert "goal_text" in data

    @pytest.mark.asyncio
    async def test_crystallize_invalid_json_returns_400(self, handler):
        body = b"bad json"
        request = make_mocked_request("POST", "/api/v1/interrogation/crystallize")
        with patch.object(request, "read", new_callable=AsyncMock, return_value=body):
            response = await handler.handle_crystallize(request)
        assert response.status == 400

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def test_register_routes(self, handler):
        app = web.Application()
        handler.register_routes(app)
        routes = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]
        assert "/api/v1/interrogation/start" in routes
        assert "/api/v1/interrogation/answer" in routes
        assert "/api/v1/interrogation/crystallize" in routes
