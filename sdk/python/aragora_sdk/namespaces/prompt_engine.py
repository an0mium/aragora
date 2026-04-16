"""Prompt Engine namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


def _compact(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


class PromptEngineAPI:
    """Synchronous Prompt Engine API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_runs(
        self,
        *,
        status: str | None = None,
        plan_id: str | None = None,
        debate_id: str | None = None,
        execution_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List persisted prompt-engine runs."""
        params = _compact(
            {
                "status": status,
                "plan_id": plan_id,
                "debate_id": debate_id,
                "execution_id": execution_id,
                "limit": limit,
                "offset": offset,
            }
        )
        return self._client.request("GET", "/api/prompt-engine/runs", params=params or None)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch a persisted prompt-engine run."""
        return self._client.request("GET", f"/api/prompt-engine/runs/{run_id}")

    def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the full prompt-to-specification pipeline."""
        return self._client.request(
            "POST", "/api/prompt-engine/run", json={"prompt": prompt, **kwargs}
        )

    def decompose(self, prompt: str, *, context: Any | None = None) -> dict[str, Any]:
        """Decompose a prompt into structured intent."""
        return self._client.request(
            "POST",
            "/api/prompt-engine/decompose",
            json=_compact({"prompt": prompt, "context": context}),
        )

    def interrogate(self, intent: dict[str, Any], *, depth: str | None = None) -> dict[str, Any]:
        """Generate clarifying questions for an intent."""
        return self._client.request(
            "POST",
            "/api/prompt-engine/interrogate",
            json=_compact({"intent": intent, "depth": depth}),
        )

    def research(self, intent: dict[str, Any], *, context: Any | None = None) -> dict[str, Any]:
        """Research supporting context for an intent."""
        return self._client.request(
            "POST",
            "/api/prompt-engine/research",
            json=_compact({"intent": intent, "context": context}),
        )

    def specify(
        self,
        intent: dict[str, Any],
        *,
        questions: list[dict[str, Any]] | None = None,
        research: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Build a specification from intent, questions, research, and context."""
        return self._client.request(
            "POST",
            "/api/prompt-engine/specify",
            json=_compact(
                {
                    "intent": intent,
                    "questions": questions,
                    "research": research,
                    "context": context,
                }
            ),
        )

    def validate(self, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate a prompt-engine specification."""
        return self._client.request(
            "POST",
            "/api/prompt-engine/validate",
            json={"specification": specification},
        )


class AsyncPromptEngineAPI:
    """Asynchronous Prompt Engine API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_runs(
        self,
        *,
        status: str | None = None,
        plan_id: str | None = None,
        debate_id: str | None = None,
        execution_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List persisted prompt-engine runs."""
        params = _compact(
            {
                "status": status,
                "plan_id": plan_id,
                "debate_id": debate_id,
                "execution_id": execution_id,
                "limit": limit,
                "offset": offset,
            }
        )
        return await self._client.request("GET", "/api/prompt-engine/runs", params=params or None)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch a persisted prompt-engine run."""
        return await self._client.request("GET", f"/api/prompt-engine/runs/{run_id}")

    async def run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the full prompt-to-specification pipeline."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/run",
            json={"prompt": prompt, **kwargs},
        )

    async def decompose(self, prompt: str, *, context: Any | None = None) -> dict[str, Any]:
        """Decompose a prompt into structured intent."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/decompose",
            json=_compact({"prompt": prompt, "context": context}),
        )

    async def interrogate(
        self, intent: dict[str, Any], *, depth: str | None = None
    ) -> dict[str, Any]:
        """Generate clarifying questions for an intent."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/interrogate",
            json=_compact({"intent": intent, "depth": depth}),
        )

    async def research(
        self, intent: dict[str, Any], *, context: Any | None = None
    ) -> dict[str, Any]:
        """Research supporting context for an intent."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/research",
            json=_compact({"intent": intent, "context": context}),
        )

    async def specify(
        self,
        intent: dict[str, Any],
        *,
        questions: list[dict[str, Any]] | None = None,
        research: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Build a specification from intent, questions, research, and context."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/specify",
            json=_compact(
                {
                    "intent": intent,
                    "questions": questions,
                    "research": research,
                    "context": context,
                }
            ),
        )

    async def validate(self, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate a prompt-engine specification."""
        return await self._client.request(
            "POST",
            "/api/prompt-engine/validate",
            json={"specification": specification},
        )
