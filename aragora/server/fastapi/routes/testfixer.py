"""TestFixer API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory, FixTarget
from aragora.nomic.testfixer.http_api import (
    TestFixerRunConfig,
    analyze_failure,
    apply_proposal,
    propose_fix,
    run_fix_loop,
    save_proposal,
)
from aragora.nomic.testfixer.runner import TestFailure

router = APIRouter(prefix="/api/testfixer", tags=["TestFixer"])


class FailurePayload(BaseModel):
    test_file: str
    test_name: str
    error_type: str
    error_message: str
    stack_trace: str = ""
    line_number: int | None = None
    relevant_code: str = ""


class RunRequest(BaseModel):
    repo_path: str
    test_command: str
    agents: list[str] = Field(default_factory=lambda: ["codex", "claude"])
    max_iterations: int = 10
    min_confidence: float = 0.5
    timeout_seconds: float = 300.0
    attempt_store_path: str | None = None
    artifacts_dir: str | None = None
    enable_diagnostics: bool = True


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and (text.startswith("{") or text.startswith("[")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


async def _load_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    data: Any
    if "application/json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)
    if isinstance(data, dict):
        return {key: _maybe_json(value) for key, value in data.items()}
    return {"payload": _maybe_json(data)}


def _normalize_agents(raw_agents: Any) -> list[str]:
    if not raw_agents:
        return ["codex", "claude"]
    if isinstance(raw_agents, str):
        return [part.strip() for part in raw_agents.split(",") if part.strip()]
    if isinstance(raw_agents, (list, tuple)):
        return [str(item).strip() for item in raw_agents if str(item).strip()]
    return ["codex", "claude"]


def _coerce_category(value: Any) -> FailureCategory:
    if isinstance(value, FailureCategory):
        return value
    if isinstance(value, str):
        try:
            return FailureCategory(value)
        except ValueError:
            return FailureCategory.UNKNOWN
    return FailureCategory.UNKNOWN


def _coerce_fix_target(value: Any) -> FixTarget:
    if isinstance(value, FixTarget):
        return value
    if isinstance(value, str):
        try:
            return FixTarget(value)
        except ValueError:
            return FixTarget.TEST_FILE
    return FixTarget.TEST_FILE


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@router.post("/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    payload = await _load_payload(request)
    failure_data = payload.get("failure", payload)
    failure_data = _maybe_json(failure_data)
    if not isinstance(failure_data, dict):
        raise HTTPException(status_code=400, detail="failure payload must be an object")

    repo_path = payload.get("repo_path") or failure_data.get("repo_path")
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path is required")

    failure = TestFailure(
        test_name=failure_data.get("test_name", ""),
        test_file=failure_data.get("test_file", ""),
        error_type=failure_data.get("error_type", "Unknown"),
        error_message=failure_data.get("error_message", ""),
        stack_trace=failure_data.get("stack_trace", ""),
        line_number=failure_data.get("line_number"),
        relevant_code=failure_data.get("relevant_code", ""),
    )
    result = await analyze_failure(Path(repo_path), failure)
    return {"repo_path": repo_path, "analysis": result["analysis_dict"]}


@router.post("/propose")
async def propose(request: Request) -> dict[str, Any]:
    payload = await _load_payload(request)
    analysis_data = payload.get("analysis", payload)
    analysis_data = _maybe_json(analysis_data)
    if not isinstance(analysis_data, dict):
        raise HTTPException(status_code=400, detail="analysis payload must be an object")

    repo_path = payload.get("repo_path") or analysis_data.get("repo_path")
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path is required")

    failure_dict = analysis_data.get("failure") or {}
    failure = TestFailure(
        test_name=failure_dict.get("test_name", ""),
        test_file=failure_dict.get("test_file", ""),
        error_type=failure_dict.get("error_type", ""),
        error_message=failure_dict.get("error_message", ""),
        stack_trace=failure_dict.get("stack_trace", ""),
    )
    analysis = FailureAnalysis(failure=failure)
    analysis.category = _coerce_category(analysis_data.get("category", analysis.category))
    analysis.fix_target = _coerce_fix_target(analysis_data.get("fix_target", analysis.fix_target))
    analysis.confidence = _coerce_confidence(
        analysis_data.get("confidence", analysis.confidence), analysis.confidence
    )
    analysis.root_cause = (
        analysis_data.get("root_cause", analysis.root_cause) or analysis.root_cause
    )
    analysis.root_cause_file = (
        analysis_data.get("root_cause_file", analysis.root_cause_file) or analysis.root_cause_file
    )
    analysis.suggested_approach = (
        analysis_data.get("suggested_approach", analysis.suggested_approach)
        or analysis.suggested_approach
    )

    agents = _normalize_agents(payload.get("agents"))
    proposal = await propose_fix(Path(repo_path), analysis, agents)
    save_proposal(Path(repo_path), proposal)
    return {
        "repo_path": repo_path,
        "proposal": {
            "id": proposal.id,
            "description": proposal.description,
            "confidence": proposal.post_debate_confidence,
            "diff": proposal.as_diff(),
        },
    }


@router.post("/apply")
async def apply_fix(request: Request) -> dict[str, Any]:
    payload = await _load_payload(request)
    proposal_id = payload.get("proposal_id")
    proposal_data = payload.get("proposal")
    if not proposal_id and isinstance(proposal_data, dict):
        proposal_id = proposal_data.get("id")
    repo_path = payload.get("repo_path")
    if not repo_path or not proposal_id:
        raise HTTPException(status_code=400, detail="repo_path and proposal_id are required")
    return apply_proposal(Path(repo_path), proposal_id)


@router.post("/run")
async def run(request: RunRequest) -> dict[str, Any]:
    config = TestFixerRunConfig(
        repo_path=Path(request.repo_path),
        test_command=request.test_command,
        agents=request.agents,
        max_iterations=request.max_iterations,
        min_confidence=request.min_confidence,
        timeout_seconds=request.timeout_seconds,
        attempt_store_path=Path(request.attempt_store_path) if request.attempt_store_path else None,
        artifacts_dir=Path(request.artifacts_dir) if request.artifacts_dir else None,
        enable_diagnostics=request.enable_diagnostics,
    )
    result = await run_fix_loop(config)
    return result.to_dict()
