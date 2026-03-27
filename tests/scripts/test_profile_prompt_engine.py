from __future__ import annotations

from aragora.prompt_engine.timing import OperationTiming, PipelineTiming
from scripts.profile_prompt_engine import build_profile_report


def test_build_profile_report_aggregates_stage_and_operation_baselines() -> None:
    report = build_profile_report(
        prompt="Profile the prompt engine",
        timings=[
            PipelineTiming(
                total_duration_ms=120.0,
                stage_durations_ms={"decompose": 20.0, "research": 70.0},
                operation_timings=[
                    OperationTiming("decompose.agent_generate", 15.0, category="llm"),
                    OperationTiming("research.agent_generate", 50.0, category="llm"),
                ],
            ),
            PipelineTiming(
                total_duration_ms=80.0,
                stage_durations_ms={"decompose": 10.0, "research": 50.0},
                operation_timings=[
                    OperationTiming("decompose.agent_generate", 8.0, category="llm"),
                    OperationTiming("research.agent_generate", 30.0, category="llm"),
                ],
            ),
        ],
        profile="founder",
        depth="quick",
        skip_research=False,
        skip_interrogation=False,
        mode="mock-agent",
    )

    assert report["runs"] == 2
    assert report["aggregate"]["total_duration_ms"]["mean"] == 100.0
    assert report["aggregate"]["stage_duration_ms"]["research"]["median"] == 60.0
    assert report["aggregate"]["top_operations"][0]["operation"] == "research.agent_generate"
