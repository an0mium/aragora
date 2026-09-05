"""Every model a default can reach must be a priced, active catalog row."""

import pytest
from aragora.models.catalog import spec_or_none
from aragora.models.upgrade_map import resolve_model_id

# Reviewer families PR 2 (Task 10) still needs to add a catalog row for /
# move the reviewer map for. Literal allow-list (NOT computed from
# spec_or_none) so a genuine regression in any OTHER reviewer/codex_default
# entry cannot silently mask itself as an expected failure -- computing the
# xfail predicate from the same spec_or_none(resolve_model_id(...)) check
# the test itself asserts would make every marked row un-failable and every
# unmarked row un-xfailable by construction, defeating the point of this
# reverse-completeness test.
_PR2_PENDING_FAMILIES = frozenset({"tencent", "bytedance"})


def _reachable_defaults() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    from aragora.config import model_pins as mp

    for role in (
        "proposer",
        "critic",
        "synthesizer",
        "devils_advocate",
        "researcher",
        "reviewer",
        "quality_reviewer",
        "security_auditor",
        "compliance_auditor",
        "judge",
        "default",
    ):
        out.append((f"pins.{role}.direct", mp.direct_model_for_role(role)))
        out.append((f"pins.{role}.openrouter", mp.openrouter_alias_for_role(role)))
    from aragora.agents.model_selector import MODEL_PROFILES

    for name, prof in MODEL_PROFILES.items():
        out.append((f"profile.{name}", prof.model_id))
    from aragora.agents.api_agents import (
        anthropic,
        openai,
        gemini,
        grok,
        mistral,
        openai_compatible,
    )

    for mod in (anthropic, openai, gemini, grok, mistral):
        out.append((f"{mod.__name__}.DEFAULT_MODEL", getattr(mod, "DEFAULT_MODEL")))
    out.append(
        ("openai_compatible.DEFAULT_FALLBACK_MODEL", openai_compatible.DEFAULT_FALLBACK_MODEL)
    )
    from aragora.server.handlers.debates import cost_estimation

    for m in cost_estimation.DEFAULT_MODELS:
        out.append(("cost_estimation.DEFAULT_MODELS", m))
    from aragora.swarm import quorum_evidence as qe

    for fam, slug in qe._OPENROUTER_REVIEWER_MODELS.items():
        where = f"reviewer.{fam}"
        if fam in _PR2_PENDING_FAMILIES:
            out.append(
                pytest.param(
                    where,
                    slug,
                    marks=pytest.mark.xfail(
                        strict=True, reason="PR 2 adds catalog rows / moves the reviewer map"
                    ),
                )
            )
        else:
            out.append((where, slug))
    for m in qe._CODEX_DEFAULT_MODELS:
        out.append(("codex_default", m))
    return out


@pytest.mark.parametrize("where,model_id", _reachable_defaults())
def test_reachable_default_is_priced_and_active(where: str, model_id: str) -> None:
    spec = spec_or_none(resolve_model_id(model_id))
    assert spec is not None, f"{where}: {model_id!r} has no catalog row"
    assert not spec.retired, f"{where}: {model_id!r} is retired"
    assert spec.input_per_mtok > 0 and spec.output_per_mtok > 0, f"{where}: {model_id!r} unpriced"
