"""Every model a default can reach must be a priced, active catalog row."""

import pytest
from aragora.models.catalog import spec_or_none
from aragora.models.upgrade_map import resolve_model_id

# Definers whose value is deliberately NOT a catalog id: a native provider's
# own model code, for a family the catalog carries only as an OpenRouter row
# (``ModelSpec.direct_id`` is a placeholder there, not a code the native
# endpoint would accept). The literal still has to RESOLVE to an active,
# priced row -- that is what keeps cost accounting honest -- but it is
# expected to have no catalog row of its own. The assertion below is written
# so the entry must be deleted from this dict the day the catalog gains a
# real native row for it.
_NATIVE_ID_EXEMPT = {
    "registry.qwen-cli": "native-provider model code; catalog carries only the OpenRouter row",
    "registry.deepseek-cli": "native-provider model code; catalog carries only the OpenRouter row",
    "registry.kimi-legacy": "native-provider model code; catalog carries only the OpenRouter row",
}


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
        out.append((f"reviewer.{fam}", slug))
    for m in qe._CODEX_DEFAULT_MODELS:
        out.append(("codex_default", m))
    # The claude CLI reviewer's --model literal (hardcoded, not imported from
    # aragora.config.model_pins -- see quorum_evidence.py's own comment) needs
    # its own direct drift guard since it isn't a member of either dict/tuple
    # above.
    out.append(("quorum_evidence._FABLE_51_DIRECT", qe._FABLE_51_DIRECT))

    # The three per-provider OpenRouter fallback maps. These live on the live
    # server path and were the last unmigrated model tables in the repo.
    from aragora.agents.api_agents.openrouter import OPENROUTER_FALLBACK_MODELS
    from aragora.server.handlers.agents import agents as agents_handler
    from aragora.server.stream import debate_executor

    for provider, slug in debate_executor._OPENROUTER_FALLBACK_MODELS.items():
        out.append((f"debate_executor.fallback.{provider}", slug))
    out.append(
        (
            "debate_executor.generic_fallback",
            debate_executor._OPENROUTER_GENERIC_FALLBACK_MODEL,
        )
    )
    for provider, slug in agents_handler._OPENROUTER_FALLBACK_MODELS.items():
        out.append((f"agents_handler.fallback.{provider}", slug))
    # Only the VALUES: the keys are deliberately legacy/retired spellings a
    # caller may still pin, and the point of the table is to route them to a
    # live model.
    for primary, slug in OPENROUTER_FALLBACK_MODELS.items():
        out.append((f"openrouter.fallback_for[{primary}]", slug))

    # Cold-start routing roster. Before the frontier refresh this list was
    # 100% retired spellings, so a cold-start debate could not see the
    # current frontier at all.
    from aragora.routing.provider_router import DEFAULT_PROVIDER_ORDER

    for m in DEFAULT_PROVIDER_ORDER:
        out.append(("provider_router.DEFAULT_PROVIDER_ORDER", m))

    # Native-provider entry points (see _NATIVE_ID_EXEMPT): listed so the gap
    # is named and pinned rather than merely absent from this test.
    import aragora.agents.api_agents.openrouter  # noqa: F401 - registers kimi-legacy
    import aragora.agents.cli_agents  # noqa: F401 - registers the CLI agents
    from aragora.agents.registry import AgentRegistry

    for agent_type in ("qwen-cli", "deepseek-cli", "kimi-legacy"):
        registered = AgentRegistry.get_spec(agent_type)
        assert registered is not None, f"{agent_type} is no longer registered"
        out.append((f"registry.{agent_type}", registered.default_model))
    return out


@pytest.mark.parametrize("where,model_id", _reachable_defaults())
def test_reachable_default_is_priced_and_active(where: str, model_id: str) -> None:
    raw = spec_or_none(model_id)
    if where in _NATIVE_ID_EXEMPT:
        assert raw is None, (
            f"{where}: {model_id!r} now has a catalog row of its own -- "
            f"drop it from _NATIVE_ID_EXEMPT ({_NATIVE_ID_EXEMPT[where]})"
        )
    # The raw spelling must not itself be a retired row. Without this,
    # "not spec.retired" below is unreachable and the test cannot fail on the
    # condition it names: resolve_model_id() returns an active row on every
    # branch (UPGRADES targets are asserted active in test_upgrade_map,
    # branch 2 returns an active row, branch 3 returns a family frontier), so
    # a definer literally pinned to a retired id passed silently.
    assert raw is None or not raw.retired, (
        f"{where}: {model_id!r} is a retired spelling; it only passes because "
        f"resolve_model_id() upgrades it to {resolve_model_id(model_id)!r}"
    )
    spec = spec_or_none(resolve_model_id(model_id))
    assert spec is not None, f"{where}: {model_id!r} has no catalog row"
    assert not spec.retired, f"{where}: {model_id!r} is retired"
    assert spec.input_per_mtok > 0 and spec.output_per_mtok > 0, f"{where}: {model_id!r} unpriced"
