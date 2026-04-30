from __future__ import annotations

from aragora.heterogeneity.prompts import (
    DEFAULT_PILOT_CLASS_QUOTAS,
    load_prompt_file,
    load_prompt_set,
    select_pilot_prompts,
)


def test_load_prompt_set_reads_50_authored_prompts() -> None:
    prompts = load_prompt_set("tests/heterogeneity/probe_prompts")
    assert len(prompts) == 50
    assert {prompt.prompt_class for prompt in prompts} == set(DEFAULT_PILOT_CLASS_QUOTAS)


def test_load_prompt_file_reads_seeded_error() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/single_seeded_error/01_revert_window_off_by_one.md"
    )
    assert prompt.prompt_id == "sse_01_revert_window_off_by_one"
    assert prompt.seeded_error is not None
    assert "14" in prompt.seeded_error.description


def test_select_pilot_prompts_satisfies_minimum_gate() -> None:
    prompts = load_prompt_set("tests/heterogeneity/probe_prompts")
    selected = select_pilot_prompts(prompts)
    counts = {}
    for prompt in selected:
        counts[prompt.prompt_class] = counts.get(prompt.prompt_class, 0) + 1
    assert len(selected) == sum(DEFAULT_PILOT_CLASS_QUOTAS.values())
    assert counts == DEFAULT_PILOT_CLASS_QUOTAS
    assert all(count >= 2 for count in counts.values())
