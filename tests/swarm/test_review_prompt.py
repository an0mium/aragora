"""The reviewer prompt must not induce '[P1] None:' no-finding lines.

#8508 fixed the gate scanner to ignore explicit no-finding heads, but that relies
on a fixed list of phrasings. The deeper fix is at the source: tell the reviewer
to OMIT priority levels with no finding instead of writing '[P1] None'. This
hardens against no-finding phrasings the scanner's list doesn't enumerate.
"""

from __future__ import annotations

from aragora.swarm.quorum_evidence import build_review_prompt


def _prompt() -> str:
    return build_review_prompt(
        repo="owner/repo",
        pr=1,
        head_sha="a" * 40,
        diff_text="diff --git a/x.py b/x.py\n+print('x')\n",
    )


def test_prompt_instructs_omitting_empty_priority_levels():
    low = _prompt().lower()
    assert "omit" in low
    # explicitly tells the reviewer not to emit a no-finding [Pn] line
    assert "never write" in low
    assert "none" in low  # references the "[Pn] None" anti-pattern it forbids


def test_prompt_gives_a_no_findings_path():
    assert "no findings" in _prompt().lower()


def test_prompt_still_requires_verdict_and_tag_format():
    p = _prompt()
    assert "Verdict: PASS" in p
    assert "Verdict: CHANGES-REQUESTED" in p
    assert "[P1]" in p  # real findings are still tagged


def test_prompt_drops_old_state_explicitly_phrasing():
    # The old "or state explicitly that there are no blocking issues" wording is
    # what models turned into "[P1] None:" -- it must be gone.
    assert "state explicitly that there are no blocking issues" not in _prompt().lower()
