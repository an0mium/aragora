"""Comment verdict parsing helpers for review-queue evidence gates."""

from __future__ import annotations

import re

_REASONING_TAG_NAMES = "think|thinking|reasoning|thought|scratchpad|analysis"
_REASONING_TAG_RE = re.compile(rf"</?\s*(?:{_REASONING_TAG_NAMES})\s*>", re.I)

# Explicit "no Pn finding" heads. A `[P0]`/`[P1]`/`[P2]` line is blocking UNLESS the text
# before its first colon is EXACTLY one of these — models emit `"[P1] None:"`,
# `"[P1] N/A"`, `"[P1] no issues: ..."` to declare the absence of a finding. Matched
# exactly (not as a prefix) so a real finding that merely starts with "none"/"no"
# (e.g. "[P1] None of the inputs are validated") still blocks.
_NO_FINDING_HEADS = frozenset(
    {
        "none",
        "none found",
        "none identified",
        "none noted",
        "none here",
        "n/a",
        "na",
        "nil",
        "0",
        "zero",
        "false",
        "[]",
        "not applicable",
        "no issues",
        "no issue",
        "no findings",
        "no finding",
        "no blockers",
        "no blocker",
        "no blocking",
        "no blocking findings",
        "no concerns",
        "no concern",
        "no critical issues",
        "no critical issue",
        "no critical findings",
    }
)

# Prefixes a `Verdict:/Decision:/Recommendation:` value may start with to count as a
# *negative* verdict line. Shared by ``has_blocking_or_negative_verdict``.
_NEGATIVE_VERDICT_PREFIXES = (
    "fail",
    "failed",
    "failing",
    "fails",
    "failure",
    "block",
    "blocked",
    "blocking",
    "request changes",
    "request_changes",
    "changes requested",
    "reject",
    "rejected",
    "not ready",
    "needs repair",
)

_BLOCKING_DISSENT_PHRASE_RE = re.compile(
    r"\b(?:do\s+not|don't|must\s+not|cannot|can't|should\s+not)\s+"
    r"(?:merge|ship|land)\b"
    r"|"
    r"\b(?:unsafe|not\s+safe)\s+to\s+(?:merge|ship|land)\b"
    r"|"
    r"\bneeds\s+revision\s+before\s+(?:merge|ship|land)\b"
    r"|"
    r"\b(?:security\s+hole|auth(?:entication)?\s+bypass|sql\s*injection)\b",
    re.I,
)
_BENIGN_BLOCKING_PHRASE_TAIL_RE = re.compile(
    r"^[\s.,;:!?)\]—–-]*"
    r"(?:found|identified|detected|present|observed|noted|remaining)?"
    r"(?:[\s.,;:!?)\]—–-]+(?:in|by|during|across|for|on)\s+"
    r"(?:tests?|coverage|regressions?|fixtures?|guardrails?|cases?|"
    r"exact\s+head|this\s+diff|the\s+diff|diff|code|review|"
    r"endpoint|path|flow|handler|route|parser|implementation))?"
    r"[\s.,;:!?)\]—–-]*$",
    re.I,
)
_BENIGN_REVIEWED_PHRASE_TAIL_RE = re.compile(
    r"^[\s.,;:!?)\]—–-]*"
    r"(?:tests?|coverage|regressions?|fixtures?|guardrails?|cases?)\b",
    re.I,
)

# Prefixes a populated Blocker-label value may start with that mean "no blocker".
# NOTE: a bare ``"no"`` is intentionally absent. ``"no"`` alone matched real
# security findings such as "no authentication on admin endpoint" / "no validation"
# / "no rate limiting" and silently demoted them to advisory — a merge-gate bypass.
# A leading ``"no"`` only counts as non-blocking when it is immediately followed by
# a no-finding NOUN (issue(s)/finding(s)/blocker(s)/concern(s)/problem(s)/change(s)),
# handled by :data:`_NO_FINDING_NO_PHRASE`.
_NON_BLOCKING_PREFIXES = (
    "none",
    "none found",
    "not found",
    "0",
    "zero",
    "false",
    "n/a",
    "not applicable",
    "[]",
)

# A populated Blocker-label value that is a no-finding declaration: "no issues",
# "no blockers", "no concerns", optionally hedged with a closed set of adjectives
# ("no MAJOR concerns", "no SIGNIFICANT issues", "no REMAINING blockers"). The
# adjective allowlist is bounded so the match stays fail-CLOSED for real findings:
# "no authentication", "no validation", "no SQLi" do NOT match (their head word is
# a real subject, not a no-finding noun, and is not an allowlisted hedge), so they
# still block. The standalone "blocking" alternative is intentionally absent — it
# made "no blocking on the auth path but SQLi" match as non-blocking, re-opening the
# exact bypass class this guard exists to close. The legacy "no blocking findings"
# form is still covered by the optional ``blocking\s+`` prefix on ``findings?``.
# NOTE (bounded heuristic): a trailing "… but <real finding>" after a matched
# no-finding phrase is not parsed out (prefix match), so "no major concerns but X"
# reads as non-blocking; this is a pre-existing limit of label-value prefix matching
# (the prior bare-"no" was strictly more permissive) and is low-realism in a Blocker
# label. Substantive blockers belong in the label value itself, where they block.
_NO_FINDING_HEDGE = (
    r"(?:major|minor|significant|serious|critical|real|actual|further|other|"
    r"remaining|additional|outstanding|notable|material|known|new|obvious)"
)
_NO_FINDING_NO_PHRASE = re.compile(
    r"no\s+"
    rf"(?:{_NO_FINDING_HEDGE}\s+){{0,2}}"
    r"(?:blocking\s+)?"
    r"(?:issues?|findings?|blockers?|concerns?|problems?|changes?)(?!\w)",
    re.I,
)

# What may follow a matched no-finding phrase for the value to still read as a PURE
# no-finding declaration: only benign closers ("found"/"identified"/"noted"/…) and
# punctuation/whitespace. Anything else (notably a contrastive "… but <real
# finding>") means the value carries a substantive blocker and must NOT be demoted.
_NO_FINDING_TAIL = re.compile(
    r"[\s.,;:!?)\]—–-]*"
    r"(?:found|identified|noted|detected|seen|present|reported|observed|raised|flagged|"
    r"needed|required|necessary|requested|warranted|remaining|here|at all|whatsoever)?"
    r"[\s.,;:!?)\]—–-]*",
    re.I,
)
_NO_FINDING_MARKER_TAIL = re.compile(
    r"^[\s.,;:!?)\]—–-]*(?:blocking\s+)?"
    r"(?:issues?|findings?|blockers?|concerns?|problems?|changes?)(?!\w)"
    r"(?P<tail>.*)$",
    re.I,
)

_BLOCKER_LABELS = frozenset({"blocking finding", "blocking findings", "blocker", "blockers"})


def _starts_with_phrase(
    value: str, phrases: tuple[str, ...], *, match_no_finding: bool = False
) -> bool:
    # ``_NO_FINDING_NO_PHRASE`` ("no issues" / "no blockers" / …) means "absence of
    # a blocker" ONLY in the Blocker-label non-blocking check (``match_no_finding=True``).
    # It MUST NOT short-circuit the negative-verdict check: a *positive* verdict such
    # as "Verdict: no concerns" / "Recommendation: no changes needed" / "Decision: no
    # blockers" would otherwise be promoted to a blocking dissent — a regression in the
    # default flag-OFF merge-gate path. So the no-finding match is opt-in per caller.
    if match_no_finding:
        # Fail-CLOSED for EVERY no-finding token (the no-<noun> phrase AND each legacy
        # ``_NON_BLOCKING_PREFIXES`` entry like "none"/"n/a"/"zero"): the token must be
        # essentially the WHOLE value. Only benign trailing words ("found"/"identified"/
        # …) and punctuation may follow. A contrastive continuation leaves substantive
        # content in the tail ("none but SQLi on L40", "n/a - auth bypass remains", "no
        # major concerns BUT X"), so it does NOT read as no-finding and the Blocker
        # label still blocks — closing the prefix-bypass class (claude/grok/openai #8574).
        m = _NO_FINDING_NO_PHRASE.match(value)
        if m and _NO_FINDING_TAIL.fullmatch(value[m.end() :]):
            return True
        for phrase in phrases:
            pm = re.match(rf"{re.escape(phrase)}(?!\w)", value)
            if pm and _NO_FINDING_TAIL.fullmatch(value[pm.end() :]):
                return True
        return False
    return any(re.match(rf"{re.escape(phrase)}(?!\w)", value) for phrase in phrases)


# The DEFAULT (flag-OFF) blocking scanner treats `[P0]`/`[P1]`/`[P2]` marker lines as
# blocking findings — #8555 added `[P2]` to the default merge gate. ``has_blocking_or_
# negative_verdict`` (the flag-OFF path) uses these so flag-OFF stays byte-identical to
# main.
_PRIORITY_MARKER_WRAPPER = r"(?:[`'\"])?"
_DEFAULT_BLOCKING_MARKER_TOKEN = (
    rf"{_PRIORITY_MARKER_WRAPPER}(?:\*\*)?\[(?:p0|p1|p2)\](?:\*\*)?"
    rf"{_PRIORITY_MARKER_WRAPPER}"
)
_DEFAULT_BLOCKING_MARKER = re.compile(rf"^{_DEFAULT_BLOCKING_MARKER_TOKEN}(?:\s|$|[:.;—–-])", re.I)
_DEFAULT_BLOCKING_MARKER_STRIP = re.compile(rf"^{_DEFAULT_BLOCKING_MARKER_TOKEN}\s*", re.I)
_DEFAULT_BLOCKING_MARKER_ANYWHERE = re.compile(
    rf"(?:^|[\s;:,.()[\]{{}}—–`'\"\-])(?P<marker>{_DEFAULT_BLOCKING_MARKER_TOKEN})"
    r"(?:\s|$|[:.;—–`'\"\-])",
    re.I,
)
_BENIGN_PARENTHETICAL_PRIORITY_REFERENCE_RE = re.compile(
    r"^(?:"
    r"(?:tracked|logged|filed|classified|noted)\s+(?:as|under)\s+"
    r"(?:\*\*)?\[(?:p0|p1|p2)\](?:\*\*)?"
    r"(?:\s+(?:in|on|for)\s+(?:the\s+)?(?:backlog|follow(?:\s|-)?up|queue))?"
    r"|"
    r"(?:backlog|follow(?:\s|-)?up|queue)\s+"
    r"(?:tracked|logged|filed|classified|noted)\s+(?:as|under)\s+"
    r"(?:\*\*)?\[(?:p0|p1|p2)\](?:\*\*)?"
    r")"
    r"[\s.,;:!—–-]*$",
    re.I,
)

# The SEVERITY-GATE marker is deliberately `[P0]`/`[P1]` ONLY. ``_PRIORITY_MARKER``
# detects a marker line (sev captured); ``_PRIORITY_MARKER_STRIP`` removes the prefix to
# expose the finding head. Shared by ``_priority_finding_severity`` /
# ``highest_blocking_severity`` / ``has_blocking_finding_or_label`` — the flag-ON path.
# It intentionally diverges from ``_DEFAULT_BLOCKING_MARKER``: with
# ARAGORA_ENABLE_SEVERITY_GATED_DISSENT=1 a `[P2]`-only dissent is NOT a blocking
# severity, so it becomes advisory (that is the whole point of the flag). `[P2]` blocks
# by default and is advisory under the flag — these two marker sets encode exactly that.
_PRIORITY_MARKER_TOKEN = (
    rf"{_PRIORITY_MARKER_WRAPPER}(?:\*\*)?\[(?P<sev>p0|p1)\](?:\*\*)?"
    rf"{_PRIORITY_MARKER_WRAPPER}"
)
_PRIORITY_MARKER = re.compile(rf"^{_PRIORITY_MARKER_TOKEN}(?:\s|$|[:.;—–-])", re.I)
_PRIORITY_MARKER_STRIP = re.compile(
    rf"^{_PRIORITY_MARKER_WRAPPER}(?:\*\*)?\[(?:p0|p1)\](?:\*\*)?"
    rf"{_PRIORITY_MARKER_WRAPPER}\s*",
    re.I,
)
_PRIORITY_MARKER_ANYWHERE = re.compile(
    rf"(?:^|[\s;:,.()[\]{{}}—–`'\"\-])(?P<marker>{_PRIORITY_MARKER_TOKEN})"
    r"(?:\s|$|[:.;—–`'\"\-])",
    re.I,
)


def _strip_decoration(text: str) -> str:
    return re.sub(r"^(?:[#>\-*+\s]+|\d+[.)]\s+)+", "", text.strip())


def _normalize_value(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"[-_]+", " ", text)
    return re.sub(r"\s+", " ", text.strip().strip("*_").strip().lower())


def _split_reasoning_tags_for_scan(text: str) -> str:
    return _REASONING_TAG_RE.sub("\n", str(text or ""))


_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)\S")


def _is_markdown_indented_code_line(line: str) -> bool:
    return bool(_INDENTED_CODE_RE.match(line.rstrip("\r\n")))


def _priority_finding_severity(stripped: str) -> str | None:
    """Return ``"P0"``/``"P1"`` if ``stripped`` is a *real* `[P0]`/`[P1]` finding
    line (head-before-colon NOT in :data:`_NO_FINDING_HEADS`), else ``None``.

    Reuses the exact decoration/normalization/head-extraction logic that
    :func:`has_blocking_or_negative_verdict` applies, so the severity scanner and
    the blocking scanner never disagree on what a real finding line is.
    """
    priority_marker_line = _strip_decoration(stripped)
    marker = _PRIORITY_MARKER.match(priority_marker_line)
    if not marker:
        return None
    rest = _PRIORITY_MARKER_STRIP.sub("", priority_marker_line)
    head = _normalize_value(rest).split(":", 1)[0].strip(" .;—–-")
    if head in _NO_FINDING_HEADS:
        # explicit "[Pn] None:/N/A/no issues" non-finding
        return None
    return marker.group("sev").upper()


def _priority_finding_severity_anywhere(text: str) -> str | None:
    best: str | None = None
    for match in _PRIORITY_MARKER_ANYWHERE.finditer(text):
        if _inside_benign_parenthetical_priority_reference(text, match.start("marker")):
            continue
        if _marker_inside_explicit_no_finding_phrase(
            text, match.start("marker"), match.end("marker")
        ):
            continue
        severity = _priority_finding_severity(text[match.start("marker") :])
        if severity == "P0":
            return "P0"
        if severity == "P1":
            best = "P1"
    return best


def _default_blocking_marker_finding(stripped: str) -> bool:
    priority_marker_line = _strip_decoration(stripped)
    marker = _DEFAULT_BLOCKING_MARKER.match(priority_marker_line)
    if not marker:
        return False
    rest = _DEFAULT_BLOCKING_MARKER_STRIP.sub("", priority_marker_line)
    head = _normalize_value(rest).split(":", 1)[0].strip(" .;—–-")
    return head not in _NO_FINDING_HEADS


def _blocking_dissent_phrase_match_is_benign(value: str, match: re.Match[str]) -> bool:
    prefix = value[: match.start()].strip()
    suffix = value[match.end() :].strip()
    if re.search(r"(?:^|[.;:!?]\s*)no(?:\s+(?:known|remaining|actual))?$", prefix):
        return bool(_BENIGN_BLOCKING_PHRASE_TAIL_RE.match(suffix))
    if re.search(r"(?:^|[.;:!?]\s*)(?:i\s+)?reviewed(?:\s+the)?$", prefix):
        return bool(_BENIGN_REVIEWED_PHRASE_TAIL_RE.match(suffix))
    return False


def _marker_inside_explicit_no_finding_phrase(
    text: str, marker_start: int, marker_end: int
) -> bool:
    """Return True for prose such as ``no [P1] findings``.

    Same-line marker scanning must still catch ``Verdict: PASS; [P1] auth
    bypass`` and ``no [P1] auth bypass``. This exemption is limited to an
    explicit ``no`` + optional no-finding hedge before the marker and a
    no-finding noun immediately after it.
    """

    line_start = text.rfind("\n", 0, marker_start) + 1
    line_end = text.find("\n", marker_start)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    relative_start = marker_start - line_start
    relative_end = marker_end - line_start
    prefix = _normalize_value(line[:relative_start])
    suffix = _normalize_value(line[relative_end:])
    if not re.search(
        rf"(?:^|[.;:!?]\s*)no(?:\s+{_NO_FINDING_HEDGE}){{0,2}}$",
        prefix,
        re.I,
    ):
        return False
    tail_match = _NO_FINDING_MARKER_TAIL.match(suffix)
    if not tail_match:
        return False
    return bool(_NO_FINDING_TAIL.fullmatch(tail_match.group("tail")))


def _has_blocking_dissent_phrase(text: str) -> bool:
    value = _normalize_value(text)
    for match in _BLOCKING_DISSENT_PHRASE_RE.finditer(value):
        if _blocking_dissent_phrase_match_is_benign(value, match):
            continue
        return True
    return False


def _untrusted_dissent_phrase_is_benign_example(
    raw_line: str, stripped: str, *, in_fence: bool
) -> bool:
    """Whether an untrusted prose blocker is only a quoted/code example.

    The scanner still fails closed for concrete merge/security dissent in
    blockquotes, indented code, or fences. The exemption is intentionally narrow:
    existing fixture/example prose stays non-blocking, while real secondary review
    lines such as ``> Do not merge until auth is fixed`` still block.
    """

    normalized = _normalize_value(stripped)
    if in_fence:
        return bool(
            re.search(
                r"\b(?:fixture|example|sample|literal|assert|expected|mock|stub)\b",
                normalized,
            )
            or re.search(r"[`'\"]", stripped)
            or re.search(r"\b(?:assert|return|raise|print|const|let|var|def|class)\b", stripped)
        )
    if stripped.startswith(">") or _is_markdown_indented_code_line(raw_line):
        if not re.search(r"\b(?:fixture|example|sample|literal)\b", normalized):
            return False
        if re.search(r"\b(?:auth(?:entication)?|bypass|security|sql|injection)\b", normalized):
            return False
        return True
    return False


def _inside_benign_parenthetical_priority_reference(text: str, marker_start: int) -> bool:
    """Return True for parenthetical severity references, not findings.

    Evidence bodies sometimes say things like ``Verdict: PASS (tracked as [P2] in
    backlog)``. The marker is metadata about another queue item, not a concrete
    finding. Keep this exemption deliberately narrow: the marker must be inside a
    same-line parenthetical and that parenthetical must be a whole metadata
    phrase. A parenthetical such as ``([P1] issue: auth bypass)`` or
    ``(see issue [P1] auth bypass)`` still blocks.
    """

    line_start = text.rfind("\n", 0, marker_start) + 1
    line_end = text.find("\n", marker_start)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    relative_start = marker_start - line_start
    open_idx = line.rfind("(", 0, relative_start)
    close_idx = line.find(")", relative_start)
    if open_idx == -1 or close_idx == -1:
        return False
    parenthetical = line[open_idx + 1 : close_idx]
    return bool(
        _BENIGN_PARENTHETICAL_PRIORITY_REFERENCE_RE.fullmatch(_normalize_value(parenthetical))
    )


def _has_default_blocking_marker_anywhere(text: str) -> bool:
    for match in _DEFAULT_BLOCKING_MARKER_ANYWHERE.finditer(text):
        if _inside_benign_parenthetical_priority_reference(text, match.start("marker")):
            continue
        if _marker_inside_explicit_no_finding_phrase(
            text, match.start("marker"), match.end("marker")
        ):
            continue
        candidate = text[match.start("marker") :]
        if _default_blocking_marker_finding(candidate):
            return True
    return False


def _populated_blocker_label(stripped: str, follow_lines: list[str]) -> bool:
    """Whether ``stripped`` is a populated ``Blocking finding(s):/Blocker(s):`` label.

    ``follow_lines`` are the remaining stripped (non-empty filtered happens here)
    lines after this one, used to resolve a label whose value is on the next line.
    Mirrors the Blocker-label branch of :func:`has_blocking_or_negative_verdict`.
    """
    line = _strip_decoration(stripped).replace("**", "").replace("__", "")
    match = re.match(r"^(?P<label>[^:—–-]+?)\s*(?::|—|–|-)\s*(?P<value>.*)$", line)
    if not match:
        return False
    normalized_label = re.sub(r"\s+", " ", match.group("label").strip().lower())
    normalized_label = normalized_label.strip("*_ ")
    if normalized_label not in _BLOCKER_LABELS:
        return False
    normalized_value = _normalize_value(match.group("value"))
    candidate = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", normalized_value)
    if candidate in {"-", "*", "[]", "[ ]", "—", "–"}:
        return False
    if not candidate:
        follow = next((entry for entry in follow_lines if entry), "")
        is_list_item = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", follow))
        if not is_list_item and (follow.startswith("#") or re.match(r"^[^:]+?:\s+\S", follow)):
            return False
        candidate = _normalize_value(_strip_decoration(follow))
    return bool(candidate) and not _starts_with_phrase(
        candidate, _NON_BLOCKING_PREFIXES, match_no_finding=True
    )


def has_blocking_or_negative_verdict(body: str) -> bool:
    """Return True for explicit evidence comments that report blockers."""
    raw_lines = _split_reasoning_tags_for_scan(str(body or "")).splitlines()
    lines = [raw_line.strip() for raw_line in raw_lines]
    in_fence = False
    fence_marker = ""
    for idx, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fence = re.match(r"^(```|~~~)", stripped)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if _has_blocking_dissent_phrase(
            stripped
        ) and not _untrusted_dissent_phrase_is_benign_example(
            raw_line, stripped, in_fence=in_fence
        ):
            return True
        if _default_blocking_marker_finding(stripped):
            return True
        if _DEFAULT_BLOCKING_MARKER.match(_strip_decoration(stripped)):
            # explicit "[Pn] None:/N/A/no issues" non-finding -> keep scanning
            continue
        line = _strip_decoration(stripped).replace("**", "").replace("__", "")
        match = re.match(r"^(?P<label>[^:—–-]+?)\s*(?::|—|–|-)\s*(?P<value>.*)$", line)
        if not match:
            continue
        normalized_label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        normalized_label = normalized_label.strip("*_ ")
        normalized_value = _normalize_value(match.group("value"))
        if normalized_label in {"verdict", "decision", "recommendation"}:
            if _has_default_blocking_marker_anywhere(match.group("value")):
                return True
            if _starts_with_phrase(normalized_value, _NEGATIVE_VERDICT_PREFIXES):
                return True
            continue
        if normalized_label not in _BLOCKER_LABELS:
            continue
        candidate = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", normalized_value)
        if candidate in {"-", "*", "[]", "[ ]", "—", "–"}:
            continue
        if not candidate:
            follow = next((entry for entry in lines[idx + 1 :] if entry), "")
            is_list_item = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", follow))
            if not is_list_item and (follow.startswith("#") or re.match(r"^[^:]+?:\s+\S", follow)):
                continue
            candidate = _normalize_value(_strip_decoration(follow))
        if candidate and not _starts_with_phrase(
            candidate, _NON_BLOCKING_PREFIXES, match_no_finding=True
        ):
            return True
    return False


def has_default_blocking_finding_or_label(body: str) -> bool:
    """Return True for a real default-gate `[P0]`/`[P1]`/`[P2]` finding or Blocker label.

    This deliberately excludes bare negative ``Verdict:`` lines while reusing the
    same no-finding head and Blocker-label parsing as
    :func:`has_blocking_or_negative_verdict`.
    """
    lines = [
        raw_line.strip()
        for raw_line in _split_reasoning_tags_for_scan(str(body or "")).splitlines()
    ]
    for idx, stripped in enumerate(lines):
        if not stripped:
            continue
        if _has_default_blocking_marker_anywhere(stripped):
            return True
        priority_marker_line = _strip_decoration(stripped)
        if _DEFAULT_BLOCKING_MARKER.match(priority_marker_line):
            rest = _DEFAULT_BLOCKING_MARKER_STRIP.sub("", priority_marker_line)
            head = _normalize_value(rest).split(":", 1)[0].strip(" .;—–-")
            if head not in _NO_FINDING_HEADS:
                return True
            continue
        if _populated_blocker_label(stripped, lines[idx + 1 :]):
            return True
    return False


def highest_blocking_severity(body: str) -> str | None:
    """Return ``"P0"``/``"P1"`` if ``body`` carries a real (non-:data:`_NO_FINDING_HEADS`)
    `[P0]`/`[P1]` finding line, else ``None``.

    Reuses the EXACT `[P0]`/`[P1]` detection that drives
    :func:`has_blocking_or_negative_verdict`. ``"P0"`` is reported in preference to
    ``"P1"`` when both are present.
    """
    best: str | None = None
    for raw_line in _split_reasoning_tags_for_scan(str(body or "")).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        severity = _priority_finding_severity_anywhere(stripped)
        if severity == "P0":
            return "P0"
        if severity == "P1":
            best = "P1"
    return best


def has_blocking_finding_or_label(body: str) -> bool:
    """Return True when ``body`` carries a real `[P0]`/`[P1]` finding line OR a
    populated Blocker-label.

    This is everything :func:`has_blocking_or_negative_verdict` blocks on EXCEPT a
    bare negative ``Verdict:/Decision:/Recommendation:`` line that carries no real
    finding and no populated Blocker-label. It is the severity-gated trigger: a
    ``CHANGES-REQUESTED`` comment promotes a *blocking* dissent only when it is
    backed by a real `[P0]`/`[P1]` finding or a populated Blocker label.
    """
    lines = [
        raw_line.strip()
        for raw_line in _split_reasoning_tags_for_scan(str(body or "")).splitlines()
    ]
    for idx, stripped in enumerate(lines):
        if not stripped:
            continue
        if _priority_finding_severity_anywhere(stripped) is not None:
            return True
        if _populated_blocker_label(stripped, lines[idx + 1 :]):
            return True
    return False
