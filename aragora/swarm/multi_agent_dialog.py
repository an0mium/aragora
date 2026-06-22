"""Cross-agent dialog harness for parallel agent dispatch.

This module unblocks the 30c-Phase-D limitation: claude code, codex
CLI, and droid CLI can all answer focused review prompts non-
interactively, but only with the correct invocation flags. Previous
rounds tried ``claude --bare`` (requires explicit env-var auth) and
``codex exec`` under default ``reasoning_effort=xhigh`` (extensive
investigation chain that exceeds 90s timeouts). This harness encodes
the working flag set so future rounds always succeed.

Design contract:

- **Pure file-based state**: every dialog round is a JSONL stream;
  no in-memory long-running state, so the harness is restart-safe
  and inspectable post-hoc.
- **Agent failures are isolated**: if one agent fails, the other
  agents' turns still complete and are recorded.
- **No GitHub mutations**: the harness writes only to the local
  round directory provided by the caller.
- **Standard library only**: ``asyncio.create_subprocess_exec``,
  ``json``, ``dataclasses``, ``pathlib``. Zero new dependencies.

Usage::

    from aragora.swarm.multi_agent_dialog import (
        AgentSpec, DialogRound, dispatch_round
    )
    agents = [
        AgentSpec.claude(),
        AgentSpec.codex(),
        AgentSpec.droid(),
    ]
    round_ = DialogRound(round_id="phase-d", prompt="Review this code...")
    transcripts = await dispatch_round(round_, agents, output_dir=Path("..."))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Round 30e Phase C — gauntlet hardening helpers                        #
# --------------------------------------------------------------------- #
#
# All eight fixes from round 30d-Phase-H's gauntlet review of this
# module live here as small, named helpers so the test suite can
# exercise each one independently.

# Match a code-fence opener (```) at the start of a line so we can
# escape adversarial agent output that tries to break out of the
# transcript's fenced ``stdout`` block.
_FENCE_RE: re.Pattern[str] = re.compile(r"^(\s*)(```)", re.MULTILINE)

# Strip CSI / OSC ANSI escapes (color, cursor moves, OSC-8 hyperlinks)
# from agent output before persisting. Two patterns cover the bulk:
#   - ESC [ ... letter (CSI)
#   - ESC ] ... BEL/ST  (OSC)
_ANSI_CSI_RE: re.Pattern[str] = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE: re.Pattern[str] = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Round IDs are interpolated into output filenames; this regex keeps
# them confined to the intended output directory.
_ROUND_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")

# Distinct sentinel returncodes so operators can tell ``binary not
# found`` apart from a genuine CLI rc=-1 failure.
RC_BINARY_NOT_FOUND: int = -127
RC_DISPATCH_ERROR: int = -126

# Cap on persisted stdout/stderr per turn — chatty CLIs can produce
# multi-MB lines that overflow downstream JSONL parsers.
MAX_OUTPUT_BYTES: int = 1_000_000


def _strip_ansi(s: str) -> str:
    """Return ``s`` with CSI and OSC ANSI escape sequences removed."""
    return _ANSI_OSC_RE.sub("", _ANSI_CSI_RE.sub("", s))


def _truncate_output(s: str, *, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Cap a string at ``max_bytes`` UTF-8 bytes with a sentinel suffix.

    The returned string is guaranteed to encode to at most ``max_bytes``
    UTF-8 bytes *including* the sentinel suffix. This is a Phase D
    correctness fix — codex review observed that the previous version
    appended the sentinel *after* the budget, so persisted output
    could exceed ``max_bytes``.
    """
    encoded = s.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return s
    sentinel = f"\n... [TRUNCATED: original {len(encoded)} bytes]"
    sentinel_bytes = sentinel.encode("utf-8")
    # Reserve sentinel bytes from the budget. Guard against a sentinel
    # somehow exceeding the budget (would only happen with absurdly
    # small max_bytes in tests).
    budget = max(0, max_bytes - len(sentinel_bytes))
    truncated = encoded[:budget].decode("utf-8", errors="replace")
    return truncated + sentinel


def _escape_md_fence(s: str) -> str:
    """Escape ``` fence opens so an agent cannot break the rendered code block.

    Inserts a zero-width space (``\\u200b``) between the leading
    whitespace and the backticks, which preserves the visual content
    but prevents Markdown's parser from closing the surrounding fence.
    """
    return _FENCE_RE.sub("\\1\u200b\\2", s)


def _validate_round_id(round_id: str) -> None:
    """Reject ``round_id`` values that would traverse outside ``output_dir``."""
    if not _ROUND_ID_RE.match(round_id or ""):
        raise ValueError(f"invalid round_id {round_id!r}: must match {_ROUND_ID_RE.pattern}")


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` atomically via tmp file + ``os.replace``.

    Crashes mid-write leave the original file (if any) intact instead
    of producing a half-written transcript. The temp file lives in the
    same directory so ``os.replace`` is a same-filesystem rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding=encoding) as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems (notably tmpfs in CI) don't support fsync;
            # the os.replace still gives atomic visibility.
            pass
    os.replace(tmp_path, path)


# --------------------------------------------------------------------- #
# Verified working flag sets (round 30d agent verification)             #
# --------------------------------------------------------------------- #
#
# These are the exact CLI invocations that succeed in non-interactive
# dispatch. Encoded as data so any future round can reuse the harness
# without re-deriving the flags.

DEFAULT_CLAUDE_TIMEOUT: int = 180
"""Default Claude CLI budget.

Live 30f dogfood showed Claude Sonnet needs roughly two minutes for
repo-grounded review prompts, so the old 60s budget was too short for
quality work.
"""

DEFAULT_CODEX_TIMEOUT: int = 420
"""Default Codex CLI budget.

Live 30f dogfood completed a high-quality Codex review in ~273s; this
keeps enough margin for normal GitHub/repo inspection variance without
making hung subprocesses unbounded.
"""

DEFAULT_DROID_TIMEOUT: int = 300
"""Default Droid CLI budget for read/search-only review prompts."""

DEFAULT_MODEL_TIMEOUT: int = 360
"""Default heterogeneous-model budget for non-Codex model lanes."""

CLAUDE_FLAGS: tuple[str, ...] = ("--print",)
"""``claude --print`` works without ``--bare``. ``--bare`` requires
explicit ``ANTHROPIC_API_KEY`` env var; the keychain OAuth flow used
by interactive ``claude`` is only honored without ``--bare``."""

CODEX_FLAGS: tuple[str, ...] = (
    "exec",
    "--skip-git-repo-check",
    "-c",
    "reasoning_effort=minimal",
    "-c",
    "sandbox_mode=read-only",
)
"""``codex exec`` defaults to ``reasoning_effort=xhigh`` which produces
extensive memory/file investigation chains that exceed 90s timeouts.
``minimal`` reasoning + ``read-only`` sandbox keeps the model focused
on the prompt and skips the investigation phase."""

DROID_REVIEW_SYSTEM_PROMPT: str = (
    "Do not run shell commands or modify files. Use read/search tools only "
    "and return a concise text answer."
)
"""Additional Droid guardrail for local-only dialog review turns."""

DROID_FLAGS: tuple[str, ...] = (
    "exec",
    "--auto",
    "high",
    "--disabled-tools",
    "Execute",
    "--append-system-prompt",
    DROID_REVIEW_SYSTEM_PROMPT,
)
"""``droid exec --auto high`` avoids non-interactive permission prompts
while disabling ``Execute`` keeps dialog turns on file/read/search tools only.

Round 30f dogfood showed that repo-review prompts under the previous
``--auto low`` setting could fail with a permission/autonomy error
instead of timing out. The read/search-only guardrail produced usable
output in ~183s while preserving the local-only contract.
"""


# --------------------------------------------------------------------- #
# Dataclasses                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgentSpec:
    """Specification for one agent's turn in a dialog round."""

    name: str
    """Short display name (e.g. ``claude``, ``codex``, ``droid``)."""

    binary: str
    """Path to the CLI executable (or just the name if on PATH)."""

    base_flags: tuple[str, ...]
    """Flags prepended before the prompt (or before ``-`` for stdin)."""

    timeout_seconds: int
    """Hard cap on dispatch time."""

    stdin_mode: str = "argv"
    """Either ``argv`` (prompt as final arg) or ``stdin`` (piped to stdin)."""

    @classmethod
    def claude(cls, timeout_seconds: int = DEFAULT_CLAUDE_TIMEOUT) -> "AgentSpec":
        return cls(
            name="claude",
            binary="claude",
            base_flags=CLAUDE_FLAGS,
            timeout_seconds=timeout_seconds,
            stdin_mode="stdin",
        )

    @classmethod
    def codex(cls, timeout_seconds: int = DEFAULT_CODEX_TIMEOUT) -> "AgentSpec":
        return cls(
            name="codex",
            binary="codex",
            base_flags=CODEX_FLAGS,
            timeout_seconds=timeout_seconds,
            stdin_mode="stdin",
        )

    @classmethod
    def droid(cls, timeout_seconds: int = DEFAULT_DROID_TIMEOUT) -> "AgentSpec":
        return cls(
            name="droid",
            binary="droid",
            base_flags=DROID_FLAGS,
            timeout_seconds=timeout_seconds,
            stdin_mode="stdin",
        )

    # ----------------------------------------------------------------- #
    # Heterogeneous model factories (round 30e)                         #
    # ----------------------------------------------------------------- #
    #
    # The base ``claude()`` / ``codex()`` / ``droid()`` factories above
    # use the CLI's *default* model, which means a 3-CLI panel is in
    # practice "same-family-three-times". The maximalist vision wants
    # **heterogeneous-model consensus**: independent perspectives from
    # different model families catch what one family misses.
    #
    # These factories pin a specific model on a specific CLI surface
    # so a single round can dispatch e.g. claude-opus, claude-sonnet,
    # gpt-5.4, gemini-3.1-pro, kimi-k2.5 in parallel.

    @classmethod
    def with_model(
        cls,
        cli: str,
        model: str,
        *,
        name: str | None = None,
        timeout_seconds: int = DEFAULT_MODEL_TIMEOUT,
    ) -> "AgentSpec":
        """Build an AgentSpec pinned to a specific model on a specific CLI.

        Args:
            cli: One of ``"claude"``, ``"codex"``, ``"droid"``.
            model: Model identifier accepted by the chosen CLI.
            name: Optional display name; defaults to ``f"{cli}:{model}"``.
            timeout_seconds: Hard cap on dispatch.

        ``codex`` does not currently expose a stable per-invocation
        model flag in its public CLI; passing ``cli="codex"`` will
        raise ``ValueError`` to make the limitation explicit.
        """
        cli_norm = (cli or "").strip().lower()
        display = name or f"{cli_norm}:{model}"
        if cli_norm == "claude":
            return cls(
                name=display,
                binary="claude",
                base_flags=(*CLAUDE_FLAGS, "--model", model),
                timeout_seconds=timeout_seconds,
                stdin_mode="stdin",
            )
        if cli_norm == "droid":
            return cls(
                name=display,
                binary="droid",
                base_flags=(*DROID_FLAGS, "-m", model),
                timeout_seconds=timeout_seconds,
                stdin_mode="stdin",
            )
        if cli_norm == "codex":
            raise ValueError(
                "codex CLI does not currently expose a stable per-invocation "
                "model flag; use the codex() factory which inherits the "
                "user's ~/.codex/config.toml model setting."
            )
        raise ValueError(f"unknown cli {cli!r}; expected claude / codex / droid")

    # Concrete heterogeneous factories — known-good model IDs verified
    # in round 30e Phase A.
    @classmethod
    def claude_opus(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """Claude Opus via the ``claude`` CLI."""
        return cls.with_model("claude", "opus", name="claude-opus", timeout_seconds=timeout_seconds)

    @classmethod
    def claude_sonnet(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """Claude Sonnet via the ``claude`` CLI."""
        return cls.with_model(
            "claude", "sonnet", name="claude-sonnet", timeout_seconds=timeout_seconds
        )

    @classmethod
    def droid_gpt5(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """GPT-5.4 via the ``droid`` CLI."""
        return cls.with_model(
            "droid", "gpt-5.4", name="droid-gpt5", timeout_seconds=timeout_seconds
        )

    @classmethod
    def droid_gemini(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """Gemini 3.1 Pro via the ``droid`` CLI."""
        return cls.with_model(
            "droid",
            "gemini-3.1-pro-preview",
            name="droid-gemini",
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def droid_kimi(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """Kimi K2.5 (Chinese frontier) via the ``droid`` CLI."""
        return cls.with_model(
            "droid", "kimi-k2.5", name="droid-kimi", timeout_seconds=timeout_seconds
        )

    @classmethod
    def droid_glm(cls, timeout_seconds: int = DEFAULT_MODEL_TIMEOUT) -> "AgentSpec":
        """GLM 5.1 (Chinese frontier) via the ``droid`` CLI."""
        return cls.with_model("droid", "glm-5.1", name="droid-glm", timeout_seconds=timeout_seconds)

    @classmethod
    def heterogeneous_panel(
        cls,
        *,
        codex_timeout: int = DEFAULT_CODEX_TIMEOUT,
        model_timeout: int = DEFAULT_MODEL_TIMEOUT,
    ) -> "tuple[AgentSpec, ...]":
        """Return the canonical heterogeneous review panel.

        Six independent model surfaces spanning Anthropic, OpenAI,
        Google, and Chinese frontier families. Used by the
        ``--agents-spec heterogeneous`` shorthand in the CLI.
        """
        return (
            cls.claude_opus(timeout_seconds=model_timeout),
            cls.claude_sonnet(timeout_seconds=model_timeout),
            cls.codex(timeout_seconds=codex_timeout),
            cls.droid_gpt5(timeout_seconds=model_timeout),
            cls.droid_gemini(timeout_seconds=model_timeout),
            cls.droid_kimi(timeout_seconds=model_timeout),
        )


@dataclass
class DialogTurn:
    """One agent's response for one prompt."""

    agent: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None

    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.error

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DialogRound:
    """One round of cross-agent dialog: a single prompt for many agents."""

    round_id: str
    prompt: str
    extra_context: str = ""
    """Optional extra context (e.g. source file contents) appended to prompt."""

    metadata: dict[str, str] = field(default_factory=dict)

    def render_prompt(self) -> str:
        if self.extra_context:
            return f"{self.prompt}\n\n{self.extra_context}"
        return self.prompt


# --------------------------------------------------------------------- #
# Dispatch                                                              #
# --------------------------------------------------------------------- #


async def _dispatch_one(
    spec: AgentSpec,
    rendered_prompt: str,
) -> DialogTurn:
    """Dispatch one agent and return a recorded ``DialogTurn``.

    Errors (subprocess launch failure, timeout) are caught and
    encoded into the turn rather than propagated. This guarantees
    that one agent's failure cannot cascade.
    """
    started_at = datetime.now(timezone.utc)

    if spec.stdin_mode == "stdin":
        argv = [spec.binary, *spec.base_flags]
        stdin_data: bytes | None = rendered_prompt.encode("utf-8")
    else:
        argv = [spec.binary, *spec.base_flags, rendered_prompt]
        stdin_data = None

    timed_out = False
    err_msg: str | None = None
    rc = -1
    out = ""
    err = ""

    # Round 30e Phase C fix #1: launch each child in its own process
    # group on POSIX so a timeout can reap any grandchildren the CLI
    # spawned (some agent CLIs fork sub-helpers that leak otherwise).
    #
    # Phase D follow-up: prefer ``start_new_session=True`` over
    # ``preexec_fn=os.setsid`` per Python docs — it avoids the
    # interpreter-thread fork hazard while achieving the same setsid()
    # effect.
    new_session = os.name == "posix"

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=new_session,
        )
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=spec.timeout_seconds,
            )
            rc = proc.returncode if proc.returncode is not None else -1
            out = out_b.decode("utf-8", errors="replace")
            err = err_b.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            timed_out = True
            # Round 30e Phase C fix #2: SIGKILL the whole process group,
            # not just the leader, so child helpers don't leak.
            try:
                if os.name == "posix":
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                await proc.wait()
            except (ProcessLookupError, PermissionError):
                pass
    except FileNotFoundError:
        # Round 30e Phase C fix #3: distinct sentinel rc so the operator
        # can tell "CLI missing on $PATH" apart from "CLI ran and
        # returned -1".
        err_msg = f"binary not found: {spec.binary}"
        rc = RC_BINARY_NOT_FOUND
        logger.warning("multi_agent_dialog: %s", err_msg)
    except Exception as exc:  # noqa: BLE001 — capture any subprocess setup error
        err_msg = f"dispatch failed: {type(exc).__name__}: {exc}"
        rc = RC_DISPATCH_ERROR
        logger.warning("multi_agent_dialog: %s", err_msg)

    finished_at = datetime.now(timezone.utc)
    elapsed = (finished_at - started_at).total_seconds()

    # Round 30e Phase C fix #5 (truncate first — Phase D follow-up):
    # cap output BEFORE ANSI-strip so a multi-MB ANSI-laden line
    # doesn't get fully regex-scanned in memory. This keeps both CPU
    # and persisted size bounded by MAX_OUTPUT_BYTES.
    out = _truncate_output(out)
    err = _truncate_output(err)

    # Round 30e Phase C fix #4: strip ANSI escapes before persisting so
    # a malicious agent can't poison downstream tooling with cursor
    # moves or OSC-8 hyperlinks.
    out = _strip_ansi(out)
    err = _strip_ansi(err)

    return DialogTurn(
        agent=spec.name,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        elapsed_seconds=round(elapsed, 3),
        returncode=rc,
        stdout=out,
        stderr=err,
        timed_out=timed_out,
        error=err_msg,
    )


async def dispatch_round(
    round_: DialogRound,
    agents: Sequence[AgentSpec],
) -> list[DialogTurn]:
    """Dispatch all agents in parallel; return turn records.

    Failures of any single agent do not affect the others.
    """
    if not agents:
        return []
    rendered = round_.render_prompt()
    tasks = [_dispatch_one(spec, rendered) for spec in agents]

    # Round 30e Phase C fix #6: ``return_exceptions=True`` ensures one
    # agent's unexpected raise (e.g. asyncio.CancelledError, OSError
    # from preexec_fn) cannot cascade and abort all peer dispatches.
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[DialogTurn] = []
    for spec, result in zip(agents, raw):
        if isinstance(result, BaseException):
            now = datetime.now(timezone.utc).isoformat()
            out.append(
                DialogTurn(
                    agent=spec.name,
                    started_at=now,
                    finished_at=now,
                    elapsed_seconds=0.0,
                    returncode=RC_DISPATCH_ERROR,
                    stdout="",
                    stderr="",
                    timed_out=False,
                    error=f"unexpected dispatch exception: {type(result).__name__}: {result}",
                )
            )
        else:
            out.append(result)
    return out


# --------------------------------------------------------------------- #
# Persistence                                                           #
# --------------------------------------------------------------------- #


def write_round_jsonl(
    round_: DialogRound,
    turns: Sequence[DialogTurn],
    output_dir: Path,
) -> Path:
    """Persist round + turns as a JSONL stream under ``output_dir``.

    The first line is the round metadata. Subsequent lines are turns.
    """
    # Round 30e Phase C fix #7: validate round_id (path traversal) and
    # write atomically so a crash mid-write doesn't leave a half-baked
    # JSONL behind.
    _validate_round_id(round_.round_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dialog-{round_.round_id}.jsonl"

    lines: list[str] = [
        json.dumps(
            {
                "type": "round",
                "round_id": round_.round_id,
                "prompt": round_.prompt,
                "extra_context_chars": len(round_.extra_context),
                "metadata": round_.metadata,
            },
            sort_keys=True,
        )
    ]
    for turn in turns:
        payload = {"type": "turn", **turn.to_json()}
        lines.append(json.dumps(payload, sort_keys=True))
    _atomic_write_text(out_path, "\n".join(lines) + "\n")
    return out_path


def render_transcript_markdown(
    round_: DialogRound,
    turns: Sequence[DialogTurn],
) -> str:
    """Render round + turns as a single markdown transcript."""
    lines: list[str] = [
        f"# Cross-agent dialog — round `{round_.round_id}`",
        "",
        f"_Prompt size: {len(round_.prompt)} chars; extra context: "
        f"{len(round_.extra_context)} chars._",
        "",
        "## Prompt",
        "",
        "```",
        round_.prompt.strip() or "(empty)",
        "```",
        "",
    ]
    if round_.extra_context.strip():
        lines.extend(
            [
                "## Extra context",
                "",
                "```",
                round_.extra_context.strip()[:2000]
                + ("..." if len(round_.extra_context) > 2000 else ""),
                "```",
                "",
            ]
        )

    success_count = sum(1 for t in turns if t.succeeded())
    lines.extend(
        [
            "## Summary",
            "",
            f"- Agents dispatched: **{len(turns)}**",
            f"- Successful: **{success_count}**",
            f"- Failed: **{len(turns) - success_count}**",
            "",
        ]
    )

    lines.extend(["## Per-agent responses", ""])
    for turn in turns:
        # Round 30e Phase C fix #8a: explicit [TIMED OUT] badge so a
        # zero-stdout timeout doesn't read as a quiet success in scan.
        if turn.timed_out:
            status = "FAILED [TIMED OUT]"
        elif turn.succeeded():
            status = "succeeded"
        else:
            status = "FAILED"
        lines.extend(
            [
                f"### `{turn.agent}` — {status} (rc={turn.returncode}, "
                f"{turn.elapsed_seconds:.1f}s, "
                f"timed_out={turn.timed_out})",
                "",
            ]
        )
        if turn.error:
            lines.extend([f"_Error: {turn.error}_", ""])
        if turn.stdout.strip():
            # Round 30e Phase C fix #8b: escape any nested ``` so an
            # adversarial agent can't break out of our fenced block and
            # poison the rest of the transcript.
            lines.extend(
                [
                    "**stdout:**",
                    "",
                    "```",
                    _escape_md_fence(turn.stdout.rstrip()),
                    "```",
                    "",
                ]
            )
        if turn.stderr.strip() and not turn.succeeded():
            tail = "\n".join(turn.stderr.rstrip().splitlines()[-10:])
            lines.extend(
                [
                    "**stderr (last 10 lines):**",
                    "",
                    "```",
                    _escape_md_fence(tail),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def write_transcript_markdown(
    round_: DialogRound,
    turns: Sequence[DialogTurn],
    output_dir: Path,
) -> Path:
    """Persist a markdown transcript and return its path."""
    _validate_round_id(round_.round_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dialog-{round_.round_id}.md"
    _atomic_write_text(out_path, render_transcript_markdown(round_, turns))
    return out_path


# --------------------------------------------------------------------- #
# Convenience high-level entry point                                    #
# --------------------------------------------------------------------- #


async def run_round_and_persist(
    round_: DialogRound,
    agents: Sequence[AgentSpec],
    output_dir: Path,
) -> tuple[Path, Path, list[DialogTurn]]:
    """Dispatch the round, persist JSONL + markdown, return both paths and turns."""
    turns = await dispatch_round(round_, agents)
    jsonl_path = write_round_jsonl(round_, turns, output_dir)
    md_path = write_transcript_markdown(round_, turns, output_dir)
    return jsonl_path, md_path, turns
