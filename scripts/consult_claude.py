#!/usr/bin/env python3
"""Bounded advisory consult of a specific Claude model (default: Claude Fable 5).

Gives any agent (Codex conductor, Droid, Claude Code, humans) a reliable way to
ask a named Claude model for read-only advice with hard per-attempt and overall
timeouts. The known failure mode this fixes: ad-hoc ``timeout 120 claude -p
"..."`` calls hang or time out with no output and no diagnostics.

Backends, in order:

1. ``claude`` CLI (subscription auth) — routed through the authenticated
   ``claude_profile.sh`` pool when available, with ``--model`` forwarded and a
   hard subprocess timeout.
2. Anthropic Messages API — used only if the CLI is missing, fails, or times
   out. The key comes from ``ANTHROPIC_API_KEY`` or the aragora secrets
   manager; if neither is present the fallback is skipped silently.

Output is the raw model text on stdout, or a JSON envelope with ``--json``.
Exit codes: 0 ok, 2 all backends timed out, 3 no prompt, 4 all backends
failed, 64 usage/config error.

Examples::

    python scripts/consult_claude.py "Which PR should I settle next?"
    python scripts/consult_claude.py --prompt-file /tmp/question.md --json
    echo "$QUESTION" | python scripts/consult_claude.py --timeout 300 --overall-timeout 900
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL = "claude-fable-5"
FALLBACK_MODEL = "claude-opus-4-8"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_OVERALL_TIMEOUT_SECONDS = 600
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
API_MAX_TOKENS = 8192

EXIT_OK = 0
EXIT_TIMEOUT = 2
EXIT_NO_PROMPT = 3
EXIT_ALL_FAILED = 4
EXIT_USAGE = 64


def _safe_cli_error(*, returncode: int | None = None, empty: bool | None = None) -> str:
    """Public CLI failure string safe for JSON logs and durable artifacts."""

    parts = ["claude CLI failed"]
    if returncode is not None:
        parts.append(f"rc={returncode}")
    if empty is not None:
        parts.append(f"empty={empty}")
    return ", ".join(parts)


def _safe_api_error(message: str) -> str:
    """Public API failure string safe for JSON logs and durable artifacts."""

    return f"API {message}"


@contextmanager
def _claude_empty_mcp_config_file():
    """Write an empty Claude MCP config to avoid wedged local server handshakes."""

    fd, path_text = tempfile.mkstemp(prefix="aragora-consult-claude-mcp-", suffix=".json")
    path = Path(path_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"mcpServers": {}}, handle)
            handle.write("\n")
        yield path
    finally:
        path.unlink(missing_ok=True)


def _build_cli_command(model: str, mcp_config_path: Path) -> tuple[list[str], bool]:
    """Return the claude CLI command, profile-pool wrapped when possible."""
    base = [
        "claude",
        "--print",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config_path),
        "--model",
        model,
        "-p",
        "-",
    ]
    try:
        from aragora.agents.claude_profile_pool import build_claude_command

        return build_claude_command(base)
    except Exception:
        return base, False


def _strip_preamble(text: str) -> str:
    try:
        from aragora.agents.claude_profile_pool import strip_profile_preamble

        return strip_profile_preamble(text)
    except Exception:
        return text.strip()


def _run_cli(prompt: str, model: str, timeout: float) -> dict:
    """One bounded claude CLI attempt. Never raises; returns a result dict."""
    if shutil.which("claude") is None:
        return {"ok": False, "backend": "cli", "error": "claude CLI not on PATH"}
    started = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    try:
        with _claude_empty_mcp_config_file() as mcp_config_path:
            command, used_profile = _build_cli_command(model, mcp_config_path)
            backend = "cli"
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, _stderr = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc is not None:
            _kill_process_group(proc)
        return {
            "ok": False,
            "backend": locals().get("backend", "cli"),
            "timed_out": True,
            "elapsed_s": round(time.monotonic() - started, 1),
            "error": f"claude CLI exceeded {timeout:.0f}s timeout",
        }
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "ok": False,
            "backend": locals().get("backend", "cli"),
            "error": f"claude CLI launch failed: {type(exc).__name__}",
        }
    elapsed = round(time.monotonic() - started, 1)
    text = _strip_preamble(stdout) if used_profile else stdout.strip()
    if proc.returncode != 0 or not text:
        return {
            "ok": False,
            "backend": backend,
            "elapsed_s": elapsed,
            "error": _safe_cli_error(returncode=proc.returncode, empty=not text),
        }
    return {"ok": True, "backend": backend, "elapsed_s": elapsed, "text": text}


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Best-effort cleanup for spawned Claude and any nested child process."""

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _resolve_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        from aragora.config.secrets import get_secret

        return get_secret("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _run_api(prompt: str, model: str, timeout: float, system: str | None) -> dict:
    """One bounded Anthropic Messages API attempt. Never raises."""
    key = _resolve_api_key()
    if not key:
        return {"ok": False, "backend": "api", "error": "no ANTHROPIC_API_KEY available"}
    payload: dict = {
        "model": model,
        "max_tokens": API_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
            if not isinstance(body, dict):
                raise ValueError("API response JSON is not an object")
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except OSError:
            pass
        return {
            "ok": False,
            "backend": "api",
            "error": _safe_api_error(f"HTTP {exc.code}: response body redacted"),
        }
    except (json.JSONDecodeError, UnicodeError, ValueError):
        return {
            "ok": False,
            "backend": "api",
            "error": _safe_api_error("response parse failed: response body redacted"),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        timed_out = "timed out" in str(exc).lower() or isinstance(exc, TimeoutError)
        return {
            "ok": False,
            "backend": "api",
            "timed_out": timed_out,
            "error": _safe_api_error(f"request failed: {type(exc).__name__}"),
        }
    elapsed = round(time.monotonic() - started, 1)
    content = body.get("content", [])
    if not isinstance(content, list):
        content = []
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if not text:
        return {
            "ok": False,
            "backend": "api",
            "elapsed_s": elapsed,
            "error": "API returned no text",
        }
    return {"ok": True, "backend": "api", "elapsed_s": elapsed, "text": text}


def _remaining_timeout(started: float, overall_timeout: float, per_attempt_timeout: float) -> float:
    remaining = overall_timeout - (time.monotonic() - started)
    return max(0.0, min(per_attempt_timeout, remaining))


def _append_budget_exhausted(attempts: list[dict], *, model: str, backend: str) -> None:
    attempts.append(
        {
            "model": model,
            "ok": False,
            "backend": backend,
            "timed_out": True,
            "error": "overall consult timeout exhausted before attempt",
        }
    )


def consult(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    overall_timeout: float = DEFAULT_OVERALL_TIMEOUT_SECONDS,
    fallback_model: str | None = FALLBACK_MODEL,
    system: str | None = None,
    api_fallback: bool = True,
) -> dict:
    """Run the consult across backends and return the first success.

    ``timeout`` is the per-attempt ceiling. ``overall_timeout`` is the total
    consult budget shared by every CLI/API attempt.
    """
    if system:
        prompt = f"{system}\n\n---\n\n{prompt}"
    attempts: list[dict] = []
    started = time.monotonic()
    attempt_timeout = _remaining_timeout(started, overall_timeout, timeout)
    if attempt_timeout <= 0:
        _append_budget_exhausted(attempts, model=model, backend="cli")
    else:
        result = _run_cli(prompt, model, attempt_timeout)
        attempts.append({"model": model, **result})
        if result.get("ok"):
            return {**result, "model": model, "attempts": attempts}
    if fallback_model and fallback_model != model:
        attempt_timeout = _remaining_timeout(started, overall_timeout, timeout)
        if attempt_timeout <= 0:
            _append_budget_exhausted(attempts, model=fallback_model, backend="cli")
        else:
            result = _run_cli(prompt, fallback_model, attempt_timeout)
            attempts.append({"model": fallback_model, **result})
            if result.get("ok"):
                return {**result, "model": fallback_model, "attempts": attempts}
    if api_fallback:
        for api_model in (model, fallback_model):
            if not api_model:
                continue
            if any(
                attempt.get("backend") == "api" and attempt.get("model") == api_model
                for attempt in attempts
            ):
                continue
            attempt_timeout = _remaining_timeout(started, overall_timeout, timeout)
            if attempt_timeout <= 0:
                _append_budget_exhausted(attempts, model=api_model, backend="api")
                continue
            result = _run_api(prompt, api_model, attempt_timeout, system=None)
            attempts.append({"model": api_model, **result})
            if result.get("ok"):
                return {**result, "model": api_model, "attempts": attempts}
    timed_out = all(a.get("timed_out") for a in attempts) and bool(attempts)
    return {
        "ok": False,
        "model": model,
        "timed_out": timed_out,
        "attempts": attempts,
        "error": "; ".join(str(a.get("error")) for a in attempts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="?", help="Question text (or use --prompt-file / stdin)")
    parser.add_argument("--prompt-file", help="Read the question from a file")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model id (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        help=f"Second CLI attempt if the primary model errors (default {FALLBACK_MODEL}; '' disables)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Hard per-attempt timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--overall-timeout",
        type=float,
        default=DEFAULT_OVERALL_TIMEOUT_SECONDS,
        help=f"Hard total consult timeout in seconds (default {DEFAULT_OVERALL_TIMEOUT_SECONDS})",
    )
    parser.add_argument("--system", help="Optional system-style preamble prepended to the prompt")
    parser.add_argument(
        "--no-api-fallback",
        action="store_true",
        help="Do not fall back to the Anthropic API when the CLI fails",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result envelope")
    args = parser.parse_args(argv)

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        print("error: --timeout must be a positive finite number", file=sys.stderr)
        return EXIT_USAGE
    if not math.isfinite(args.overall_timeout) or args.overall_timeout <= 0:
        print("error: --overall-timeout must be a positive finite number", file=sys.stderr)
        return EXIT_USAGE

    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as handle:
                prompt = handle.read()
        except OSError as exc:
            print(f"error: cannot read --prompt-file: {exc}", file=sys.stderr)
            return EXIT_NO_PROMPT
    elif args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        parser.print_usage(sys.stderr)
        print("error: no prompt given (arg, --prompt-file, or stdin)", file=sys.stderr)
        return EXIT_NO_PROMPT
    prompt = prompt.strip()
    if not prompt:
        print("error: prompt is empty", file=sys.stderr)
        return EXIT_NO_PROMPT

    result = consult(
        prompt,
        model=args.model,
        timeout=args.timeout,
        overall_timeout=args.overall_timeout,
        fallback_model=args.fallback_model or None,
        system=args.system,
        api_fallback=not args.no_api_fallback,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("ok"):
        print(result["text"])
    else:
        print(f"consult failed: {result.get('error')}", file=sys.stderr)
    if result.get("ok"):
        return EXIT_OK
    return EXIT_TIMEOUT if result.get("timed_out") else EXIT_ALL_FAILED


if __name__ == "__main__":
    sys.exit(main())
