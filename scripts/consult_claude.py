#!/usr/bin/env python3
"""Bounded advisory consult of a specific Claude model (default: Claude Fable 5).

Gives any agent (Codex conductor, Droid, Claude Code, humans) a reliable way to
ask a named Claude model for read-only advice with a hard timeout. The known
failure mode this fixes: ad-hoc ``timeout 120 claude -p "..."`` calls hang or
time out with no output and no diagnostics.

Backends, in order:

1. ``claude`` CLI (subscription auth) — routed through the authenticated
   ``claude_profile.sh`` pool when available, with ``--model`` forwarded and a
   hard subprocess timeout.
2. Anthropic Messages API — used only if the CLI is missing, fails, or times
   out. The key comes from ``ANTHROPIC_API_KEY`` or the aragora secrets
   manager; if neither is present the fallback is skipped silently.

Output is the raw model text on stdout, or a JSON envelope with ``--json``.
Exit codes: 0 ok, 2 all backends timed out, 3 no prompt, 4 all backends failed.

Examples::

    python scripts/consult_claude.py "Which PR should I settle next?"
    python scripts/consult_claude.py --prompt-file /tmp/question.md --json
    echo "$QUESTION" | python scripts/consult_claude.py --timeout 900
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "claude-fable-5"
FALLBACK_MODEL = "claude-opus-4-8"
DEFAULT_TIMEOUT_SECONDS = 600
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
API_MAX_TOKENS = 8192

EXIT_OK = 0
EXIT_TIMEOUT = 2
EXIT_NO_PROMPT = 3
EXIT_ALL_FAILED = 4


def _build_cli_command(model: str) -> tuple[list[str], bool]:
    """Return the claude CLI command, profile-pool wrapped when possible."""
    base = ["claude", "--print", "--model", model, "-p", "-"]
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
    command, used_profile = _build_cli_command(model)
    backend = "cli-profile" if used_profile else "cli"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "backend": backend,
            "timed_out": True,
            "elapsed_s": round(time.monotonic() - started, 1),
            "error": f"claude CLI exceeded {timeout:.0f}s timeout",
        }
    except OSError as exc:
        return {"ok": False, "backend": backend, "error": f"claude CLI launch failed: {exc}"}
    elapsed = round(time.monotonic() - started, 1)
    text = _strip_preamble(proc.stdout) if used_profile else proc.stdout.strip()
    if proc.returncode != 0 or not text:
        stderr_tail = (proc.stderr or "").strip()[-500:]
        return {
            "ok": False,
            "backend": backend,
            "elapsed_s": elapsed,
            "error": f"claude CLI rc={proc.returncode}, empty={not text}: {stderr_tail}",
        }
    return {"ok": True, "backend": backend, "elapsed_s": elapsed, "text": text}


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
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return {"ok": False, "backend": "api", "error": f"API HTTP {exc.code}: {detail}"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        timed_out = "timed out" in str(exc).lower() or isinstance(exc, TimeoutError)
        return {
            "ok": False,
            "backend": "api",
            "timed_out": timed_out,
            "error": f"API request failed: {exc}",
        }
    elapsed = round(time.monotonic() - started, 1)
    text = "".join(
        block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
    ).strip()
    if not text:
        return {
            "ok": False,
            "backend": "api",
            "elapsed_s": elapsed,
            "error": "API returned no text",
        }
    return {"ok": True, "backend": "api", "elapsed_s": elapsed, "text": text}


def consult(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    fallback_model: str | None = FALLBACK_MODEL,
    system: str | None = None,
    api_fallback: bool = True,
) -> dict:
    """Run the consult across backends and return the first success.

    The full timeout budget is granted to each attempt independently so a CLI
    hang cannot starve the API fallback of time.
    """
    if system:
        prompt = f"{system}\n\n---\n\n{prompt}"
    attempts: list[dict] = []
    result = _run_cli(prompt, model, timeout)
    attempts.append({"model": model, **result})
    if result.get("ok"):
        return {**result, "model": model, "attempts": attempts}
    if fallback_model and fallback_model != model and not result.get("timed_out"):
        result = _run_cli(prompt, fallback_model, timeout)
        attempts.append({"model": fallback_model, **result})
        if result.get("ok"):
            return {**result, "model": fallback_model, "attempts": attempts}
    if api_fallback:
        result = _run_api(prompt, model, timeout, system=None)
        attempts.append({"model": model, **result})
        if result.get("ok"):
            return {**result, "model": model, "attempts": attempts}
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
    parser.add_argument("--system", help="Optional system-style preamble prepended to the prompt")
    parser.add_argument(
        "--no-api-fallback",
        action="store_true",
        help="Do not fall back to the Anthropic API when the CLI fails",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result envelope")
    args = parser.parse_args(argv)

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompt = handle.read()
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
