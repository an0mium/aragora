"""Fail-closed monthly spend cap for metered API calls (predictable-cost guard).

This is the anti-runaway-bill primitive: a hard, calendar-month USD ceiling that
**fails closed** — once the configured cap is reached, further metered calls are
refused (an exception) rather than silently billed. It is the safety net that
makes using metered providers (DeepSeek/Kimi/OpenAI/Anthropic API) predictable
even in autonomous loops.

Design:
- **Default OFF.** With no cap configured (``ARAGORA_MONTHLY_BUDGET_USD`` unset or
  <= 0) every call is allowed and nothing is persisted — zero behavior change.
  Opt in by exporting ``ARAGORA_MONTHLY_BUDGET_USD=200`` (or any limit).
- **Fail closed.** When a cap is set, :func:`assert_within_budget` raises
  :class:`BudgetExceededError` once ``spent + estimated > cap`` for the current
  calendar month.
- **Self-contained + cheap.** A tiny JSON counter (``{month, spent_usd}``) under
  the data dir; calendar-month rollover resets it. Thread-safe.
- **Never breaks a call by accident.** :func:`record_spend` swallows its own I/O
  errors (a guard that can't persist must not crash the agent); only the
  deliberate :class:`BudgetExceededError` from :func:`assert_within_budget`
  propagates.

This complements (does not replace) ``BudgetManager``'s richer per-org/overage
accounting — it is a process-local, always-available hard stop for the common
single-operator case.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "BudgetExceededError",
    "monthly_cap_usd",
    "is_enabled",
    "assert_within_budget",
    "record_spend",
    "current_spend_usd",
    "remaining_usd",
    "status",
]

_CAP_ENV = "ARAGORA_MONTHLY_BUDGET_USD"
_STORE_ENV = "ARAGORA_BUDGET_GUARD_STORE"
_lock = threading.Lock()


class BudgetExceededError(RuntimeError):
    """Raised when a metered call would exceed the configured monthly cap."""


def monthly_cap_usd() -> float:
    """The configured monthly cap in USD, or 0.0 if unset/disabled."""
    raw = os.environ.get(_CAP_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        cap = float(raw)
    except ValueError:
        return 0.0
    return cap if cap > 0 else 0.0


def is_enabled() -> bool:
    """True iff a positive monthly cap is configured (otherwise the guard is a no-op)."""
    return monthly_cap_usd() > 0.0


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _store_path() -> Path:
    override = os.environ.get(_STORE_ENV, "").strip()
    if override:
        return Path(override)
    try:
        from aragora.config import get_default_data_dir

        base = Path(get_default_data_dir())
    except Exception:  # noqa: BLE001 - data dir resolution must never crash the guard
        base = Path.home() / ".aragora"
    return base / "budget_guard.json"


def _read_state() -> dict[str, object]:
    path = _store_path()
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"month": _current_month(), "spent_usd": 0.0}
    month = data.get("month")
    spent = data.get("spent_usd", 0.0)
    if month != _current_month():  # calendar-month rollover resets the counter
        return {"month": _current_month(), "spent_usd": 0.0}
    try:
        spent_val = float(spent)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        spent_val = 0.0
    return {"month": _current_month(), "spent_usd": spent_val}


def _write_state(spent_usd: float) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump({"month": _current_month(), "spent_usd": round(spent_usd, 6)}, fh)
        os.replace(tmp, path)  # atomic; readers never see a partial file
    except OSError:
        # A guard that cannot persist must not crash the caller; the in-flight
        # cap check still used the best-known total.
        pass


def current_spend_usd() -> float:
    """USD spent so far in the current calendar month (0.0 if guard disabled)."""
    if not is_enabled():
        return 0.0
    with _lock:
        return float(_read_state()["spent_usd"])  # type: ignore[arg-type]


def remaining_usd() -> float:
    """USD remaining under the cap this month (``inf`` if guard disabled)."""
    cap = monthly_cap_usd()
    if cap <= 0:
        return float("inf")
    return max(0.0, cap - current_spend_usd())


def assert_within_budget(estimated_usd: float = 0.0, *, label: str | None = None) -> None:
    """Raise :class:`BudgetExceededError` if this spend would exceed the monthly cap.

    No-op when the guard is disabled (no cap configured). ``estimated_usd`` is an
    optional pre-call estimate of the next call's cost; pass 0.0 to gate purely on
    already-recorded spend.
    """
    cap = monthly_cap_usd()
    if cap <= 0:
        return
    with _lock:
        spent = float(_read_state()["spent_usd"])  # type: ignore[arg-type]
    if spent + max(0.0, estimated_usd) > cap:
        where = f" ({label})" if label else ""
        raise BudgetExceededError(
            f"Monthly budget cap reached{where}: spent ${spent:.2f} + est "
            f"${max(0.0, estimated_usd):.2f} > cap ${cap:.2f} "
            f"({_CAP_ENV}). Refusing the call (fail-closed). Raise the cap or wait "
            f"for the next calendar month."
        )


def record_spend(amount_usd: float) -> None:
    """Add ``amount_usd`` to the current month's running total (no-op if disabled).

    Best-effort: never raises (a metering hiccup must not break an agent call).
    """
    if not is_enabled() or amount_usd <= 0:
        return
    try:
        with _lock:
            spent = float(_read_state()["spent_usd"])  # type: ignore[arg-type]
            _write_state(spent + float(amount_usd))
    except Exception:  # noqa: BLE001 - metering must never crash the caller
        pass


def status() -> dict[str, float | bool | str]:
    """Snapshot of the guard for operator/CLI display."""
    cap = monthly_cap_usd()
    spent = current_spend_usd()
    return {
        "enabled": cap > 0,
        "month": _current_month(),
        "cap_usd": cap,
        "spent_usd": spent,
        "remaining_usd": remaining_usd(),
    }
