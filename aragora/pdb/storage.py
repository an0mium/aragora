"""Back-compat shim for :mod:`aragora.brief_engine.storage`.

The Mode 3 PDB storage layer moved to
:mod:`aragora.brief_engine.storage` in the Phase 1 brief-engine
extraction. On-disk layout, filename conventions, and function
signatures are unchanged; this module re-exports every name so callers
importing from ``aragora.pdb.storage`` continue to work.

Do not add new storage primitives here — put them in
:mod:`aragora.brief_engine.storage` instead.
"""

from __future__ import annotations

# ``os`` is re-exported so legacy tests that patch ``storage.os.replace``
# (``patch.object(storage.os, "replace", ...)``) continue to resolve a
# real module attribute. All I/O still runs through the generic layer.
import os  # noqa: F401

from aragora.brief_engine.storage import (
    FAILED_SUBDIR,
    INDEX_FILENAME,
    INVALIDATED_SUBDIR,
    QUEUED_SUBDIR,
    RUNNING_SUBDIR,
    append_index_event,
    briefs_root,
    cancel_generation,
    find_ready_briefs_for_pr,
    get_state,
    invalidate_if_head_changed,
    load_latest_ready_brief,
    load_ready_brief,
    mark_failed,
    mark_ready,
    mark_running,
    queue_generation,
    write_running_phase,
)

__all__ = [
    "briefs_root",
    "QUEUED_SUBDIR",
    "RUNNING_SUBDIR",
    "FAILED_SUBDIR",
    "INVALIDATED_SUBDIR",
    "INDEX_FILENAME",
    "get_state",
    "load_ready_brief",
    "load_latest_ready_brief",
    "find_ready_briefs_for_pr",
    "queue_generation",
    "mark_running",
    "write_running_phase",
    "mark_ready",
    "mark_failed",
    "invalidate_if_head_changed",
    "cancel_generation",
    "append_index_event",
]
