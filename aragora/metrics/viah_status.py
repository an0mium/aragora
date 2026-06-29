"""Deprecated re-export shim for the VIAH status report.

The originals now live in :mod:`aragora.evaluation.viah_status`. This
module re-exports them so the legacy ``aragora.metrics.viah_status``
import path keeps working for one release.
"""

import warnings

from aragora.evaluation.viah_status import generate_viah_status_report

warnings.warn(
    "aragora.metrics.viah_status is deprecated; import from "
    "aragora.evaluation.viah_status instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["generate_viah_status_report"]
