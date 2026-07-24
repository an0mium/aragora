"""
Server-resident job queue workers.

Workers here process durable queue jobs but are homed in ``aragora.server``
(interface layer) rather than ``aragora.queue`` (infrastructure layer)
because they import interface-layer packages
(docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §6.2, §10).
"""

from __future__ import annotations
