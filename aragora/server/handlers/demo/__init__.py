"""Adversarial demo handler for live multi-agent debate showcase.

Provides endpoints to run and monitor demonstration debates that highlight
Aragora's adversarial vetting, calibrated trust, and decision receipt generation.
"""

from .adversarial_demo import (
    handle_adversarial_demo,
    handle_demo_status,
    register_routes,
)

__all__ = [
    "handle_adversarial_demo",
    "handle_demo_status",
    "register_routes",
]
