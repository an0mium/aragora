"""
Advanced debate features for nomic loop.

Provides:
- graph_debates: Graph-based debate topology
- forking: Counterfactual branching and parallel exploration
"""

from .graph_debates import GraphDebateRunner
from .forking import ForkingRunner, ForkOutcome

__all__ = [
    "GraphDebateRunner",
    "ForkingRunner",
    "ForkOutcome",
]
