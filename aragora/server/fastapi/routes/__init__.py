"""FastAPI Route modules."""

from . import health
from . import debates
from . import decisions
from . import testfixer
from . import receipts
from . import gauntlet
from . import agents
from . import consensus
from . import pipeline
from . import knowledge
from . import workflows
from . import compliance
from . import auth
from . import memory
from . import api_explorer

__all__ = [
    "health",
    "debates",
    "decisions",
    "testfixer",
    "receipts",
    "gauntlet",
    "agents",
    "consensus",
    "pipeline",
    "knowledge",
    "workflows",
    "compliance",
    "auth",
    "memory",
    "api_explorer",
]
