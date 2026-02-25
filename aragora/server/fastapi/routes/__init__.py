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
from . import security
from . import auth
from . import memory
from . import api_explorer
from . import costs
from . import tasks
from . import notifications
from . import inbox

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
    "security",
    "auth",
    "memory",
    "api_explorer",
    "costs",
    "tasks",
    "notifications",
    "inbox",
]
