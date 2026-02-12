"""FastAPI Route modules."""

from . import health
from . import debates
from . import decisions
from . import testfixer
from . import receipts
from . import gauntlet
from . import agents

__all__ = ["health", "debates", "decisions", "testfixer", "receipts", "gauntlet", "agents"]
