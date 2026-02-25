# aragora/topics.py
from enum import Enum


class EvidenceLevel(Enum):
    """Defines the rules of evidence for a debate."""

    NO_EVIDENCE = "disallow-external"
    CITED_EVIDENCE = "allow-cited"
    REQUIRED_EVIDENCE = "require-cited"
