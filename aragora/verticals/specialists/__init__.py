"""
Vertical Specialist Implementations.

Each specialist extends VerticalSpecialistAgent with domain-specific
capabilities, tools, and compliance checking.
"""

from aragora.verticals.specialists.software import SoftwareSpecialist
from aragora.verticals.specialists.legal import LegalSpecialist
from aragora.verticals.specialists.healthcare import HealthcareSpecialist
from aragora.verticals.specialists.accounting import AccountingSpecialist
from aragora.verticals.specialists.research import ResearchSpecialist

__all__ = [
    "SoftwareSpecialist",
    "LegalSpecialist",
    "HealthcareSpecialist",
    "AccountingSpecialist",
    "ResearchSpecialist",
]
