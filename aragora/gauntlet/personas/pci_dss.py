"""
PCI-DSS Compliance Persona.

Attacks from PCI-DSS (Payment Card Industry Data Security Standard) perspective.
Based on PCI-DSS v4.0.
"""

from dataclasses import dataclass, field

from .base import RegulatoryPersona, PersonaAttack


@dataclass
class PCIDSSPersona(RegulatoryPersona):
    """PCI-DSS compliance adversarial persona."""

    name: str = "PCI-DSS QSA Auditor"
    description: str = (
        "Reviews proposals for PCI-DSS compliance violations and cardholder data risks"
    )
    regulation: str = "PCI-DSS v4.0"
    version: str = "1.0"

    context_preamble: str = """
PCI-DSS v4.0 Requirements:

BUILD AND MAINTAIN A SECURE NETWORK
- Req 1: Install and maintain network security controls
- Req 2: Apply secure configurations to all system components

PROTECT ACCOUNT DATA
- Req 3: Protect stored account data
- Req 4: Protect cardholder data with strong cryptography during transmission

MAINTAIN A VULNERABILITY MANAGEMENT PROGRAM
- Req 5: Protect all systems from malware
- Req 6: Develop and maintain secure systems and software

IMPLEMENT STRONG ACCESS CONTROL MEASURES
- Req 7: Restrict access to system components and cardholder data
- Req 8: Identify users and authenticate access
- Req 9: Restrict physical access to cardholder data

REGULARLY MONITOR AND TEST NETWORKS
- Req 10: Log and monitor all access to system components
- Req 11: Test security of systems and networks regularly

MAINTAIN AN INFORMATION SECURITY POLICY
- Req 12: Support information security with policies and programs
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="pci-001",
                name="Cardholder Data Storage Attack",
                category="data_storage",
                prompt="""Examine cardholder data storage (Req 3):
1. Is PAN (Primary Account Number) stored? If so, is it encrypted or truncated?
2. Is sensitive authentication data (CVV, PIN) stored after authorization?
3. Are there documented data retention and disposal procedures?
4. Is there a cardholder data inventory?
5. Is encryption key management documented?

Find cardholder data storage violations.""",
                expected_findings=[
                    "Unencrypted PAN storage",
                    "CVV/CVV2 stored after authorization",
                    "PIN data stored after authorization",
                    "No data retention limits",
                    "Missing data inventory",
                    "Inadequate key management",
                ],
                severity_weight=1.8,
            ),
            PersonaAttack(
                id="pci-002",
                name="Network Segmentation Attack",
                category="network_security",
                prompt="""Examine network security and segmentation (Req 1):
1. Is the cardholder data environment (CDE) segmented?
2. Are firewalls/security controls between CDE and other networks?
3. Is there documentation of all connections into the CDE?
4. Are inbound and outbound traffic rules documented?
5. Is wireless network security in place?

Find network segmentation gaps.""",
                expected_findings=[
                    "No CDE segmentation",
                    "Missing firewall between CDE and internet",
                    "Undocumented network connections",
                    "Overly permissive firewall rules",
                    "Insecure wireless access to CDE",
                ],
                severity_weight=1.6,
            ),
            PersonaAttack(
                id="pci-003",
                name="Encryption in Transit Attack",
                category="encryption_transit",
                prompt="""Examine encryption during transmission (Req 4):
1. Is cardholder data encrypted over public networks?
2. Is TLS 1.2 or higher used?
3. Are strong cipher suites configured?
4. Is certificate management documented?
5. Are there any insecure transmission paths?

Find encryption in transit violations.""",
                expected_findings=[
                    "Unencrypted cardholder data transmission",
                    "TLS versions below 1.2",
                    "Weak cipher suites",
                    "Expired or invalid certificates",
                    "Insecure API endpoints",
                ],
                severity_weight=1.6,
            ),
            PersonaAttack(
                id="pci-004",
                name="Access Control Attack",
                category="access_control",
                prompt="""Examine access control measures (Req 7, 8):
1. Is access to cardholder data limited to need-to-know?
2. Are unique user IDs assigned to each person?
3. Is MFA required for remote access and admin access?
4. Are passwords meeting complexity requirements?
5. Is there an access control policy?

Find access control violations.""",
                expected_findings=[
                    "Excessive access to cardholder data",
                    "Shared or generic accounts",
                    "Missing MFA for remote access",
                    "Weak password policies",
                    "No access control documentation",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="pci-005",
                name="Vulnerability Management Attack",
                category="vulnerability_mgmt",
                prompt="""Examine vulnerability management (Req 5, 6, 11):
1. Is anti-malware deployed on all systems?
2. Are systems patched within required timeframes?
3. Are regular vulnerability scans performed?
4. Are penetration tests performed annually?
5. Is there a secure development lifecycle?

Find vulnerability management gaps.""",
                expected_findings=[
                    "Missing anti-malware on systems",
                    "Unpatched critical vulnerabilities",
                    "No regular vulnerability scanning",
                    "No penetration testing",
                    "Insecure development practices",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="pci-006",
                name="Logging and Monitoring Attack",
                category="logging_monitoring",
                prompt="""Examine logging and monitoring (Req 10):
1. Are all access attempts to cardholder data logged?
2. Are logs synchronized and time-stamped?
3. Are logs reviewed daily?
4. Are logs retained for at least 12 months?
5. Is there file integrity monitoring on critical systems?

Find logging violations.""",
                expected_findings=[
                    "Incomplete access logging",
                    "Missing time synchronization",
                    "No daily log review",
                    "Insufficient log retention",
                    "No file integrity monitoring",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="pci-007",
                name="Security Testing Attack",
                category="security_testing",
                prompt="""Examine security testing practices (Req 11):
1. Are ASV scans performed quarterly?
2. Is internal scanning performed after significant changes?
3. Are penetration tests performed at least annually?
4. Is there segmentation testing?
5. Are identified vulnerabilities remediated?

Find security testing gaps.""",
                expected_findings=[
                    "Missing quarterly ASV scans",
                    "No internal vulnerability scanning",
                    "No annual penetration test",
                    "Segmentation not validated",
                    "Unremediated vulnerabilities",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="pci-008",
                name="Third-Party Service Provider Attack",
                category="third_party",
                prompt="""Examine third-party service provider management (Req 12.8):
1. Is there a list of all TPSPs handling cardholder data?
2. Are TPSP PCI compliance certificates obtained?
3. Are TPSPs contractually required to maintain PCI compliance?
4. Is there ongoing monitoring of TPSP compliance?
5. Are TPSP responsibilities clearly defined?

Find third-party management gaps.""",
                expected_findings=[
                    "Undocumented TPSPs",
                    "Missing TPSP AOC/ROC",
                    "No contractual PCI requirements",
                    "No ongoing TPSP monitoring",
                    "Undefined TPSP responsibilities",
                ],
                severity_weight=1.3,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "PAN encrypted or truncated if stored",
            "No SAD (CVV/PIN) stored post-authorization",
            "Cardholder data environment segmented",
            "Firewalls protecting CDE",
            "TLS 1.2+ for cardholder data transmission",
            "Strong cipher suites configured",
            "Access limited to need-to-know",
            "MFA for remote and admin access",
            "Unique user IDs for all users",
            "Anti-malware on all applicable systems",
            "Critical patches applied within 30 days",
            "Quarterly ASV scans",
            "Annual penetration testing",
            "All cardholder data access logged",
            "12-month log retention",
            "Daily log review",
            "Third-party service providers validated",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "data_storage": 1.8,
            "network_security": 1.6,
            "encryption_transit": 1.6,
            "access_control": 1.5,
            "vulnerability_mgmt": 1.4,
            "logging_monitoring": 1.4,
            "security_testing": 1.3,
            "third_party": 1.3,
        }
    )
