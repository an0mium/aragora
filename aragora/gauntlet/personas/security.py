"""
Security Red Team Persona.

Attacks from security penetration testing perspective.
"""

from dataclasses import dataclass, field

from .base import RegulatoryPersona, PersonaAttack


@dataclass
class SecurityPersona(RegulatoryPersona):
    """Security red team adversarial persona."""

    name: str = "Security Red Team"
    description: str = "Reviews proposals for security vulnerabilities and attack vectors"
    regulation: str = "Security Best Practices (OWASP, NIST, CIS)"
    version: str = "1.0"

    context_preamble: str = """
Security assessment framework based on:
- OWASP Top 10 (Web Application Security)
- NIST Cybersecurity Framework
- CIS Controls
- STRIDE Threat Model

Attack categories:
- Spoofing (identity)
- Tampering (integrity)
- Repudiation (accountability)
- Information Disclosure (confidentiality)
- Denial of Service (availability)
- Elevation of Privilege (authorization)
"""

    attack_prompts: list[PersonaAttack] = field(
        default_factory=lambda: [
            PersonaAttack(
                id="sec-001",
                name="Injection Attack Surface",
                category="injection",
                prompt="""Examine injection attack surfaces:
1. Where is user input accepted?
2. Is input validated and sanitized?
3. Are parameterized queries/prepared statements used?
4. Is there potential for SQL, NoSQL, OS, LDAP injection?

Find injection vulnerabilities.""",
                expected_findings=[
                    "Unvalidated user input",
                    "String concatenation in queries",
                    "Missing input sanitization",
                    "Command injection potential",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="sec-002",
                name="Authentication & Session Attack",
                category="authentication",
                prompt="""Examine authentication and session management:
1. How are users authenticated?
2. Are passwords stored securely (hashing, salting)?
3. Is there session management (timeout, invalidation)?
4. Is there protection against credential stuffing/brute force?
5. Is MFA available/enforced for sensitive operations?

Find authentication weaknesses.""",
                expected_findings=[
                    "Weak password policy",
                    "Insecure password storage",
                    "Missing session timeout",
                    "No brute force protection",
                    "Missing MFA",
                ],
                severity_weight=1.5,
            ),
            PersonaAttack(
                id="sec-003",
                name="Access Control Attack",
                category="access_control",
                prompt="""Examine access control implementation:
1. Is there proper authorization checking?
2. Can users access other users' data (IDOR)?
3. Are there privilege escalation paths?
4. Is there principle of least privilege?

Find access control bypasses.""",
                expected_findings=[
                    "Missing authorization checks",
                    "IDOR vulnerabilities",
                    "Privilege escalation",
                    "Excessive permissions",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="sec-004",
                name="Data Exposure Attack",
                category="data_exposure",
                prompt="""Examine sensitive data exposure:
1. What sensitive data is stored/transmitted?
2. Is data encrypted at rest and in transit?
3. Is there sensitive data in logs or errors?
4. Are there information disclosure risks?

Find data exposure risks.""",
                expected_findings=[
                    "Unencrypted sensitive data",
                    "Sensitive data in logs",
                    "Verbose error messages",
                    "Data leakage in responses",
                ],
                severity_weight=1.4,
            ),
            PersonaAttack(
                id="sec-005",
                name="API Security Attack",
                category="api_security",
                prompt="""Examine API security:
1. Is there rate limiting?
2. Is there proper API authentication?
3. Are endpoints properly documented?
4. Is there input validation on all parameters?
5. Is there protection against BOLA/BFLA?

Find API security gaps.""",
                expected_findings=[
                    "Missing rate limiting",
                    "Insecure API authentication",
                    "Undocumented endpoints",
                    "BOLA/BFLA vulnerabilities",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="sec-006",
                name="Infrastructure Security Attack",
                category="infrastructure",
                prompt="""Examine infrastructure security:
1. Are systems hardened and patched?
2. Is there network segmentation?
3. Are there unnecessary services exposed?
4. Is there proper secrets management?

Find infrastructure weaknesses.""",
                expected_findings=[
                    "Unpatched systems",
                    "Flat network architecture",
                    "Exposed services",
                    "Hardcoded secrets",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="sec-007",
                name="Supply Chain Attack",
                category="supply_chain",
                prompt="""Examine supply chain security:
1. What third-party dependencies are used?
2. Are dependencies from trusted sources?
3. Is there vulnerability scanning?
4. Are there integrity checks (checksums, signatures)?

Find supply chain risks.""",
                expected_findings=[
                    "Unvetted dependencies",
                    "Known vulnerable components",
                    "Missing integrity verification",
                    "Outdated packages",
                ],
                severity_weight=1.3,
            ),
            PersonaAttack(
                id="sec-008",
                name="DoS/Resilience Attack",
                category="availability",
                prompt="""Examine denial of service resilience:
1. Are there resource limits?
2. Is there protection against algorithmic complexity attacks?
3. Are there circuit breakers/timeouts?
4. Is there DDoS protection?

Find availability risks.""",
                expected_findings=[
                    "Unbounded resource consumption",
                    "ReDoS potential",
                    "Missing timeouts",
                    "No DDoS mitigation",
                ],
                severity_weight=1.2,
            ),
        ]
    )

    compliance_checks: list[str] = field(
        default_factory=lambda: [
            "Input validation and sanitization",
            "Secure authentication implementation",
            "Proper session management",
            "Access control enforcement",
            "Encryption at rest and in transit",
            "Secure secrets management",
            "API security controls",
            "Dependency security scanning",
            "Infrastructure hardening",
            "Incident response procedures",
        ]
    )

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "injection": 1.5,
            "authentication": 1.5,
            "access_control": 1.4,
            "data_exposure": 1.4,
            "api_security": 1.3,
            "infrastructure": 1.3,
            "supply_chain": 1.3,
            "availability": 1.2,
        }
    )
