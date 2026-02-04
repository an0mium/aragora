"""
SAST Scanner rules - constant dictionaries and pattern definitions.

Contains:
- AVAILABLE_RULESETS: Semgrep rulesets with descriptions
- CWE_FIX_RECOMMENDATIONS: Fix recommendations by CWE category
- CWE_TO_OWASP: CWE to OWASP mapping for common vulnerabilities
- LOCAL_PATTERNS: Local fallback patterns when Semgrep is not available
- LANGUAGE_EXTENSIONS: Language extension mapping
"""

from __future__ import annotations

from typing import Any

from aragora.analysis.codebase.sast.models import OWASPCategory, SASTSeverity

# Available Semgrep rulesets with descriptions
AVAILABLE_RULESETS: dict[str, dict[str, str]] = {
    # OWASP rulesets
    "p/owasp-top-ten": {
        "name": "OWASP Top 10",
        "description": "Rules covering OWASP Top 10 2021 vulnerabilities",
        "category": "owasp",
    },
    "p/owasp-top-ten-2017": {
        "name": "OWASP Top 10 2017",
        "description": "Rules covering OWASP Top 10 2017 vulnerabilities",
        "category": "owasp",
    },
    # Security rulesets
    "p/security-audit": {
        "name": "Security Audit",
        "description": "Comprehensive security audit rules",
        "category": "security",
    },
    "p/secrets": {
        "name": "Secrets Detection",
        "description": "Detection of hardcoded secrets and credentials",
        "category": "security",
    },
    "p/supply-chain": {
        "name": "Supply Chain",
        "description": "Supply chain security vulnerabilities",
        "category": "security",
    },
    # CWE rulesets
    "p/cwe-top-25": {
        "name": "CWE Top 25",
        "description": "CWE Top 25 Most Dangerous Software Weaknesses",
        "category": "cwe",
    },
    # Language-specific rulesets
    "p/python": {
        "name": "Python Security",
        "description": "Python-specific security rules",
        "category": "language",
    },
    "p/javascript": {
        "name": "JavaScript Security",
        "description": "JavaScript-specific security rules",
        "category": "language",
    },
    "p/typescript": {
        "name": "TypeScript Security",
        "description": "TypeScript-specific security rules",
        "category": "language",
    },
    "p/go": {
        "name": "Go Security",
        "description": "Go-specific security rules",
        "category": "language",
    },
    "p/java": {
        "name": "Java Security",
        "description": "Java-specific security rules",
        "category": "language",
    },
    "p/ruby": {
        "name": "Ruby Security",
        "description": "Ruby-specific security rules",
        "category": "language",
    },
    # Framework-specific
    "p/django": {
        "name": "Django Security",
        "description": "Django framework security rules",
        "category": "framework",
    },
    "p/flask": {
        "name": "Flask Security",
        "description": "Flask framework security rules",
        "category": "framework",
    },
    "p/react": {
        "name": "React Security",
        "description": "React framework security rules",
        "category": "framework",
    },
    "p/nodejs": {
        "name": "Node.js Security",
        "description": "Node.js security rules",
        "category": "framework",
    },
    # Additional rulesets
    "p/insecure-transport": {
        "name": "Insecure Transport",
        "description": "Detection of insecure transport layer configurations",
        "category": "security",
    },
    "p/jwt": {
        "name": "JWT Security",
        "description": "JSON Web Token security vulnerabilities",
        "category": "security",
    },
    "p/sql-injection": {
        "name": "SQL Injection",
        "description": "SQL injection vulnerability detection",
        "category": "injection",
    },
    "p/xss": {
        "name": "XSS",
        "description": "Cross-site scripting vulnerability detection",
        "category": "injection",
    },
    "p/command-injection": {
        "name": "Command Injection",
        "description": "OS command injection vulnerability detection",
        "category": "injection",
    },
}

# Fix recommendations by CWE category
CWE_FIX_RECOMMENDATIONS: dict[str, str] = {
    # Injection
    "CWE-78": "Use parameterized commands or shell escaping. Avoid shell=True with user input.",
    "CWE-79": "Sanitize user input before rendering. Use templating engines with auto-escaping.",
    "CWE-89": "Use parameterized queries or an ORM. Never concatenate user input in SQL.",
    "CWE-94": "Avoid eval/exec with user input. Use safe alternatives like ast.literal_eval.",
    "CWE-95": "Never use eval() with untrusted data. Parse data using safe methods.",
    # Cryptographic
    "CWE-327": "Use strong cryptographic algorithms (AES-256, SHA-256+). Avoid MD5/SHA1 for security.",
    "CWE-328": "Use bcrypt, scrypt, or Argon2 for password hashing. Never use MD5/SHA1.",
    "CWE-330": "Use cryptographically secure random number generators (secrets module in Python).",
    "CWE-338": "Replace weak PRNG with secrets.token_bytes() or os.urandom().",
    # Authentication
    "CWE-259": "Store credentials in environment variables or secure vaults, not in code.",
    "CWE-287": "Implement proper authentication. Verify credentials on every request.",
    "CWE-306": "Add authentication checks before accessing sensitive functionality.",
    "CWE-798": "Move hardcoded credentials to environment variables or secret management.",
    # Access Control
    "CWE-22": "Validate and sanitize file paths. Use os.path.realpath() and check allowed directories.",
    "CWE-284": "Implement proper access control checks. Follow principle of least privilege.",
    "CWE-352": "Implement CSRF tokens for state-changing operations.",
    "CWE-862": "Add authorization checks before accessing resources.",
    # Data Integrity
    "CWE-502": "Avoid deserializing untrusted data. Use safe serialization formats like JSON.",
    "CWE-494": "Verify integrity of downloaded code using checksums or signatures.",
    # Configuration
    "CWE-16": "Review security configuration. Disable debug mode in production.",
    "CWE-614": "Set Secure flag on cookies containing sensitive data.",
    "CWE-1004": "Set HttpOnly flag on session cookies.",
    # SSRF
    "CWE-918": "Validate and whitelist allowed URLs. Block internal network ranges.",
    # Logging
    "CWE-117": "Sanitize user input before logging to prevent log injection.",
    "CWE-532": "Avoid logging sensitive data. Mask or redact sensitive information.",
}

# CWE to OWASP mapping for common vulnerabilities
CWE_TO_OWASP: dict[str, OWASPCategory] = {
    # A01: Broken Access Control
    "CWE-22": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Path Traversal
    "CWE-23": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Relative Path Traversal
    "CWE-35": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Path Traversal
    "CWE-59": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Link Following
    "CWE-200": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Information Exposure
    "CWE-201": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Insertion of Sensitive Info
    "CWE-219": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Storage of File with Sensitive Data
    "CWE-264": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Permissions, Privileges
    "CWE-275": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Permission Issues
    "CWE-276": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Default Permissions
    "CWE-284": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Improper Access Control
    "CWE-285": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Improper Authorization
    "CWE-352": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # CSRF
    "CWE-359": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Privacy Violation
    "CWE-425": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Direct Request
    "CWE-639": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Authorization Bypass via IDOR
    "CWE-732": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Permission Assignment
    "CWE-862": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Missing Authorization
    "CWE-863": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Authorization
    # A02: Cryptographic Failures
    "CWE-261": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Weak Encoding
    "CWE-296": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Improper Certificate Validation
    "CWE-310": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Cryptographic Issues
    "CWE-319": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Cleartext Transmission
    "CWE-321": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Hard-coded Cryptographic Key
    "CWE-322": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Key Exchange without Entity Auth
    "CWE-323": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Reusing Nonce
    "CWE-324": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Use of Expired Key
    "CWE-325": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Missing Cryptographic Step
    "CWE-326": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Inadequate Encryption Strength
    "CWE-327": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Broken Crypto Algorithm
    "CWE-328": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Reversible One-Way Hash
    "CWE-329": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Not Using Random IV
    "CWE-330": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Insufficient Randomness
    "CWE-331": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Insufficient Entropy
    "CWE-335": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Incorrect Usage of Seeds
    "CWE-336": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Same Seed in PRNG
    "CWE-337": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Predictable Seed
    "CWE-338": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Weak PRNG
    "CWE-340": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Predictable from Observable State
    "CWE-347": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Improper Signature Verification
    "CWE-523": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Unprotected Credentials
    "CWE-720": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # OWASP Top Ten 2007 A8
    # A03: Injection
    "CWE-20": OWASPCategory.A03_INJECTION,  # Improper Input Validation
    "CWE-74": OWASPCategory.A03_INJECTION,  # Injection
    "CWE-75": OWASPCategory.A03_INJECTION,  # Failure to Sanitize Special Elements
    "CWE-77": OWASPCategory.A03_INJECTION,  # Command Injection
    "CWE-78": OWASPCategory.A03_INJECTION,  # OS Command Injection
    "CWE-79": OWASPCategory.A03_INJECTION,  # XSS
    "CWE-80": OWASPCategory.A03_INJECTION,  # Basic XSS
    "CWE-83": OWASPCategory.A03_INJECTION,  # XSS in Script Block
    "CWE-87": OWASPCategory.A03_INJECTION,  # XSS in Alternate Syntax
    "CWE-88": OWASPCategory.A03_INJECTION,  # Argument Injection
    "CWE-89": OWASPCategory.A03_INJECTION,  # SQL Injection
    "CWE-90": OWASPCategory.A03_INJECTION,  # LDAP Injection
    "CWE-91": OWASPCategory.A03_INJECTION,  # XML Injection
    "CWE-93": OWASPCategory.A03_INJECTION,  # CRLF Injection
    "CWE-94": OWASPCategory.A03_INJECTION,  # Code Injection
    "CWE-95": OWASPCategory.A03_INJECTION,  # Eval Injection
    "CWE-96": OWASPCategory.A03_INJECTION,  # Static Code Injection
    "CWE-97": OWASPCategory.A03_INJECTION,  # Server-Side Include Injection
    "CWE-98": OWASPCategory.A03_INJECTION,  # PHP Remote File Inclusion
    "CWE-99": OWASPCategory.A03_INJECTION,  # Resource Injection
    "CWE-113": OWASPCategory.A03_INJECTION,  # HTTP Response Splitting
    "CWE-116": OWASPCategory.A03_INJECTION,  # Improper Encoding
    "CWE-138": OWASPCategory.A03_INJECTION,  # Improper Neutralization of Special Elements
    "CWE-564": OWASPCategory.A03_INJECTION,  # SQL Injection: Hibernate
    "CWE-611": OWASPCategory.A03_INJECTION,  # XXE
    "CWE-643": OWASPCategory.A03_INJECTION,  # XPath Injection
    "CWE-652": OWASPCategory.A03_INJECTION,  # XQuery Injection
    "CWE-917": OWASPCategory.A03_INJECTION,  # Expression Language Injection
    # A04: Insecure Design
    "CWE-73": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of File Name
    "CWE-183": OWASPCategory.A04_INSECURE_DESIGN,  # Permissive Whitelist
    "CWE-209": OWASPCategory.A04_INSECURE_DESIGN,  # Error Message Information Exposure
    "CWE-213": OWASPCategory.A04_INSECURE_DESIGN,  # Intentional Information Exposure
    "CWE-235": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Handling of Extra Parameters
    "CWE-256": OWASPCategory.A04_INSECURE_DESIGN,  # Plaintext Storage of Password
    "CWE-257": OWASPCategory.A04_INSECURE_DESIGN,  # Storing Passwords in Recoverable Format
    "CWE-266": OWASPCategory.A04_INSECURE_DESIGN,  # Incorrect Privilege Assignment
    "CWE-269": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Privilege Management
    "CWE-280": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Handling of Insufficient Privileges
    "CWE-311": OWASPCategory.A04_INSECURE_DESIGN,  # Missing Encryption of Sensitive Data
    "CWE-312": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage of Sensitive Info
    "CWE-313": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage in File
    "CWE-316": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage in Memory
    "CWE-419": OWASPCategory.A04_INSECURE_DESIGN,  # Unprotected Primary Channel
    "CWE-430": OWASPCategory.A04_INSECURE_DESIGN,  # Deployment of Wrong Handler
    "CWE-434": OWASPCategory.A04_INSECURE_DESIGN,  # Unrestricted Upload
    "CWE-444": OWASPCategory.A04_INSECURE_DESIGN,  # HTTP Request Smuggling
    "CWE-451": OWASPCategory.A04_INSECURE_DESIGN,  # UI Misrepresentation
    "CWE-472": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of Assumed-Immutable
    "CWE-501": OWASPCategory.A04_INSECURE_DESIGN,  # Trust Boundary Violation
    "CWE-522": OWASPCategory.A04_INSECURE_DESIGN,  # Insufficiently Protected Credentials
    "CWE-525": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Browser Caching
    "CWE-539": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Persistent Cookies
    "CWE-579": OWASPCategory.A04_INSECURE_DESIGN,  # J2EE Bad Practices
    "CWE-598": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Query Strings
    "CWE-602": OWASPCategory.A04_INSECURE_DESIGN,  # Client-Side Enforcement of Server-Side Security
    "CWE-642": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of Critical State Data
    "CWE-646": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on File Name or Extension
    "CWE-650": OWASPCategory.A04_INSECURE_DESIGN,  # Trusting HTTP Permission Methods
    "CWE-653": OWASPCategory.A04_INSECURE_DESIGN,  # Insufficient Compartmentalization
    "CWE-656": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on Security Through Obscurity
    "CWE-657": OWASPCategory.A04_INSECURE_DESIGN,  # Violation of Secure Design Principles
    "CWE-799": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Control of Interaction Frequency
    "CWE-807": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on Untrusted Inputs
    "CWE-840": OWASPCategory.A04_INSECURE_DESIGN,  # Business Logic Errors
    "CWE-841": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Enforcement of Behavioral Workflow
    "CWE-927": OWASPCategory.A04_INSECURE_DESIGN,  # Implicit Intent
    # A05: Security Misconfiguration
    "CWE-2": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Environmental Security Flaw
    "CWE-11": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # ASP.NET Misconfiguration
    "CWE-13": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # ASP.NET Misconfiguration
    "CWE-15": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # External Control of System Setting
    "CWE-16": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Configuration
    "CWE-260": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Password in Configuration File
    "CWE-315": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Cleartext Storage in Cookie
    "CWE-520": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # .NET Misconfiguration
    "CWE-526": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Information Exposure Through Environment Variables
    "CWE-537": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Runtime Error Message Containing Sensitive Info
    "CWE-541": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Information Exposure Through Include Source Code
    "CWE-547": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Use of Hard-coded Security-relevant Constants
    # Note: CWE-611 (XXE) is mapped to A03_INJECTION above
    "CWE-614": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Sensitive Cookie Without Secure Attribute
    "CWE-756": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Missing Custom Error Page
    "CWE-776": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Recursive Entity Reference
    "CWE-942": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Permissive Cross-domain Policy
    "CWE-1004": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Sensitive Cookie Without HttpOnly
    "CWE-1032": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # OWASP Top Ten 2017 A6
    # A07: Identification and Authentication Failures
    "CWE-255": OWASPCategory.A07_AUTH_FAILURES,  # Credentials Management
    "CWE-259": OWASPCategory.A07_AUTH_FAILURES,  # Hard-coded Password
    "CWE-287": OWASPCategory.A07_AUTH_FAILURES,  # Improper Authentication
    "CWE-288": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass Using Alternate Path
    "CWE-290": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Spoofing
    "CWE-294": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Capture-replay
    "CWE-295": OWASPCategory.A07_AUTH_FAILURES,  # Improper Certificate Validation
    "CWE-297": OWASPCategory.A07_AUTH_FAILURES,  # Improper Validation of Certificate with Host Mismatch
    "CWE-300": OWASPCategory.A07_AUTH_FAILURES,  # Channel Accessible by Non-Endpoint
    "CWE-302": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Assumed-Immutable Data
    "CWE-304": OWASPCategory.A07_AUTH_FAILURES,  # Missing Critical Step in Authentication
    "CWE-306": OWASPCategory.A07_AUTH_FAILURES,  # Missing Authentication for Critical Function
    "CWE-307": OWASPCategory.A07_AUTH_FAILURES,  # Improper Restriction of Excessive Authentication Attempts
    "CWE-346": OWASPCategory.A07_AUTH_FAILURES,  # Origin Validation Error
    "CWE-384": OWASPCategory.A07_AUTH_FAILURES,  # Session Fixation
    "CWE-521": OWASPCategory.A07_AUTH_FAILURES,  # Weak Password Requirements
    "CWE-613": OWASPCategory.A07_AUTH_FAILURES,  # Insufficient Session Expiration
    "CWE-620": OWASPCategory.A07_AUTH_FAILURES,  # Unverified Password Change
    "CWE-640": OWASPCategory.A07_AUTH_FAILURES,  # Weak Password Recovery Mechanism
    "CWE-798": OWASPCategory.A07_AUTH_FAILURES,  # Hard-coded Credentials
    # A08: Software and Data Integrity Failures
    "CWE-345": OWASPCategory.A08_DATA_INTEGRITY,  # Insufficient Verification of Data Authenticity
    "CWE-353": OWASPCategory.A08_DATA_INTEGRITY,  # Missing Support for Integrity Check
    "CWE-426": OWASPCategory.A08_DATA_INTEGRITY,  # Untrusted Search Path
    "CWE-494": OWASPCategory.A08_DATA_INTEGRITY,  # Download of Code Without Integrity Check
    "CWE-502": OWASPCategory.A08_DATA_INTEGRITY,  # Deserialization of Untrusted Data
    "CWE-565": OWASPCategory.A08_DATA_INTEGRITY,  # Reliance on Cookies without Validation and Integrity Checking
    "CWE-784": OWASPCategory.A08_DATA_INTEGRITY,  # Reliance on Cookies in Security Decision
    "CWE-829": OWASPCategory.A08_DATA_INTEGRITY,  # Inclusion of Functionality from Untrusted Control Sphere
    "CWE-830": OWASPCategory.A08_DATA_INTEGRITY,  # Inclusion of Web Functionality from Untrusted Source
    "CWE-915": OWASPCategory.A08_DATA_INTEGRITY,  # Improperly Controlled Modification of Dynamically-Determined Object Attributes
    # A09: Security Logging and Monitoring Failures
    "CWE-117": OWASPCategory.A09_LOGGING_FAILURES,  # Improper Output Neutralization for Logs
    "CWE-223": OWASPCategory.A09_LOGGING_FAILURES,  # Omission of Security-relevant Information
    "CWE-532": OWASPCategory.A09_LOGGING_FAILURES,  # Information Exposure Through Log Files
    "CWE-778": OWASPCategory.A09_LOGGING_FAILURES,  # Insufficient Logging
    # A10: Server-Side Request Forgery
    "CWE-918": OWASPCategory.A10_SSRF,  # Server-Side Request Forgery
}

# Local fallback patterns when Semgrep is not available
LOCAL_PATTERNS: dict[str, dict[str, Any]] = {
    # SQL Injection patterns
    "sql-injection-python": {
        "pattern": r'execute\s*\(\s*[\'"].*%s.*[\'"]\s*%',
        "languages": ["python"],
        "message": "Potential SQL injection via string formatting",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-89",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "sql-injection-fstring": {
        "pattern": r'execute\s*\(\s*f[\'"].*\{.*\}.*[\'"]',
        "languages": ["python"],
        "message": "Potential SQL injection via f-string",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-89",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Command Injection
    "command-injection-subprocess": {
        "pattern": r'subprocess\.(call|run|Popen)\s*\(\s*[\'"].*\+.*shell\s*=\s*True',
        "languages": ["python"],
        "message": "Potential command injection with shell=True",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-78",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "command-injection-os-system": {
        "pattern": r"os\.system\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential command injection via os.system",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-78",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Eval/Exec injection
    "eval-injection": {
        "pattern": r"\beval\s*\([^)]*\)",
        "languages": ["python", "javascript"],
        "message": "Use of eval() is dangerous and may allow code injection",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-95",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "exec-injection": {
        "pattern": r"\bexec\s*\([^)]*\)",
        "languages": ["python"],
        "message": "Use of exec() is dangerous and may allow code injection",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-95",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Hardcoded credentials
    "hardcoded-password": {
        "pattern": r'(password|passwd|pwd)\s*=\s*[\'"][^\'"]{4,}[\'"]',
        "languages": ["python", "javascript", "java", "go"],
        "message": "Potential hardcoded password",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-259",
        "owasp": OWASPCategory.A07_AUTH_FAILURES,
    },
    "hardcoded-api-key": {
        "pattern": r'(api[_-]?key|apikey|api[_-]?secret)\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"]',
        "languages": ["python", "javascript", "java", "go"],
        "message": "Potential hardcoded API key",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-798",
        "owasp": OWASPCategory.A07_AUTH_FAILURES,
    },
    # XSS patterns
    "xss-innerHTML": {
        "pattern": r"\.innerHTML\s*=",
        "languages": ["javascript", "typescript"],
        "message": "Use of innerHTML may lead to XSS",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-79",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "xss-document-write": {
        "pattern": r"document\.write\s*\(",
        "languages": ["javascript", "typescript"],
        "message": "Use of document.write may lead to XSS",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-79",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Insecure deserialization
    "pickle-load": {
        "pattern": r"pickle\.loads?\s*\(",
        "languages": ["python"],
        "message": "Pickle deserialization of untrusted data is dangerous",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-502",
        "owasp": OWASPCategory.A08_DATA_INTEGRITY,
    },
    "yaml-unsafe-load": {
        "pattern": r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)",
        "languages": ["python"],
        "message": "YAML load without safe loader is dangerous",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-502",
        "owasp": OWASPCategory.A08_DATA_INTEGRITY,
    },
    # Weak cryptography
    "weak-hash-md5": {
        "pattern": r"(md5|MD5)\s*\(",
        "languages": ["python", "javascript", "java", "go"],
        "message": "MD5 is a weak hash algorithm",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-327",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
    "weak-hash-sha1": {
        "pattern": r"(sha1|SHA1)\s*\(",
        "languages": ["python", "javascript", "java", "go"],
        "message": "SHA1 is a weak hash algorithm",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-327",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
    # SSRF
    "ssrf-requests": {
        "pattern": r"requests\.(get|post|put|delete|head|options)\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential SSRF via user-controlled URL",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-918",
        "owasp": OWASPCategory.A10_SSRF,
    },
    # Path traversal
    "path-traversal": {
        "pattern": r"open\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential path traversal via string concatenation in open()",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-22",
        "owasp": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
    },
    # Debug enabled
    "debug-enabled": {
        "pattern": r"DEBUG\s*=\s*True",
        "languages": ["python"],
        "message": "Debug mode enabled in production may expose sensitive information",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-215",
        "owasp": OWASPCategory.A05_SECURITY_MISCONFIGURATION,
    },
    # JWT without verification
    "jwt-no-verify": {
        "pattern": r"jwt\.decode\s*\([^)]*verify\s*=\s*False",
        "languages": ["python"],
        "message": "JWT decoded without verification",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-347",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
}

# Language extension mapping
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py", ".pyw"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "java": [".java"],
    "ruby": [".rb"],
    "php": [".php"],
    "csharp": [".cs"],
}
