"""
Software Audit Type.

Specialized auditor for code and software artifacts targeting:
- SAST-style vulnerability detection (OWASP Top 10)
- Secret/API key detection
- License compliance (SPDX)
- Dependency vulnerability checking
- Code quality and security patterns
- Infrastructure misconfigurations

Designed for software companies, security teams, and DevSecOps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..base_auditor import AuditorCapabilities, AuditContext, BaseAuditor, ChunkData
from ..document_auditor import AuditFinding, AuditType, FindingSeverity

logger = logging.getLogger(__name__)


class SecurityCategory(str, Enum):
    """Categories of security findings."""

    INJECTION = "injection"
    XSS = "xss"
    AUTH = "authentication"
    CRYPTO = "cryptography"
    SECRETS = "secrets"
    CONFIG = "configuration"
    DEPENDENCY = "dependency"
    LICENSE = "license"
    INFRASTRUCTURE = "infrastructure"
    DATA_EXPOSURE = "data_exposure"
    ACCESS_CONTROL = "access_control"


class CWE(str, Enum):
    """Common Weakness Enumeration IDs for findings."""

    SQL_INJECTION = "CWE-89"
    COMMAND_INJECTION = "CWE-78"
    PATH_TRAVERSAL = "CWE-22"
    XSS = "CWE-79"
    HARDCODED_CREDS = "CWE-798"
    WEAK_CRYPTO = "CWE-327"
    INSECURE_RANDOM = "CWE-330"
    SSRF = "CWE-918"
    XXE = "CWE-611"
    DESERIALIZATION = "CWE-502"
    OPEN_REDIRECT = "CWE-601"
    LOG_INJECTION = "CWE-117"
    SENSITIVE_DATA = "CWE-200"
    MISSING_AUTH = "CWE-306"


@dataclass
class VulnerabilityPattern:
    """Pattern for detecting security vulnerabilities."""

    name: str
    pattern: str
    category: SecurityCategory
    severity: FindingSeverity
    cwe: Optional[CWE]
    description: str
    recommendation: str
    languages: list[str] = field(default_factory=lambda: ["*"])  # * = all languages
    flags: int = re.IGNORECASE | re.MULTILINE


@dataclass
class SecretPattern:
    """Pattern for detecting secrets and API keys."""

    name: str
    pattern: str
    severity: FindingSeverity
    entropy_check: bool = False  # Whether to also check entropy
    description: str = ""


@dataclass
class LicenseInfo:
    """Information about a detected license."""

    spdx_id: str
    name: str
    category: str  # permissive, copyleft, proprietary
    osi_approved: bool
    location: str


class SoftwareAuditor(BaseAuditor):
    """
    Auditor for source code and software artifacts.

    Detects:
    - OWASP Top 10 vulnerabilities
    - Hardcoded secrets and API keys
    - License compliance issues
    - Insecure coding patterns
    - Infrastructure misconfigurations
    - Dependency vulnerabilities
    """

    # SAST vulnerability patterns
    VULNERABILITY_PATTERNS: list[VulnerabilityPattern] = [
        # SQL Injection
        VulnerabilityPattern(
            name="sql_injection_string_concat",
            pattern=r"(?:execute|query|cursor\.execute|raw|rawQuery)\s*\(\s*['\"].*?%[sd]|(?:execute|query)\s*\(\s*['\"].*?\+",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.SQL_INJECTION,
            description="Potential SQL injection via string concatenation or formatting",
            recommendation="Use parameterized queries or prepared statements instead of string concatenation",
            languages=["python", "javascript", "java", "php"],
        ),
        VulnerabilityPattern(
            name="sql_injection_fstring",
            pattern=r"(?:execute|query|cursor\.execute)\s*\(\s*f['\"].*?\{",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.SQL_INJECTION,
            description="SQL injection via f-string interpolation",
            recommendation="Use parameterized queries instead of f-strings for SQL",
            languages=["python"],
        ),
        # Command Injection
        VulnerabilityPattern(
            name="command_injection_shell",
            pattern=r"(?:subprocess\.(?:call|run|Popen)|os\.(?:system|popen)|exec|shell_exec)\s*\([^)]*(?:\+|%|format|\{)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.COMMAND_INJECTION,
            description="Potential command injection via shell execution",
            recommendation="Avoid shell=True and use subprocess with list arguments, sanitize all inputs",
            languages=["python", "javascript", "php"],
        ),
        VulnerabilityPattern(
            name="command_injection_eval",
            pattern=r"\b(?:eval|exec)\s*\([^)]*(?:input|request|user|param|arg)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.COMMAND_INJECTION,
            description="Code execution with user-controlled input",
            recommendation="Never use eval/exec with user input; use safe alternatives",
            languages=["python", "javascript", "php"],
        ),
        # Path Traversal
        VulnerabilityPattern(
            name="path_traversal",
            pattern=r"(?:open|read|write|file|include|require)\s*\([^)]*(?:\.\./|\.\.\\|request|user|param)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=CWE.PATH_TRAVERSAL,
            description="Potential path traversal vulnerability",
            recommendation="Validate and sanitize file paths; use safe path joining; implement allowlists",
            languages=["*"],
        ),
        # XSS
        VulnerabilityPattern(
            name="xss_innerhtml",
            pattern=r"\.innerHTML\s*=|\.outerHTML\s*=|document\.write\s*\(",
            category=SecurityCategory.XSS,
            severity=FindingSeverity.HIGH,
            cwe=CWE.XSS,
            description="DOM-based XSS via innerHTML/document.write",
            recommendation="Use textContent instead of innerHTML; sanitize HTML input",
            languages=["javascript", "typescript"],
        ),
        VulnerabilityPattern(
            name="xss_dangerously_set",
            pattern=r"dangerouslySetInnerHTML",
            category=SecurityCategory.XSS,
            severity=FindingSeverity.HIGH,
            cwe=CWE.XSS,
            description="React XSS via dangerouslySetInnerHTML",
            recommendation="Sanitize HTML content with DOMPurify before setting",
            languages=["javascript", "typescript"],
        ),
        # Cryptography Issues
        VulnerabilityPattern(
            name="weak_crypto_md5",
            pattern=r"\b(?:md5|MD5)\s*\(|hashlib\.md5|crypto\.createHash\s*\(['\"]md5['\"]",
            category=SecurityCategory.CRYPTO,
            severity=FindingSeverity.MEDIUM,
            cwe=CWE.WEAK_CRYPTO,
            description="Use of weak MD5 hash algorithm",
            recommendation="Use SHA-256 or stronger for hashing; use bcrypt/argon2 for passwords",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="weak_crypto_sha1",
            pattern=r"\b(?:sha1|SHA1)\s*\(|hashlib\.sha1|crypto\.createHash\s*\(['\"]sha1['\"]",
            category=SecurityCategory.CRYPTO,
            severity=FindingSeverity.MEDIUM,
            cwe=CWE.WEAK_CRYPTO,
            description="Use of weak SHA-1 hash algorithm",
            recommendation="Use SHA-256 or stronger for cryptographic operations",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="weak_crypto_des",
            pattern=r"\bDES\b|DESede|Blowfish|RC4",
            category=SecurityCategory.CRYPTO,
            severity=FindingSeverity.HIGH,
            cwe=CWE.WEAK_CRYPTO,
            description="Use of deprecated/weak encryption algorithm",
            recommendation="Use AES-256-GCM or ChaCha20-Poly1305",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="insecure_random",
            pattern=r"\b(?:Math\.random|random\.random|rand\(\)|mt_rand)\b",
            category=SecurityCategory.CRYPTO,
            severity=FindingSeverity.MEDIUM,
            cwe=CWE.INSECURE_RANDOM,
            description="Use of insecure random number generator",
            recommendation="Use secrets module (Python) or crypto.getRandomValues (JS) for security-sensitive operations",
            languages=["*"],
        ),
        # SSRF
        VulnerabilityPattern(
            name="ssrf_request",
            pattern=r"(?:requests\.get|fetch|urllib\.request|http\.get)\s*\([^)]*(?:request|user|param|url_for)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=CWE.SSRF,
            description="Potential Server-Side Request Forgery (SSRF)",
            recommendation="Validate and allowlist URLs; block internal IP ranges",
            languages=["*"],
        ),
        # XXE
        VulnerabilityPattern(
            name="xxe_parser",
            pattern=r"(?:XMLParser|parseString|etree\.parse|DOMParser|SAXParser).*(?:<!ENTITY|SYSTEM|PUBLIC)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=CWE.XXE,
            description="Potential XML External Entity (XXE) injection",
            recommendation="Disable external entity processing; use defusedxml in Python",
            languages=["*"],
        ),
        # Deserialization
        VulnerabilityPattern(
            name="insecure_deserialization",
            pattern=r"\b(?:pickle\.loads?|yaml\.(?:load|unsafe_load)|unserialize|ObjectInputStream\.readObject)\b",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.DESERIALIZATION,
            description="Insecure deserialization may allow remote code execution",
            recommendation="Use yaml.safe_load; avoid pickle with untrusted data; validate before deserializing",
            languages=["*"],
        ),
        # Authentication Issues
        VulnerabilityPattern(
            name="jwt_none_algorithm",
            pattern=r"algorithm\s*[=:]\s*['\"]none['\"]|alg\s*[=:]\s*['\"]none['\"]",
            category=SecurityCategory.AUTH,
            severity=FindingSeverity.CRITICAL,
            cwe=CWE.MISSING_AUTH,
            description="JWT with 'none' algorithm allows token forgery",
            recommendation="Always specify a strong algorithm (RS256, ES256); reject 'none' algorithm",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="hardcoded_jwt_secret",
            pattern=r"(?:jwt|JWT).*(?:secret|key)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
            category=SecurityCategory.AUTH,
            severity=FindingSeverity.HIGH,
            cwe=CWE.HARDCODED_CREDS,
            description="Hardcoded JWT secret in source code",
            recommendation="Store secrets in environment variables or secret management system",
            languages=["*"],
        ),
        # Infrastructure/Config Issues
        VulnerabilityPattern(
            name="debug_enabled",
            pattern=r"DEBUG\s*[=:]\s*(?:True|true|1)|debug\s*[=:]\s*(?:True|true|1)",
            category=SecurityCategory.CONFIG,
            severity=FindingSeverity.MEDIUM,
            cwe=None,
            description="Debug mode enabled (may expose sensitive information)",
            recommendation="Disable debug mode in production environments",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="cors_allow_all",
            pattern=r"(?:Access-Control-Allow-Origin|cors)\s*[=:]\s*['\"]?\*['\"]?",
            category=SecurityCategory.CONFIG,
            severity=FindingSeverity.MEDIUM,
            cwe=None,
            description="CORS allows all origins (potential security risk)",
            recommendation="Specify allowed origins explicitly; avoid wildcard in production",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="ssl_verify_disabled",
            pattern=r"verify\s*[=:]\s*False|VERIFY_NONE|SSL_VERIFY_NONE|rejectUnauthorized\s*[=:]\s*false",
            category=SecurityCategory.CONFIG,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="SSL certificate verification disabled",
            recommendation="Always verify SSL certificates in production",
            languages=["*"],
        ),
        # Modern Framework Vulnerabilities
        VulnerabilityPattern(
            name="nextjs_getserversideprops_leak",
            pattern=r"getServerSideProps.*(?:password|secret|key|token|credential)['\"]?\s*[:=]",
            category=SecurityCategory.DATA_EXPOSURE,
            severity=FindingSeverity.HIGH,
            cwe=CWE.SENSITIVE_DATA,
            description="Potential data leak through getServerSideProps props",
            recommendation="Filter sensitive data before returning props; use environment variables on server only",
            languages=["javascript", "typescript"],
        ),
        VulnerabilityPattern(
            name="prototype_pollution",
            pattern=r"Object\.assign\s*\([^,]+,\s*(?:req|request|user|param)|__proto__|constructor\[.prototype",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="Potential prototype pollution vulnerability",
            recommendation="Use Object.create(null) or Map; validate object keys",
            languages=["javascript", "typescript"],
        ),
        VulnerabilityPattern(
            name="graphql_introspection",
            pattern=r"introspection\s*[=:]\s*true|enableIntrospection",
            category=SecurityCategory.CONFIG,
            severity=FindingSeverity.MEDIUM,
            cwe=None,
            description="GraphQL introspection enabled (information disclosure)",
            recommendation="Disable introspection in production environments",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="open_redirect",
            pattern=r"(?:redirect|location)\s*[=:]\s*(?:req|request|params?|query)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.MEDIUM,
            cwe=CWE.OPEN_REDIRECT,
            description="Potential open redirect vulnerability",
            recommendation="Validate redirect URLs against allowlist; use relative paths",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="template_injection",
            pattern=r"(?:render|template)\s*\([^)]*(?:req|request|user|input|param)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="Potential server-side template injection",
            recommendation="Never pass user input directly to template rendering",
            languages=["python", "javascript", "ruby"],
        ),
        VulnerabilityPattern(
            name="nosql_injection",
            pattern=r"(?:find|update|delete)(?:One|Many)?\s*\([^)]*\$(?:where|regex|ne|gt|lt)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.HIGH,
            cwe=CWE.SQL_INJECTION,
            description="Potential NoSQL injection via MongoDB operators",
            recommendation="Sanitize user input; avoid $where and user-controlled operators",
            languages=["javascript", "typescript", "python"],
        ),
        VulnerabilityPattern(
            name="regex_dos",
            pattern=r"new\s+RegExp\s*\([^)]*(?:req|request|user|input|param)",
            category=SecurityCategory.INJECTION,
            severity=FindingSeverity.MEDIUM,
            cwe=None,
            description="User-controlled regex may cause ReDoS",
            recommendation="Validate regex patterns; use timeouts; avoid user-controlled regex",
            languages=["javascript", "typescript"],
        ),
        VulnerabilityPattern(
            name="mass_assignment",
            pattern=r"(?:create|update)\s*\([^)]*(?:req\.body|params|body\[)",
            category=SecurityCategory.ACCESS_CONTROL,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="Potential mass assignment vulnerability",
            recommendation="Whitelist allowed fields; use DTOs or strong typing",
            languages=["*"],
        ),
        VulnerabilityPattern(
            name="unsafe_file_upload",
            pattern=r"(?:multer|upload|file).*(?:any|all|destination\s*[=:]\s*['\"])",
            category=SecurityCategory.CONFIG,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="File upload may accept dangerous file types",
            recommendation="Validate file types, size, and content; use secure storage",
            languages=["javascript", "typescript", "python"],
        ),
        VulnerabilityPattern(
            name="unsafe_html_render",
            pattern=r"(?:render|html)\s*\([^)]*\+|innerHTML.*\+|createContextualFragment",
            category=SecurityCategory.XSS,
            severity=FindingSeverity.HIGH,
            cwe=CWE.XSS,
            description="Dynamic HTML rendering with concatenation",
            recommendation="Use template literals with proper escaping; sanitize HTML",
            languages=["javascript", "typescript"],
        ),
        VulnerabilityPattern(
            name="express_session_insecure",
            pattern=r"session\s*\([^)]*(?:secure\s*[=:]\s*false|httpOnly\s*[=:]\s*false)",
            category=SecurityCategory.AUTH,
            severity=FindingSeverity.HIGH,
            cwe=None,
            description="Insecure session cookie configuration",
            recommendation="Enable secure and httpOnly flags for session cookies",
            languages=["javascript", "typescript"],
        ),
    ]

    # Secret detection patterns
    SECRET_PATTERNS: list[SecretPattern] = [
        SecretPattern(
            name="aws_access_key",
            pattern=r"(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
            severity=FindingSeverity.CRITICAL,
            description="AWS Access Key ID",
        ),
        SecretPattern(
            name="aws_secret_key",
            pattern=r"(?:aws)?_?(?:secret)?_?(?:access)?_?key['\"]?\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}",
            severity=FindingSeverity.CRITICAL,
            entropy_check=True,
            description="AWS Secret Access Key",
        ),
        SecretPattern(
            name="github_token",
            pattern=r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}",
            severity=FindingSeverity.CRITICAL,
            description="GitHub Personal Access Token",
        ),
        SecretPattern(
            name="github_oauth",
            pattern=r"github.*['\"]?[0-9a-fA-F]{40}['\"]?",
            severity=FindingSeverity.HIGH,
            description="GitHub OAuth Token",
        ),
        SecretPattern(
            name="openai_key",
            pattern=r"sk-[A-Za-z0-9]{48}",
            severity=FindingSeverity.CRITICAL,
            description="OpenAI API Key",
        ),
        SecretPattern(
            name="anthropic_key",
            pattern=r"sk-ant-[A-Za-z0-9-]{90,}",
            severity=FindingSeverity.CRITICAL,
            description="Anthropic API Key",
        ),
        SecretPattern(
            name="stripe_key",
            pattern=r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}",
            severity=FindingSeverity.CRITICAL,
            description="Stripe API Key",
        ),
        SecretPattern(
            name="slack_token",
            pattern=r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
            severity=FindingSeverity.HIGH,
            description="Slack Token",
        ),
        SecretPattern(
            name="slack_webhook",
            pattern=r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
            severity=FindingSeverity.MEDIUM,
            description="Slack Webhook URL",
        ),
        SecretPattern(
            name="google_api_key",
            pattern=r"AIza[0-9A-Za-z_-]{35}",
            severity=FindingSeverity.HIGH,
            description="Google API Key",
        ),
        SecretPattern(
            name="private_key",
            pattern=r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
            severity=FindingSeverity.CRITICAL,
            description="Private Key",
        ),
        SecretPattern(
            name="password_in_code",
            pattern=r"(?:password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{6,}['\"]",
            severity=FindingSeverity.HIGH,
            entropy_check=True,
            description="Hardcoded Password",
        ),
        SecretPattern(
            name="api_key_generic",
            pattern=r"(?:api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_-]{20,}['\"]",
            severity=FindingSeverity.HIGH,
            entropy_check=True,
            description="Generic API Key",
        ),
        SecretPattern(
            name="bearer_token",
            pattern=r"(?:bearer|authorization)\s*[=:]\s*['\"]?Bearer\s+[A-Za-z0-9_-]{20,}",
            severity=FindingSeverity.HIGH,
            description="Bearer Token",
        ),
        SecretPattern(
            name="database_url",
            pattern=r"(?:postgres|mysql|mongodb|redis)://[^:]+:[^@]+@[^/]+",
            severity=FindingSeverity.CRITICAL,
            description="Database Connection String with Credentials",
        ),
        # Additional modern API keys
        SecretPattern(
            name="vercel_token",
            pattern=r"vercel_[A-Za-z0-9]{24,}",
            severity=FindingSeverity.HIGH,
            description="Vercel API Token",
        ),
        SecretPattern(
            name="supabase_key",
            pattern=r"(?:sbp|eyJ)[A-Za-z0-9_-]{30,}",
            severity=FindingSeverity.HIGH,
            entropy_check=True,
            description="Supabase API Key",
        ),
        SecretPattern(
            name="firebase_key",
            pattern=r"(?:firebase|FIREBASE).*?['\"]?[A-Za-z0-9_-]{35,}['\"]?",
            severity=FindingSeverity.HIGH,
            entropy_check=True,
            description="Firebase API Key or Config",
        ),
        SecretPattern(
            name="twilio_key",
            pattern=r"(?:SK|AC)[a-fA-F0-9]{32}",
            severity=FindingSeverity.HIGH,
            description="Twilio API Key",
        ),
        SecretPattern(
            name="sendgrid_key",
            pattern=r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
            severity=FindingSeverity.HIGH,
            description="SendGrid API Key",
        ),
        SecretPattern(
            name="mailchimp_key",
            pattern=r"[a-f0-9]{32}-us[0-9]{1,2}",
            severity=FindingSeverity.HIGH,
            description="Mailchimp API Key",
        ),
        SecretPattern(
            name="npm_token",
            pattern=r"npm_[A-Za-z0-9]{36}",
            severity=FindingSeverity.CRITICAL,
            description="npm Access Token",
        ),
        SecretPattern(
            name="pypi_token",
            pattern=r"pypi-[A-Za-z0-9_-]{50,}",
            severity=FindingSeverity.CRITICAL,
            description="PyPI API Token",
        ),
        SecretPattern(
            name="discord_token",
            pattern=r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}",
            severity=FindingSeverity.HIGH,
            description="Discord Bot Token",
        ),
        SecretPattern(
            name="datadog_api_key",
            pattern=r"(?:DD|dd)_?(?:API|APP)_?KEY['\"]?\s*[=:]\s*['\"]?[a-fA-F0-9]{32,}",
            severity=FindingSeverity.HIGH,
            description="Datadog API/APP Key",
        ),
        SecretPattern(
            name="heroku_api_key",
            pattern=r"[hH]eroku.*['\"]?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}['\"]?",
            severity=FindingSeverity.HIGH,
            description="Heroku API Key",
        ),
        SecretPattern(
            name="cloudflare_api_key",
            pattern=r"(?:cloudflare|CF).*['\"]?[a-zA-Z0-9_-]{37,40}['\"]?",
            severity=FindingSeverity.HIGH,
            entropy_check=True,
            description="Cloudflare API Token",
        ),
    ]

    # License patterns (SPDX identifiers)
    LICENSE_PATTERNS: dict[str, dict[str, Any]] = {
        # Permissive
        "MIT": {
            "category": "permissive",
            "osi_approved": True,
            "patterns": [r"\bMIT\s+License\b", r"SPDX.*MIT"],
        },
        "Apache-2.0": {
            "category": "permissive",
            "osi_approved": True,
            "patterns": [r"Apache\s+License.*2\.0", r"SPDX.*Apache-2\.0"],
        },
        "BSD-2-Clause": {
            "category": "permissive",
            "osi_approved": True,
            "patterns": [r"BSD\s+2-Clause", r"Simplified\s+BSD"],
        },
        "BSD-3-Clause": {
            "category": "permissive",
            "osi_approved": True,
            "patterns": [r"BSD\s+3-Clause", r"New\s+BSD"],
        },
        "ISC": {"category": "permissive", "osi_approved": True, "patterns": [r"\bISC\s+License\b"]},
        # Copyleft
        "GPL-2.0": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"GNU\s+General\s+Public\s+License.*[Vv]ersion\s*2", r"GPL-?2"],
        },
        "GPL-3.0": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"GNU\s+General\s+Public\s+License.*[Vv]ersion\s*3", r"GPL-?3"],
        },
        "LGPL-2.1": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"GNU\s+Lesser.*2\.1", r"LGPL-?2\.1"],
        },
        "LGPL-3.0": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"GNU\s+Lesser.*3\.0", r"LGPL-?3"],
        },
        "AGPL-3.0": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"GNU\s+Affero.*3\.0", r"AGPL-?3"],
        },
        "MPL-2.0": {
            "category": "copyleft",
            "osi_approved": True,
            "patterns": [r"Mozilla\s+Public\s+License.*2\.0"],
        },
        # Proprietary/Restrictive
        "BUSL-1.1": {
            "category": "proprietary",
            "osi_approved": False,
            "patterns": [r"Business\s+Source\s+License"],
        },
        "SSPL": {
            "category": "proprietary",
            "osi_approved": False,
            "patterns": [r"Server\s+Side\s+Public\s+License"],
        },
        "CC-BY-NC": {
            "category": "restrictive",
            "osi_approved": False,
            "patterns": [r"Creative\s+Commons.*NonCommercial"],
        },
    }

    @property
    def audit_type_id(self) -> str:
        return "software"

    @property
    def display_name(self) -> str:
        return "Software Security"

    @property
    def description(self) -> str:
        return (
            "Security analysis for source code including SAST-style vulnerability detection, "
            "secret scanning, license compliance, and infrastructure security"
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> AuditorCapabilities:
        return AuditorCapabilities(
            supports_chunk_analysis=True,
            supports_cross_document=True,
            supports_streaming=False,
            requires_llm=True,
            supported_doc_types=[
                "py",
                "js",
                "ts",
                "tsx",
                "jsx",
                "java",
                "go",
                "rb",
                "php",
                "c",
                "cpp",
                "h",
                "hpp",
                "cs",
                "swift",
                "kt",
                "rs",
                "scala",
                "yaml",
                "yml",
                "json",
                "xml",
                "toml",
                "ini",
                "env",
                "dockerfile",
                "tf",
                "hcl",
                "sh",
                "bash",
                "ps1",
            ],
            max_chunk_size=10000,
            custom_capabilities={
                "sast_scanning": True,
                "secret_detection": True,
                "license_compliance": True,
                "dependency_check": True,
                "infrastructure_analysis": True,
            },
        )

    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> list[AuditFinding]:
        """Analyze a code chunk for security issues."""
        findings: list[AuditFinding] = []
        text = chunk.content

        # Detect file type for language-specific patterns
        file_ext = self._get_file_extension(chunk.document_id)

        # Check vulnerability patterns
        findings.extend(self._check_vulnerabilities(text, chunk, file_ext))

        # Check for secrets
        findings.extend(self._check_secrets(text, chunk))

        # Check for license information
        findings.extend(self._check_licenses(text, chunk))

        # Check for dangerous patterns
        findings.extend(self._check_dangerous_patterns(text, chunk))

        return findings

    async def cross_document_analysis(
        self,
        chunks: list[ChunkData],
        context: AuditContext,
    ) -> list[AuditFinding]:
        """Analyze across files for cross-cutting security issues."""
        findings: list[AuditFinding] = []

        # Collect license information across all files
        all_licenses: list[LicenseInfo] = []
        for chunk in chunks:
            licenses = self._extract_licenses(chunk.content, chunk.document_id)
            all_licenses.extend(licenses)

        # Check for license compatibility issues
        findings.extend(self._check_license_compatibility(all_licenses))

        # Check for dependency patterns across files
        findings.extend(self._check_dependency_patterns(chunks))

        return findings

    def _get_file_extension(self, document_id: str) -> str:
        """Extract file extension from document ID."""
        if "." in document_id:
            return document_id.rsplit(".", 1)[-1].lower()
        return ""

    def _check_vulnerabilities(
        self,
        text: str,
        chunk: ChunkData,
        file_ext: str,
    ) -> list[AuditFinding]:
        """Check for security vulnerabilities."""
        findings = []

        for vuln in self.VULNERABILITY_PATTERNS:
            # Check if pattern applies to this file type
            if vuln.languages != ["*"] and file_ext not in vuln.languages:
                continue

            try:
                pattern = re.compile(vuln.pattern, vuln.flags)
                matches = pattern.finditer(text)

                for match in matches:
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    evidence = text[start:end]

                    # Calculate line number
                    line_num = text[: match.start()].count("\n") + 1

                    cwe_str = f" ({vuln.cwe.value})" if vuln.cwe else ""

                    findings.append(
                        AuditFinding(
                            title=f"Security: {vuln.name.replace('_', ' ').title()}{cwe_str}",
                            description=vuln.description,
                            severity=vuln.severity,
                            category=vuln.category.value,
                            audit_type=AuditType.SECURITY,
                            document_id=chunk.document_id,
                            confidence=0.80,
                            evidence_text=f"Line {line_num}: ...{evidence}...",
                            evidence_location=f"Chunk {chunk.id}, line {line_num}",
                            recommendation=vuln.recommendation,
                            found_by="software_auditor",
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid regex pattern {vuln.name}: {e}")

        return findings

    def _check_secrets(
        self,
        text: str,
        chunk: ChunkData,
    ) -> list[AuditFinding]:
        """Check for hardcoded secrets and API keys."""
        findings = []

        for secret in self.SECRET_PATTERNS:
            try:
                pattern = re.compile(secret.pattern, re.IGNORECASE | re.MULTILINE)
                matches = pattern.finditer(text)

                for match in matches:
                    # Skip if in comment or test file
                    if self._is_likely_example(text, match):
                        continue

                    # Optionally check entropy for generic patterns
                    if secret.entropy_check:
                        matched_text = match.group(0)
                        if not self._has_high_entropy(matched_text):
                            continue

                    # Get context (mask the actual secret)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]

                    # Mask the secret in evidence
                    secret_value = match.group(0)
                    masked = (
                        secret_value[:8] + "..." + secret_value[-4:]
                        if len(secret_value) > 12
                        else "***REDACTED***"
                    )
                    masked_context = context.replace(secret_value, masked)

                    line_num = text[: match.start()].count("\n") + 1

                    findings.append(
                        AuditFinding(
                            title=f"Secret: {secret.name.replace('_', ' ').title()}",
                            description=f"{secret.description} detected in source code",
                            severity=secret.severity,
                            category=SecurityCategory.SECRETS.value,
                            audit_type=AuditType.SECURITY,
                            document_id=chunk.document_id,
                            confidence=0.90,
                            evidence_text=f"Line {line_num}: {masked_context}",
                            evidence_location=f"Chunk {chunk.id}, line {line_num}",
                            recommendation="Remove hardcoded secrets; use environment variables or secret management (Vault, AWS Secrets Manager)",
                            found_by="software_auditor",
                        )
                    )
            except re.error as e:
                logger.warning(f"Invalid secret pattern {secret.name}: {e}")

        return findings

    def _is_likely_example(self, text: str, match: re.Match) -> bool:
        """Check if a match is likely an example/placeholder rather than a real secret."""
        context_start = max(0, match.start() - 100)
        context = text[context_start : match.start()].lower()

        # Check for common example indicators
        example_indicators = [
            "example",
            "sample",
            "test",
            "placeholder",
            "dummy",
            "fake",
            "todo",
            "fixme",
            "xxx",
            "your_",
            "replace_",
            "<your",
        ]

        return any(indicator in context for indicator in example_indicators)

    def _has_high_entropy(self, text: str) -> bool:
        """Check if text has high entropy (likely to be a real secret)."""
        import math
        from collections import Counter

        if len(text) < 8:
            return False

        # Calculate Shannon entropy
        counter = Counter(text)
        length = len(text)
        entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())

        # High entropy threshold (random strings typically > 4.0)
        return entropy > 3.5

    def _check_licenses(
        self,
        text: str,
        chunk: ChunkData,
    ) -> list[AuditFinding]:
        """Check for license declarations."""
        findings = []
        licenses = self._extract_licenses(text, chunk.document_id)

        for license_info in licenses:
            # Flag copyleft licenses that may have compliance implications
            if license_info.category == "copyleft":
                findings.append(
                    AuditFinding(
                        title=f"License: {license_info.spdx_id} (Copyleft)",
                        description=(
                            f"Copyleft license {license_info.name} detected. "
                            "This license may require derivative works to be released under the same license."
                        ),
                        severity=FindingSeverity.MEDIUM,
                        category=SecurityCategory.LICENSE.value,
                        audit_type=AuditType.COMPLIANCE,
                        document_id=chunk.document_id,
                        confidence=0.85,
                        evidence_text=f"License: {license_info.name}",
                        evidence_location=license_info.location,
                        recommendation="Review copyleft obligations; ensure compliance with license terms",
                        found_by="software_auditor",
                    )
                )

            # Flag non-OSI-approved licenses
            if not license_info.osi_approved:
                findings.append(
                    AuditFinding(
                        title=f"License: {license_info.spdx_id} (Non-OSI)",
                        description=(
                            f"Non-OSI-approved license {license_info.name} detected. "
                            "This may have restrictive terms that limit usage."
                        ),
                        severity=FindingSeverity.HIGH,
                        category=SecurityCategory.LICENSE.value,
                        audit_type=AuditType.COMPLIANCE,
                        document_id=chunk.document_id,
                        confidence=0.85,
                        evidence_text=f"License: {license_info.name}",
                        evidence_location=license_info.location,
                        recommendation="Review license terms carefully; consider alternative libraries with permissive licenses",
                        found_by="software_auditor",
                    )
                )

        return findings

    def _extract_licenses(self, text: str, document_id: str) -> list[LicenseInfo]:
        """Extract license information from text."""
        licenses = []

        for spdx_id, info in self.LICENSE_PATTERNS.items():
            for pattern in info["patterns"]:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        licenses.append(
                            LicenseInfo(
                                spdx_id=spdx_id,
                                name=spdx_id,
                                category=info["category"],
                                osi_approved=info["osi_approved"],
                                location=document_id,
                            )
                        )
                        break  # Only add each license once per file
                except re.error:
                    continue

        return licenses

    def _check_dangerous_patterns(
        self,
        text: str,
        chunk: ChunkData,
    ) -> list[AuditFinding]:
        """Check for other dangerous coding patterns."""
        findings = []

        dangerous_patterns = [
            (
                r"TODO.*(?:security|vuln|hack|fix)",
                FindingSeverity.LOW,
                "Security-related TODO comment",
            ),
            (
                r"FIXME.*(?:security|auth|password)",
                FindingSeverity.MEDIUM,
                "Security-related FIXME comment",
            ),
            (
                r"(?:console\.log|print|printf|System\.out).*(?:password|secret|key|token)",
                FindingSeverity.MEDIUM,
                "Sensitive data in logging",
            ),
            (
                r"(?://|#)\s*(?:disable|ignore).*(?:security|auth|ssl)",
                FindingSeverity.HIGH,
                "Security check disabled",
            ),
            (
                r"(?:chmod|permissions?).*777|0777",
                FindingSeverity.HIGH,
                "World-writable permissions",
            ),
        ]

        for pattern, severity, description in dangerous_patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = text[: match.start()].count("\n") + 1
                    findings.append(
                        AuditFinding(
                            title=f"Code Quality: {description}",
                            description=description,
                            severity=severity,
                            category=SecurityCategory.CONFIG.value,
                            audit_type=AuditType.SECURITY,
                            document_id=chunk.document_id,
                            confidence=0.70,
                            evidence_text=f"Line {line_num}: {match.group(0)[:100]}",
                            evidence_location=f"Chunk {chunk.id}, line {line_num}",
                            recommendation="Address the flagged issue before deployment",
                            found_by="software_auditor",
                        )
                    )
            except re.error:
                continue

        return findings

    def _check_license_compatibility(
        self,
        licenses: list[LicenseInfo],
    ) -> list[AuditFinding]:
        """Check for license compatibility issues across files."""
        findings = []

        # Group by category
        categories = {}
        for lic in licenses:
            if lic.category not in categories:
                categories[lic.category] = []
            categories[lic.category].append(lic)

        # Check for copyleft + proprietary mix
        if "copyleft" in categories and "proprietary" in categories:
            findings.append(
                AuditFinding(
                    title="License: Incompatibility Risk",
                    description=(
                        f"Both copyleft ({', '.join(l.spdx_id for l in categories['copyleft'])}) "
                        f"and proprietary ({', '.join(l.spdx_id for l in categories['proprietary'])}) "
                        "licenses detected. This combination may have legal implications."
                    ),
                    severity=FindingSeverity.HIGH,
                    category=SecurityCategory.LICENSE.value,
                    audit_type=AuditType.COMPLIANCE,
                    document_id=categories["copyleft"][0].location,
                    confidence=0.75,
                    evidence_text="Mixed license types across codebase",
                    evidence_location="Cross-file analysis",
                    recommendation="Consult legal counsel to ensure license compatibility",
                    found_by="software_auditor",
                )
            )

        # Check for AGPL (network copyleft) with proprietary
        agpl_licenses = [l for l in licenses if "AGPL" in l.spdx_id]
        if agpl_licenses and "proprietary" in categories:
            findings.append(
                AuditFinding(
                    title="License: AGPL Compatibility Issue",
                    description=(
                        "AGPL license detected alongside proprietary code. "
                        "AGPL requires source disclosure for network services."
                    ),
                    severity=FindingSeverity.CRITICAL,
                    category=SecurityCategory.LICENSE.value,
                    audit_type=AuditType.COMPLIANCE,
                    document_id=agpl_licenses[0].location,
                    confidence=0.85,
                    evidence_text=f"AGPL in: {', '.join(l.location for l in agpl_licenses)}",
                    evidence_location="Cross-file analysis",
                    recommendation="Review AGPL obligations; consider removing AGPL dependencies or open-sourcing affected code",
                    found_by="software_auditor",
                )
            )

        return findings

    def _check_dependency_patterns(
        self,
        chunks: list[ChunkData],
    ) -> list[AuditFinding]:
        """Check for dependency-related issues across files."""
        findings = []

        # Look for dependency files
        for chunk in chunks:
            doc_id = chunk.document_id.lower()

            # Check package.json for known vulnerable patterns
            if "package.json" in doc_id:
                findings.extend(self._check_npm_dependencies(chunk))

            # Check requirements.txt
            elif "requirements" in doc_id and doc_id.endswith(".txt"):
                findings.extend(self._check_python_dependencies(chunk))

        return findings

    def _check_npm_dependencies(self, chunk: ChunkData) -> list[AuditFinding]:
        """Check npm dependencies for common issues."""
        findings = []
        text = chunk.content

        # Check for wildcard versions
        if re.search(r'"[^"]+"\s*:\s*"\*"', text):
            findings.append(
                AuditFinding(
                    title="Dependency: Wildcard Version",
                    description="Wildcard (*) version specified for dependency. This allows any version, including potentially vulnerable ones.",
                    severity=FindingSeverity.HIGH,
                    category=SecurityCategory.DEPENDENCY.value,
                    audit_type=AuditType.SECURITY,
                    document_id=chunk.document_id,
                    confidence=0.95,
                    evidence_text="Wildcard version found in package.json",
                    evidence_location=chunk.document_id,
                    recommendation="Pin dependencies to specific versions; use lockfiles",
                    found_by="software_auditor",
                )
            )

        # Check for git dependencies
        if re.search(r'"[^"]+"\s*:\s*"(?:git|github|http):', text):
            findings.append(
                AuditFinding(
                    title="Dependency: Git/URL Dependency",
                    description="Git or URL-based dependency detected. These bypass npm audit and may introduce supply chain risks.",
                    severity=FindingSeverity.MEDIUM,
                    category=SecurityCategory.DEPENDENCY.value,
                    audit_type=AuditType.SECURITY,
                    document_id=chunk.document_id,
                    confidence=0.85,
                    evidence_text="Git/URL dependency in package.json",
                    evidence_location=chunk.document_id,
                    recommendation="Prefer npm registry packages; pin to specific commit hashes if git is required",
                    found_by="software_auditor",
                )
            )

        return findings

    def _check_python_dependencies(self, chunk: ChunkData) -> list[AuditFinding]:
        """Check Python dependencies for common issues."""
        findings = []
        text = chunk.content

        # Check for unpinned versions
        unpinned = re.findall(r"^([a-zA-Z0-9_-]+)\s*$", text, re.MULTILINE)
        if unpinned:
            findings.append(
                AuditFinding(
                    title="Dependency: Unpinned Versions",
                    description=f"Unpinned dependencies detected: {', '.join(unpinned[:5])}{'...' if len(unpinned) > 5 else ''}",
                    severity=FindingSeverity.MEDIUM,
                    category=SecurityCategory.DEPENDENCY.value,
                    audit_type=AuditType.SECURITY,
                    document_id=chunk.document_id,
                    confidence=0.90,
                    evidence_text=f"Found {len(unpinned)} unpinned dependencies",
                    evidence_location=chunk.document_id,
                    recommendation="Pin dependencies to specific versions (e.g., package==1.2.3)",
                    found_by="software_auditor",
                )
            )

        # Check for known vulnerable patterns
        vulnerable_patterns = [
            (r"pyyaml\s*[<>=]*\s*[0-4]\.", "PyYAML < 5.1 has CVE-2017-18342 (code execution)"),
            (r"django\s*[<>=]*\s*[12]\.", "Django 1.x/2.x may have security vulnerabilities"),
            (r"requests\s*[<>=]*\s*2\.[0-9]\.", "Requests < 2.20 has CVE-2018-18074"),
        ]

        for pattern, desc in vulnerable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append(
                    AuditFinding(
                        title="Dependency: Potentially Vulnerable Version",
                        description=desc,
                        severity=FindingSeverity.HIGH,
                        category=SecurityCategory.DEPENDENCY.value,
                        audit_type=AuditType.SECURITY,
                        document_id=chunk.document_id,
                        confidence=0.75,
                        evidence_text=desc,
                        evidence_location=chunk.document_id,
                        recommendation="Update to latest secure version; run pip-audit or safety check",
                        found_by="software_auditor",
                    )
                )

        return findings


# Register with the audit registry on import
def register_software_auditor() -> None:
    """Register the software auditor with the global registry."""
    try:
        from ..registry import audit_registry

        audit_registry.register(SoftwareAuditor())
    except ImportError:
        pass  # Registry not available


__all__ = [
    "SoftwareAuditor",
    "SecurityCategory",
    "CWE",
    "VulnerabilityPattern",
    "SecretPattern",
    "LicenseInfo",
    "register_software_auditor",
]
