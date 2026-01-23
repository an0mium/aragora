"""
Secrets Scanner for codebase security analysis.

Detects hardcoded credentials, API keys, tokens, and other sensitive data
using pattern matching and entropy analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from .models import SecretFinding, SecretsScanResult, SecretType, VulnerabilitySeverity


@dataclass
class SecretPattern:
    """A pattern for detecting a specific type of secret."""

    secret_type: SecretType
    pattern: Pattern[str]
    severity: VulnerabilitySeverity
    confidence: float = 0.9
    description: str = ""
    remediation: str = ""


# Compiled regex patterns for various secret types
SECRET_PATTERNS: List[SecretPattern] = [
    # AWS
    SecretPattern(
        secret_type=SecretType.AWS_ACCESS_KEY,
        pattern=re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="AWS Access Key ID",
        remediation="Rotate the AWS access key immediately via IAM console",
    ),
    SecretPattern(
        secret_type=SecretType.AWS_SECRET_KEY,
        pattern=re.compile(
            r"(?i)(?:aws[_-]?secret[_-]?(?:access[_-]?)?key|aws[_-]?secret)\s*[:=]\s*['\"]?"
            r"([A-Za-z0-9/+=]{40})['\"]?"
        ),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.85,
        description="AWS Secret Access Key",
        remediation="Rotate the AWS secret key immediately and update all services",
    ),
    # GitHub
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        pattern=re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,255}"),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="GitHub Personal Access Token or OAuth Token",
        remediation="Revoke token at github.com/settings/tokens and generate new one",
    ),
    SecretPattern(
        secret_type=SecretType.GITHUB_PAT,
        pattern=re.compile(r"github_pat_[A-Za-z0-9_]{22,255}"),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="GitHub Fine-grained Personal Access Token",
        remediation="Revoke token at github.com/settings/tokens and generate new one",
    ),
    # GitLab
    SecretPattern(
        secret_type=SecretType.GITLAB_TOKEN,
        pattern=re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="GitLab Personal Access Token",
        remediation="Revoke token in GitLab settings and generate new one",
    ),
    # Slack
    SecretPattern(
        secret_type=SecretType.SLACK_TOKEN,
        pattern=re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.95,
        description="Slack Bot/User/App Token",
        remediation="Regenerate token in Slack app settings",
    ),
    SecretPattern(
        secret_type=SecretType.SLACK_WEBHOOK,
        pattern=re.compile(
            r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24}"
        ),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.95,
        description="Slack Webhook URL",
        remediation="Delete webhook and create new one in Slack app settings",
    ),
    # Discord
    SecretPattern(
        secret_type=SecretType.DISCORD_TOKEN,
        pattern=re.compile(r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.85,
        description="Discord Bot Token",
        remediation="Regenerate token in Discord Developer Portal",
    ),
    SecretPattern(
        secret_type=SecretType.DISCORD_WEBHOOK,
        pattern=re.compile(
            r"https://discord(?:app)?\.com/api/webhooks/\d{17,19}/[A-Za-z0-9_-]{60,68}"
        ),
        severity=VulnerabilitySeverity.MEDIUM,
        confidence=0.95,
        description="Discord Webhook URL",
        remediation="Delete webhook and create new one in Discord server settings",
    ),
    # Stripe
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        pattern=re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{24,}"),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="Stripe Secret Key",
        remediation="Roll keys in Stripe Dashboard > Developers > API keys",
    ),
    # Twilio
    SecretPattern(
        secret_type=SecretType.TWILIO_KEY,
        pattern=re.compile(r"SK[a-f0-9]{32}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.85,
        description="Twilio API Key",
        remediation="Delete and recreate API key in Twilio Console",
    ),
    # SendGrid
    SecretPattern(
        secret_type=SecretType.SENDGRID_KEY,
        pattern=re.compile(r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.95,
        description="SendGrid API Key",
        remediation="Delete and recreate API key in SendGrid Settings",
    ),
    # Mailgun
    SecretPattern(
        secret_type=SecretType.MAILGUN_KEY,
        pattern=re.compile(r"key-[A-Za-z0-9]{32}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.85,
        description="Mailgun API Key",
        remediation="Reset API key in Mailgun Dashboard",
    ),
    # JWT
    SecretPattern(
        secret_type=SecretType.JWT_TOKEN,
        pattern=re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        severity=VulnerabilitySeverity.MEDIUM,
        confidence=0.90,
        description="JSON Web Token",
        remediation="Rotate signing key and invalidate existing tokens",
    ),
    # Private Keys
    SecretPattern(
        secret_type=SecretType.PRIVATE_KEY,
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?:\sBLOCK)?-----"
        ),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.95,
        description="Private Key",
        remediation="Generate new key pair and revoke/replace the exposed key",
    ),
    # Google
    SecretPattern(
        secret_type=SecretType.GOOGLE_API_KEY,
        pattern=re.compile(r"AIza[0-9A-Za-z_-]{35}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.90,
        description="Google API Key",
        remediation="Regenerate key in Google Cloud Console",
    ),
    # Azure
    SecretPattern(
        secret_type=SecretType.AZURE_KEY,
        pattern=re.compile(
            r"(?i)(?:azure[_-]?(?:storage[_-]?)?(?:account[_-]?)?key|"
            r"DefaultEndpointsProtocol=https;AccountName=)[^;'\"\s]{20,}"
        ),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.80,
        description="Azure Storage Key or Connection String",
        remediation="Rotate key in Azure Portal > Storage Account > Access Keys",
    ),
    # OpenAI
    SecretPattern(
        secret_type=SecretType.OPENAI_KEY,
        pattern=re.compile(r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.95,
        description="OpenAI API Key",
        remediation="Delete and create new key at platform.openai.com/api-keys",
    ),
    # Anthropic
    SecretPattern(
        secret_type=SecretType.ANTHROPIC_KEY,
        pattern=re.compile(r"sk-ant-api\d{2}-[A-Za-z0-9_-]{93}"),
        severity=VulnerabilitySeverity.HIGH,
        confidence=0.95,
        description="Anthropic API Key",
        remediation="Delete and create new key in Anthropic Console",
    ),
    # Database URLs
    SecretPattern(
        secret_type=SecretType.DATABASE_URL,
        pattern=re.compile(
            r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://" r"[^:]+:[^@]+@[^/\s]+"
        ),
        severity=VulnerabilitySeverity.CRITICAL,
        confidence=0.90,
        description="Database Connection String with Credentials",
        remediation="Rotate database password and update connection string",
    ),
    # Generic API keys (lower confidence)
    SecretPattern(
        secret_type=SecretType.GENERIC_API_KEY,
        pattern=re.compile(
            r"(?i)(?:api[_-]?key|apikey|api[_-]?secret|auth[_-]?token|"
            r"access[_-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_-]{20,})['\"]?"
        ),
        severity=VulnerabilitySeverity.MEDIUM,
        confidence=0.70,
        description="Generic API Key or Token",
        remediation="Rotate the credential with the appropriate service provider",
    ),
    # Generic secrets (even lower confidence)
    SecretPattern(
        secret_type=SecretType.GENERIC_SECRET,
        pattern=re.compile(
            r"(?i)(?:password|passwd|pwd|secret|credential)\s*[:=]\s*['\"]?" r"([^'\"\s]{8,})['\"]?"
        ),
        severity=VulnerabilitySeverity.MEDIUM,
        confidence=0.60,
        description="Generic Password or Secret",
        remediation="Rotate the credential and use environment variables",
    ),
]

# File extensions to skip
SKIP_EXTENSIONS: Set[str] = {
    # Binary files
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".o",
    ".a",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    # Audio/Video
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # Fonts
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
    # Other
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".war",
}

# Directories to skip
SKIP_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    ".vs",
    "dist",
    "build",
    "target",
    ".next",
    "vendor",
    "bower_components",
    ".terraform",
    ".serverless",
}

# Files to skip (exact match)
SKIP_FILES: Set[str] = {
    "package-lock.json",
    "yarn.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
}


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0

    entropy = 0.0
    length = len(text)
    char_counts: Dict[str, int] = {}

    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    for count in char_counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def is_high_entropy(text: str, threshold: float = 4.5) -> bool:
    """Check if a string has suspiciously high entropy."""
    if len(text) < 16:
        return False

    entropy = calculate_entropy(text)
    return entropy >= threshold


class SecretsScanner:
    """
    Scanner for detecting secrets and credentials in codebases.

    Features:
    - Pattern-based detection for known secret types
    - Entropy-based detection for unknown high-entropy strings
    - Git history scanning
    - Concurrent async file scanning
    - Configurable file/directory filtering
    """

    def __init__(
        self,
        patterns: Optional[List[SecretPattern]] = None,
        skip_extensions: Optional[Set[str]] = None,
        skip_dirs: Optional[Set[str]] = None,
        skip_files: Optional[Set[str]] = None,
        max_concurrency: int = 20,
        max_file_size_mb: float = 10.0,
        enable_entropy_detection: bool = True,
        entropy_threshold: float = 4.5,
        min_entropy_length: int = 20,
    ):
        self.patterns = patterns or SECRET_PATTERNS
        self.skip_extensions = skip_extensions or SKIP_EXTENSIONS
        self.skip_dirs = skip_dirs or SKIP_DIRS
        self.skip_files = skip_files or SKIP_FILES
        self.max_concurrency = max_concurrency
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.enable_entropy_detection = enable_entropy_detection
        self.entropy_threshold = entropy_threshold
        self.min_entropy_length = min_entropy_length
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def scan_repository(
        self,
        repo_path: str,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> SecretsScanResult:
        """
        Scan a repository for secrets.

        Args:
            repo_path: Path to the repository root
            branch: Optional branch name for metadata
            commit_sha: Optional commit SHA for metadata

        Returns:
            SecretsScanResult with all detected secrets
        """
        scan_id = str(uuid.uuid4())
        repo_name = os.path.basename(os.path.abspath(repo_path))

        result = SecretsScanResult(
            scan_id=scan_id,
            repository=repo_name,
            branch=branch,
            commit_sha=commit_sha,
            started_at=datetime.now(),
        )

        try:
            files_to_scan = self._collect_files(repo_path)
            result.files_scanned = len(files_to_scan)

            self._semaphore = asyncio.Semaphore(self.max_concurrency)

            tasks = [self._scan_file(file_path, repo_path) for file_path in files_to_scan]
            findings_lists = await asyncio.gather(*tasks, return_exceptions=True)

            for findings in findings_lists:
                if isinstance(findings, Exception):
                    continue
                result.secrets.extend(findings)

            result.status = "completed"
            result.completed_at = datetime.now()

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()

        return result

    async def scan_git_history(
        self,
        repo_path: str,
        depth: int = 100,
        branch: Optional[str] = None,
    ) -> SecretsScanResult:
        """
        Scan git history for secrets in past commits.

        Args:
            repo_path: Path to the repository root
            depth: Number of commits to scan back
            branch: Optional branch to scan (default: current)

        Returns:
            SecretsScanResult with secrets found in history
        """
        scan_id = str(uuid.uuid4())
        repo_name = os.path.basename(os.path.abspath(repo_path))

        result = SecretsScanResult(
            scan_id=scan_id,
            repository=repo_name,
            branch=branch,
            started_at=datetime.now(),
            scanned_history=True,
            history_depth=depth,
        )

        try:
            commits = await self._get_commit_list(repo_path, depth, branch)

            for commit_info in commits:
                commit_sha = commit_info["sha"]
                diff_content = await self._get_commit_diff(repo_path, commit_sha)

                findings = self._scan_diff_content(
                    diff_content,
                    commit_sha,
                    commit_info.get("author"),
                    commit_info.get("date"),
                )

                for finding in findings:
                    finding.is_in_history = True
                result.secrets.extend(findings)

            result.files_scanned = len(commits)
            result.status = "completed"
            result.completed_at = datetime.now()

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()

        return result

    def _collect_files(self, repo_path: str) -> List[str]:
        """Collect all files to scan, respecting filters."""
        files_to_scan = []
        repo_path = os.path.abspath(repo_path)

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]

            for filename in files:
                if filename in self.skip_files:
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext in self.skip_extensions:
                    continue

                file_path = os.path.join(root, filename)

                try:
                    if os.path.getsize(file_path) > self.max_file_size_bytes:
                        continue
                except OSError:
                    continue

                files_to_scan.append(file_path)

        return files_to_scan

    async def _scan_file(self, file_path: str, repo_path: str) -> List[SecretFinding]:
        """Scan a single file for secrets."""
        assert self._semaphore is not None

        async with self._semaphore:
            return await asyncio.to_thread(self._scan_file_sync, file_path, repo_path)

    def _scan_file_sync(self, file_path: str, repo_path: str) -> List[SecretFinding]:
        """Synchronous file scanning implementation."""
        findings: List[SecretFinding] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, IOError):
            return findings

        relative_path = os.path.relpath(file_path, repo_path)
        lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            line_findings = self._scan_line(line, relative_path, line_num)
            findings.extend(line_findings)

        return findings

    def _scan_line(
        self,
        line: str,
        file_path: str,
        line_number: int,
    ) -> List[SecretFinding]:
        """Scan a single line for secrets."""
        findings: List[SecretFinding] = []
        found_positions: Set[Tuple[int, int]] = set()

        for pattern in self.patterns:
            for match in pattern.pattern.finditer(line):
                start, end = match.start(), match.end()

                if (start, end) in found_positions:
                    continue
                found_positions.add((start, end))

                matched_text = match.group(0)
                if match.groups():
                    matched_text = match.group(1)

                finding = SecretFinding(
                    id=self._generate_finding_id(file_path, line_number, start),
                    secret_type=pattern.secret_type,
                    file_path=file_path,
                    line_number=line_number,
                    column_start=start,
                    column_end=end,
                    matched_text=SecretFinding.redact_secret(matched_text),
                    context_line=self._redact_line(line, start, end),
                    severity=pattern.severity,
                    confidence=pattern.confidence,
                    remediation=pattern.remediation,
                )
                findings.append(finding)

        if self.enable_entropy_detection:
            entropy_findings = self._scan_for_high_entropy(
                line, file_path, line_number, found_positions
            )
            findings.extend(entropy_findings)

        return findings

    def _scan_for_high_entropy(
        self,
        line: str,
        file_path: str,
        line_number: int,
        exclude_positions: Set[Tuple[int, int]],
    ) -> List[SecretFinding]:
        """Detect high-entropy strings that might be secrets."""
        findings: List[SecretFinding] = []

        token_pattern = re.compile(r"[A-Za-z0-9_/+=\-]{20,}")

        for match in token_pattern.finditer(line):
            start, end = match.start(), match.end()

            overlaps = any(
                start < ex_end and end > ex_start for ex_start, ex_end in exclude_positions
            )
            if overlaps:
                continue

            token = match.group(0)
            entropy = calculate_entropy(token)

            if entropy >= self.entropy_threshold:
                finding = SecretFinding(
                    id=self._generate_finding_id(file_path, line_number, start),
                    secret_type=SecretType.HIGH_ENTROPY,
                    file_path=file_path,
                    line_number=line_number,
                    column_start=start,
                    column_end=end,
                    matched_text=SecretFinding.redact_secret(token),
                    context_line=self._redact_line(line, start, end),
                    severity=VulnerabilitySeverity.LOW,
                    confidence=min(0.5 + (entropy - self.entropy_threshold) * 0.1, 0.8),
                    entropy=entropy,
                    remediation="Review if this is a secret and use environment variables",
                )
                findings.append(finding)

        return findings

    def _scan_diff_content(
        self,
        diff_content: str,
        commit_sha: str,
        author: Optional[str],
        commit_date: Optional[datetime],
    ) -> List[SecretFinding]:
        """Scan git diff content for secrets."""
        findings: List[SecretFinding] = []
        current_file: Optional[str] = None
        line_number = 0

        for line in diff_content.split("\n"):
            if line.startswith("diff --git"):
                match = re.search(r"b/(.+)$", line)
                if match:
                    current_file = match.group(1)
                line_number = 0
                continue

            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    line_number = int(match.group(1)) - 1
                continue

            if line.startswith("+") and not line.startswith("+++"):
                line_number += 1
                content = line[1:]

                if current_file:
                    line_findings = self._scan_line(content, current_file, line_number)
                    for finding in line_findings:
                        finding.commit_sha = commit_sha
                        finding.commit_author = author
                        finding.commit_date = commit_date
                    findings.extend(line_findings)
            elif not line.startswith("-"):
                line_number += 1

        return findings

    async def _get_commit_list(
        self,
        repo_path: str,
        depth: int,
        branch: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Get list of commits to scan."""
        cmd = ["git", "log", f"-{depth}", "--format=%H|%an|%aI"]
        if branch:
            cmd.append(branch)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        commits = []
        for line in stdout.decode().strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                commits.append(
                    {
                        "sha": parts[0],
                        "author": parts[1],
                        "date": datetime.fromisoformat(parts[2]),
                    }
                )

        return commits

    async def _get_commit_diff(
        self,
        repo_path: str,
        commit_sha: str,
    ) -> str:
        """Get diff for a specific commit."""
        cmd = ["git", "show", commit_sha, "--format=", "--diff-filter=AM"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        return stdout.decode(errors="ignore")

    def _generate_finding_id(self, file_path: str, line_number: int, column: int) -> str:
        """Generate a unique ID for a finding."""
        content = f"{file_path}:{line_number}:{column}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _redact_line(self, line: str, start: int, end: int) -> str:
        """Redact the secret portion of a line."""
        secret = line[start:end]
        redacted = SecretFinding.redact_secret(secret)
        return line[:start] + redacted + line[end:]


async def scan_repository_for_secrets(
    repo_path: str,
    include_history: bool = False,
    history_depth: int = 100,
    branch: Optional[str] = None,
) -> SecretsScanResult:
    """
    Convenience function to scan a repository for secrets.

    Args:
        repo_path: Path to the repository
        include_history: Whether to also scan git history
        history_depth: Number of commits to scan if include_history=True
        branch: Optional branch name

    Returns:
        SecretsScanResult with all findings
    """
    scanner = SecretsScanner()

    result = await scanner.scan_repository(repo_path, branch=branch)

    if include_history:
        history_result = await scanner.scan_git_history(
            repo_path, depth=history_depth, branch=branch
        )
        result.secrets.extend(history_result.secrets)
        result.scanned_history = True
        result.history_depth = history_depth

    return result
