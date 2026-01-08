"""
GitHub Connector - Fetch issues, PRs, and discussions.

Uses GitHub CLI (gh) for authentication and API access.
Falls back to unauthenticated requests with rate limits.

Searches:
- Issues (open, closed, all)
- Pull Requests
- Discussions (if enabled)
- Code (via search API)
"""

import asyncio
import json
import logging
import re
import subprocess
from typing import Optional
from datetime import datetime
import hashlib

from aragora.reasoning.provenance import SourceType
from aragora.connectors.base import BaseConnector, Evidence

logger = logging.getLogger(__name__)

# Regex for valid GitHub repo format: owner/repo (alphanumeric, dash, underscore, dot)
VALID_REPO_PATTERN = re.compile(r'^[\w.-]+/[\w.-]+$')

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH = 500


class GitHubConnector(BaseConnector):
    """
    Connector for GitHub issues, PRs, and code search.

    Prefers GitHub CLI (gh) for authentication.
    Falls back to API with optional token.
    """

    def __init__(
        self,
        repo: Optional[str] = None,  # owner/repo format
        provenance=None,
        use_gh_cli: bool = True,
        token: Optional[str] = None,
    ):
        super().__init__(provenance=provenance, default_confidence=0.7)
        # Validate repo format to prevent command injection
        if repo and not VALID_REPO_PATTERN.match(repo):
            raise ValueError(f"Invalid repo format: {repo}. Expected 'owner/repo'")
        self.repo = repo
        self.use_gh_cli = use_gh_cli
        self.token = token
        self._gh_available: Optional[bool] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        return "GitHub"

    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available and authenticated."""
        if self._gh_available is not None:
            return self._gh_available

        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=30,  # Auth check should be fast
                shell=False,
            )
            self._gh_available = result.returncode == 0
        except Exception as e:
            logger.debug(f"[github] gh CLI check failed: {e}")
            self._gh_available = False

        return self._gh_available

    async def _run_gh(self, args: list[str]) -> Optional[str]:
        """Run gh CLI command."""
        if not self._check_gh_cli():
            return None

        try:
            proc = await asyncio.create_subprocess_exec(
                "gh", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,  # Reduced from 60s for faster debate flow
            )

            if proc.returncode == 0:
                return stdout.decode("utf-8")
            return None
        except Exception as e:
            logger.warning(f"[github] gh command failed (args={args[:2]}...): {e}")
            return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "issues",  # issues, prs, code
        state: str = "all",  # open, closed, all
        **kwargs,
    ) -> list[Evidence]:
        """
        Search GitHub for issues, PRs, or code.

        Args:
            query: Search query
            limit: Max results
            search_type: What to search (issues, prs, code)
            state: Issue/PR state filter

        Returns:
            List of Evidence objects
        """
        # Validate query to prevent abuse
        if not query or len(query) < 2:
            return []  # Query too short
        if len(query) > MAX_QUERY_LENGTH:
            return []  # Query too long, prevent DoS

        if not self.repo and search_type != "code":
            return []  # Need repo for issues/prs

        results = []

        if search_type == "issues":
            results = await self._search_issues(query, limit, state)
        elif search_type == "prs":
            results = await self._search_prs(query, limit, state)
        elif search_type == "code":
            results = await self._search_code(query, limit)

        return results

    async def _search_issues(
        self,
        query: str,
        limit: int,
        state: str,
    ) -> list[Evidence]:
        """Search issues via gh CLI."""
        args = [
            "issue", "list",
            "--repo", self.repo,
            "--search", query,
            "--limit", str(limit),
            "--json", "number,title,body,author,createdAt,url,state,labels",
        ]

        if state != "all":
            args.extend(["--state", state])

        output = await self._run_gh(args)
        if not output:
            return []

        try:
            issues = json.loads(output)
        except json.JSONDecodeError:
            return []

        results = []
        for issue in issues:
            # Calculate authority based on labels
            labels = [l.get("name", "") for l in issue.get("labels", [])]
            authority = 0.6
            if "bug" in labels or "critical" in labels:
                authority = 0.8
            if "documentation" in labels:
                authority = 0.7

            evidence = Evidence(
                id=f"gh-issue:{self.repo}:{issue['number']}",
                source_type=self.source_type,
                source_id=f"github/{self.repo}/issues/{issue['number']}",
                content=f"# {issue['title']}\n\n{issue.get('body', '')[:2000]}",
                title=f"Issue #{issue['number']}: {issue['title']}",
                url=issue.get("url", ""),
                author=issue.get("author", {}).get("login", "unknown"),
                created_at=issue.get("createdAt", ""),
                confidence=0.7,
                freshness=self.calculate_freshness(issue.get("createdAt", "")),
                authority=authority,
                metadata={
                    "number": issue["number"],
                    "state": issue.get("state", ""),
                    "labels": labels,
                    "type": "issue",
                },
            )
            results.append(evidence)

        return results

    async def _search_prs(
        self,
        query: str,
        limit: int,
        state: str,
    ) -> list[Evidence]:
        """Search PRs via gh CLI."""
        args = [
            "pr", "list",
            "--repo", self.repo,
            "--search", query,
            "--limit", str(limit),
            "--json", "number,title,body,author,createdAt,url,state,mergedAt",
        ]

        if state != "all":
            args.extend(["--state", state])

        output = await self._run_gh(args)
        if not output:
            return []

        try:
            prs = json.loads(output)
        except json.JSONDecodeError:
            return []

        results = []
        for pr in prs:
            # Merged PRs have higher authority
            authority = 0.8 if pr.get("mergedAt") else 0.6

            evidence = Evidence(
                id=f"gh-pr:{self.repo}:{pr['number']}",
                source_type=self.source_type,
                source_id=f"github/{self.repo}/pull/{pr['number']}",
                content=f"# {pr['title']}\n\n{pr.get('body', '')[:2000]}",
                title=f"PR #{pr['number']}: {pr['title']}",
                url=pr.get("url", ""),
                author=pr.get("author", {}).get("login", "unknown"),
                created_at=pr.get("createdAt", ""),
                confidence=0.75,
                freshness=self.calculate_freshness(pr.get("createdAt", "")),
                authority=authority,
                metadata={
                    "number": pr["number"],
                    "state": pr.get("state", ""),
                    "merged": pr.get("mergedAt") is not None,
                    "type": "pr",
                },
            )
            results.append(evidence)

        return results

    async def _search_code(
        self,
        query: str,
        limit: int,
    ) -> list[Evidence]:
        """Search code via gh CLI."""
        search_query = query
        if self.repo:
            search_query = f"repo:{self.repo} {query}"

        args = [
            "search", "code",
            search_query,
            "--limit", str(limit),
            "--json", "path,repository,textMatches",
        ]

        output = await self._run_gh(args)
        if not output:
            return []

        try:
            code_results = json.loads(output)
        except json.JSONDecodeError:
            return []

        results = []
        for result in code_results:
            repo_name = result.get("repository", {}).get("fullName", "unknown")
            path = result.get("path", "unknown")

            # Extract text matches
            matches = result.get("textMatches", [])
            content = "\n---\n".join(
                m.get("fragment", "")
                for m in matches[:3]
            )

            evidence = Evidence(
                id=f"gh-code:{hashlib.sha256(f'{repo_name}/{path}'.encode()).hexdigest()[:12]}",
                source_type=SourceType.CODE_ANALYSIS,
                source_id=f"github/{repo_name}/{path}",
                content=content or f"Code match in {path}",
                title=f"{repo_name}: {path}",
                url=f"https://github.com/{repo_name}/blob/main/{path}",
                confidence=0.8,
                freshness=0.7,  # Code freshness hard to determine
                authority=0.7,
                metadata={
                    "repository": repo_name,
                    "path": path,
                    "match_count": len(matches),
                    "type": "code",
                },
            )
            results.append(evidence)

        return results

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Fetch specific issue/PR by ID."""
        if evidence_id in self._cache:
            return self._cache[evidence_id]

        # Parse evidence_id
        if evidence_id.startswith("gh-issue:"):
            parts = evidence_id.split(":")
            if len(parts) >= 3:
                repo = parts[1]
                number = parts[2]
                return await self._fetch_issue(repo, number)

        elif evidence_id.startswith("gh-pr:"):
            parts = evidence_id.split(":")
            if len(parts) >= 3:
                repo = parts[1]
                number = parts[2]
                return await self._fetch_pr(repo, number)

        return None

    async def _fetch_issue(self, repo: str, number: str) -> Optional[Evidence]:
        """Fetch single issue."""
        args = [
            "issue", "view", number,
            "--repo", repo,
            "--json", "number,title,body,author,createdAt,url,state,labels,comments",
        ]

        output = await self._run_gh(args)
        if not output:
            return None

        try:
            issue = json.loads(output)
        except json.JSONDecodeError:
            return None

        # Include comments in content
        comments = issue.get("comments", [])
        comments_text = "\n\n---\n\n".join(
            f"**{c.get('author', {}).get('login', 'unknown')}**: {c.get('body', '')[:500]}"
            for c in comments[:5]
        )

        content = f"# {issue['title']}\n\n{issue.get('body', '')}\n\n## Comments\n\n{comments_text}"

        evidence = Evidence(
            id=f"gh-issue:{repo}:{number}",
            source_type=self.source_type,
            source_id=f"github/{repo}/issues/{number}",
            content=content[:5000],
            title=f"Issue #{number}: {issue['title']}",
            url=issue.get("url", ""),
            author=issue.get("author", {}).get("login", "unknown"),
            created_at=issue.get("createdAt", ""),
            confidence=0.75,
            freshness=self.calculate_freshness(issue.get("createdAt", "")),
            authority=0.7,
            metadata={
                "number": int(number),
                "state": issue.get("state", ""),
                "comment_count": len(comments),
                "type": "issue",
            },
        )

        self._cache_put(evidence.id, evidence)
        return evidence

    async def _fetch_pr(self, repo: str, number: str) -> Optional[Evidence]:
        """Fetch single PR."""
        args = [
            "pr", "view", number,
            "--repo", repo,
            "--json", "number,title,body,author,createdAt,url,state,mergedAt,reviews",
        ]

        output = await self._run_gh(args)
        if not output:
            return None

        try:
            pr = json.loads(output)
        except json.JSONDecodeError:
            return None

        # Include reviews in content
        reviews = pr.get("reviews", [])
        reviews_text = "\n\n---\n\n".join(
            f"**{r.get('author', {}).get('login', 'unknown')}** ({r.get('state', '')}): {r.get('body', '')[:300]}"
            for r in reviews[:5]
        )

        content = f"# {pr['title']}\n\n{pr.get('body', '')}\n\n## Reviews\n\n{reviews_text}"

        evidence = Evidence(
            id=f"gh-pr:{repo}:{number}",
            source_type=self.source_type,
            source_id=f"github/{repo}/pull/{number}",
            content=content[:5000],
            title=f"PR #{number}: {pr['title']}",
            url=pr.get("url", ""),
            author=pr.get("author", {}).get("login", "unknown"),
            created_at=pr.get("createdAt", ""),
            confidence=0.8,
            freshness=self.calculate_freshness(pr.get("createdAt", "")),
            authority=0.85 if pr.get("mergedAt") else 0.7,
            metadata={
                "number": int(number),
                "state": pr.get("state", ""),
                "merged": pr.get("mergedAt") is not None,
                "review_count": len(reviews),
                "type": "pr",
            },
        )

        self._cache_put(evidence.id, evidence)
        return evidence
