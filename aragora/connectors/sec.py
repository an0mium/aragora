"""
SEC EDGAR Connector - Financial filings search for aragora agents.

Provides access to the SEC's EDGAR database for:
- Company filings (10-K, 10-Q, 8-K, etc.)
- Form search by company or CIK
- Full-text search across filings

The SEC EDGAR API is free and requires no authentication.
Rate limit: 10 requests per second.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import ProvenanceManager, SourceType

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# SEC EDGAR API endpoints
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"

# Common form types
FORM_TYPES = {
    "10-K": "Annual report",
    "10-Q": "Quarterly report",
    "8-K": "Current report (material events)",
    "S-1": "Registration statement (IPO)",
    "DEF 14A": "Proxy statement",
    "4": "Statement of changes in beneficial ownership",
    "13F": "Institutional investment manager holdings",
    "SC 13D": "Beneficial ownership report (>5%)",
    "SC 13G": "Beneficial ownership report (passive)",
    "6-K": "Foreign private issuer report",
    "20-F": "Foreign private issuer annual report",
}


class SECConnector(BaseConnector):
    """
    Connector for SEC EDGAR financial filings database.

    Enables agents to:
    - Search company filings by ticker or CIK
    - Get specific form types (10-K, 10-Q, 8-K, etc.)
    - Full-text search across filings
    - Track regulatory source provenance

    No authentication required.

    Example:
        connector = SECConnector()
        results = await connector.search("AAPL")
        for evidence in results:
            print(f"{evidence.title} - {evidence.metadata['form_type']}")
    """

    # CIK lookup cache (ticker -> CIK)
    _cik_cache: dict[str, str] = {}

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.90,  # Regulatory filings are highly reliable
        timeout: int = 30,
        rate_limit_delay: float = 0.1,  # SEC allows 10 req/sec
        max_cache_entries: int = 500,
        cache_ttl_seconds: float = 86400.0,  # 24 hour cache (filings don't change)
    ):
        """
        Initialize SECConnector.

        Args:
            provenance: Optional provenance manager for tracking
            default_confidence: Base confidence (high for regulatory data)
            timeout: HTTP request timeout in seconds
            rate_limit_delay: Delay between API requests
            max_cache_entries: Maximum cached entries
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(
            provenance=provenance,
            default_confidence=default_confidence,
            max_cache_entries=max_cache_entries,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    def source_type(self) -> SourceType:
        """SEC filings are official documents."""
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        """Human-readable connector name."""
        return "SEC EDGAR"

    @property
    def is_available(self) -> bool:
        """Check if httpx is available for making requests."""
        return HTTPX_AVAILABLE

    def _get_headers(self) -> dict:
        """Get headers for SEC requests (required User-Agent)."""
        return {
            "User-Agent": "Aragora/1.0 (contact@aragora.ai)",
            "Accept": "application/json",
        }

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def search(
        self,
        query: str,
        limit: int = 10,
        form_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search SEC EDGAR filings.

        Args:
            query: Company ticker, CIK, or company name
            limit: Maximum results to return
            form_type: Filter by form type (e.g., "10-K", "10-Q", "8-K")
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            **kwargs: Additional parameters

        Returns:
            List of Evidence objects with filing information
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, cannot search SEC EDGAR")
            return []

        # Try to interpret query as ticker or CIK
        cik = await self._resolve_to_cik(query)
        if not cik:
            logger.info(f"Could not resolve '{query}' to CIK, trying full-text search")
            return await self._fulltext_search(query, limit, form_type)

        # Fetch company submissions
        return await self._get_company_filings(cik, limit, form_type, date_from, date_to)

    async def _resolve_to_cik(self, query: str) -> Optional[str]:
        """
        Resolve ticker or company name to CIK number.

        Args:
            query: Ticker symbol, CIK, or company name

        Returns:
            10-digit CIK string or None
        """
        query = query.strip().upper()

        # Check if already a CIK
        if query.isdigit():
            return query.zfill(10)

        # Check cache
        if query in self._cik_cache:
            return self._cik_cache[query]

        # Try to fetch CIK from SEC company tickers JSON
        await self._rate_limit()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # SEC provides a JSON file mapping tickers to CIKs
                response = await client.get(
                    "https://www.sec.gov/files/company_tickers.json",
                    headers=self._get_headers(),
                )
                response.raise_for_status()

            data = response.json()

            # Search for ticker match
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                if ticker == query:
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    self._cik_cache[query] = cik
                    return cik

            return None

        except Exception as e:
            logger.debug(f"CIK lookup failed for {query}: {e}")
            return None

    async def _get_company_filings(
        self,
        cik: str,
        limit: int,
        form_type: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[Evidence]:
        """Fetch filings for a specific CIK."""
        await self._rate_limit()

        try:
            url = EDGAR_SUBMISSIONS_URL.format(cik=cik)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 404:
                    logger.debug(f"No filings found for CIK {cik}")
                    return []

                response.raise_for_status()

            data = response.json()
            return self._parse_submissions(data, limit, form_type, date_from, date_to)

        except httpx.TimeoutException:
            logger.warning(f"SEC EDGAR timeout for CIK {cik}")
            return []
        except Exception as e:
            logger.error(f"SEC EDGAR fetch failed for CIK {cik}: {e}")
            return []

    async def _fulltext_search(
        self,
        query: str,
        limit: int,
        form_type: Optional[str],
    ) -> list[Evidence]:
        """
        Full-text search across SEC filings.

        Note: SEC's full-text search API has specific requirements.
        """
        await self._rate_limit()

        try:
            params = {
                "q": query,
                "dateRange": "custom",
                "startdt": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "enddt": datetime.now().strftime("%Y-%m-%d"),
            }

            if form_type:
                params["forms"] = form_type

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    EDGAR_FULLTEXT_URL,
                    params=params,
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    logger.debug(f"SEC full-text search returned {response.status_code}")
                    return []

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            results = []
            for hit in hits[:limit]:
                evidence = self._parse_fulltext_hit(hit)
                if evidence:
                    results.append(evidence)

            return results

        except Exception as e:
            logger.debug(f"SEC full-text search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific filing.

        Args:
            evidence_id: SEC accession number or evidence ID

        Returns:
            Evidence object or None
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        # SEC filings require specific accession number format
        # For now, return None (would need filing URL to fetch)
        return None

    async def get_recent_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 5,
    ) -> list[Evidence]:
        """
        Get recent filings of a specific type for a company.

        Args:
            ticker: Company ticker symbol
            form_type: Form type (10-K, 10-Q, 8-K, etc.)
            limit: Maximum results

        Returns:
            List of Evidence objects
        """
        return await self.search(ticker, limit=limit, form_type=form_type)

    async def get_8k_filings(
        self,
        ticker: str,
        days: int = 30,
        limit: int = 10,
    ) -> list[Evidence]:
        """
        Get recent 8-K (material event) filings.

        Args:
            ticker: Company ticker symbol
            days: Look back period in days
            limit: Maximum results

        Returns:
            List of Evidence objects for 8-K filings
        """
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return await self.search(
            ticker,
            limit=limit,
            form_type="8-K",
            date_from=date_from,
        )

    def _parse_submissions(
        self,
        data: dict,
        limit: int,
        form_type: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[Evidence]:
        """Parse SEC submissions JSON into Evidence objects."""
        results = []

        company_name = data.get("name", "Unknown Company")
        cik = data.get("cik", "")
        tickers = data.get("tickers", [])
        ticker = tickers[0] if tickers else ""

        # Get recent filings
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        for i in range(min(len(forms), limit * 3)):  # Fetch more to filter
            if len(results) >= limit:
                break

            form = forms[i] if i < len(forms) else ""
            filing_date = dates[i] if i < len(dates) else ""
            accession = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            description = descriptions[i] if i < len(descriptions) else ""

            # Filter by form type
            if form_type and form != form_type:
                continue

            # Filter by date range
            if date_from and filing_date < date_from:
                continue
            if date_to and filing_date > date_to:
                continue

            evidence = self._create_filing_evidence(
                company_name=company_name,
                ticker=ticker,
                cik=cik,
                form=form,
                filing_date=filing_date,
                accession=accession,
                primary_doc=primary_doc,
                description=description,
            )

            if evidence:
                results.append(evidence)
                self._cache_put(evidence.id, evidence)

        return results

    def _create_filing_evidence(
        self,
        company_name: str,
        ticker: str,
        cik: str,
        form: str,
        filing_date: str,
        accession: str,
        primary_doc: str,
        description: str,
    ) -> Optional[Evidence]:
        """Create Evidence object for a filing."""
        if not form or not filing_date:
            return None

        # Build filing URL
        accession_clean = accession.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_doc}"
        )

        # Build evidence ID
        evidence_id = f"sec:{accession}"

        # Create content summary
        form_desc = FORM_TYPES.get(form, form)
        content = (
            f"{company_name} ({ticker}) filed {form} ({form_desc}) on {filing_date}. "
            f"{description}"
        ).strip()

        # Title
        title = f"{ticker or company_name}: {form} - {filing_date}"

        # Freshness based on filing date
        freshness = self.calculate_freshness(f"{filing_date}T00:00:00Z")

        return Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=accession,
            content=content,
            title=title,
            created_at=f"{filing_date}T00:00:00Z",
            author=company_name,
            url=filing_url,
            confidence=self.default_confidence,
            freshness=freshness,
            authority=0.95,  # Regulatory filings have very high authority
            metadata={
                "company_name": company_name,
                "ticker": ticker,
                "cik": cik,
                "form_type": form,
                "form_description": form_desc,
                "accession_number": accession,
                "primary_document": primary_doc,
                "connector": "sec",
            },
        )

    def _parse_fulltext_hit(self, hit: dict) -> Optional[Evidence]:
        """Parse a full-text search hit into Evidence."""
        source = hit.get("_source", {})
        if not source:
            return None

        company = source.get("display_names", ["Unknown"])[0]
        form = source.get("form", "")
        filing_date = source.get("file_date", "")
        accession = source.get("adsh", "")

        return self._create_filing_evidence(
            company_name=company,
            ticker="",
            cik=source.get("ciks", [""])[0],
            form=form,
            filing_date=filing_date,
            accession=accession,
            primary_doc="",
            description=source.get("_id", ""),
        )

    def get_form_types(self) -> dict[str, str]:
        """Return available SEC form types and descriptions."""
        return FORM_TYPES.copy()


__all__ = ["SECConnector", "FORM_TYPES"]
