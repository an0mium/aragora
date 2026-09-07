"""
Evidence Connectors - Ground debates in real data.

Connectors fetch evidence from external sources and integrate
with the provenance system for traceability:

- LocalDocsConnector: Search local documentation, markdown, code
- GitHubConnector: Fetch issues, PRs, discussions
- WebConnector: Search and fetch live web content
- ArXivConnector: Academic papers and preprints
- HackerNewsConnector: Tech community discussions
- WikipediaConnector: Encyclopedia articles and reference knowledge
- RedditConnector: Community discussions and sentiment
- TwitterConnector: Public discourse and real-time updates
- SQLConnector: Query SQL databases (PostgreSQL, MySQL, SQLite)
- NewsAPIConnector: News articles from multiple sources
- SECConnector: SEC EDGAR financial filings
- ConversationIngestorConnector: Parse ChatGPT/Claude exports for claim extraction

All connectors record evidence through ProvenanceManager
with proper source typing and confidence scoring.

Imports are lazy (PEP 562 ``__getattr__``): importing
``aragora.connectors`` (or any lightweight submodule such as
``aragora.connectors.base``) does NOT eagerly pull in every connector's
optional third-party dependencies. A name is only resolved — and its
backing submodule imported — on first attribute access. This keeps the
base install able to run the debate engine (which only needs
``Evidence`` / ``BaseConnector``) without requiring optional deps like
``defusedxml`` (arxiv), ``feedparser``, database drivers, etc.
"""

from __future__ import annotations

import importlib
from typing import Any

# Map every public name to the connectors submodule that defines it.
# Submodules that live in subpackages use dotted suffixes
# (e.g. "accounting.gaap" -> aragora.connectors.accounting.gaap).
_NAME_TO_MODULE: dict[str, str] = {
    # arxiv
    "ARXIV_CATEGORIES": "arxiv",
    "ArXivConnector": "arxiv",
    # base
    "BaseConnector": "base",
    "Evidence": "base",
    # whisper
    "WhisperConnector": "whisper",
    "TranscriptionResult": "whisper",
    "TranscriptionSegment": "whisper",
    "is_supported_audio": "whisper",
    "is_supported_video": "whisper",
    "is_supported_media": "whisper",
    "get_whisper_formats": "whisper",  # exported as get_supported_formats in submodule
    # exceptions
    "ConnectorAPIError": "exceptions",
    "ConnectorAuthError": "exceptions",
    "ConnectorError": "exceptions",
    "ConnectorNetworkError": "exceptions",
    "ConnectorNotFoundError": "exceptions",
    "ConnectorParseError": "exceptions",
    "ConnectorQuotaError": "exceptions",
    "ConnectorRateLimitError": "exceptions",
    "ConnectorTimeoutError": "exceptions",
    "ConnectorValidationError": "exceptions",
    "classify_exception": "exceptions",
    "connector_error_handler": "exceptions",
    "get_retry_delay": "exceptions",
    "is_retryable_error": "exceptions",
    # recovery
    "RecoveryAction": "recovery",
    "RecoveryConfig": "recovery",
    "RecoveryStrategy": "recovery",
    "create_recovery_chain": "recovery",
    "with_recovery": "recovery",
    # credentials
    "AWSSecretsManagerProvider": "credentials",
    "CachedCredentialProvider": "credentials",
    "ChainedCredentialProvider": "credentials",
    "CredentialProvider": "credentials",
    "EnvCredentialProvider": "credentials",
    "get_credential_provider": "credentials",
    # github
    "GitHubConnector": "github",
    # hackernews
    "HackerNewsConnector": "hackernews",
    # local_docs
    "LocalDocsConnector": "local_docs",
    # newsapi
    "HIGH_CREDIBILITY_SOURCES": "newsapi",
    "MEDIUM_CREDIBILITY_SOURCES": "newsapi",
    "NewsAPIConnector": "newsapi",
    # courtlistener
    "CourtListenerConnector": "courtlistener",
    # govinfo
    "GovInfoConnector": "govinfo",
    # nice_guidance
    "NICEGuidanceConnector": "nice_guidance",
    # pubmed
    "PubMedConnector": "pubmed",
    # semantic_scholar
    "SemanticScholarConnector": "semantic_scholar",
    # crossref
    "CrossRefConnector": "crossref",
    # clinical_tables
    "ClinicalTablesConnector": "clinical_tables",
    # rxnav
    "RxNavConnector": "rxnav",
    # reddit
    "RedditConnector": "reddit",
    # sec
    "FORM_TYPES": "sec",
    "SECConnector": "sec",
    # sql
    "SQLConnector": "sql",
    "SQLQueryResult": "sql",
    # twitter
    "TwitterConnector": "twitter",
    # web
    "WebConnector": "web",
    # wikipedia
    "WikipediaConnector": "wikipedia",
    # repository_crawler
    "RepositoryCrawler": "repository_crawler",
    "CrawlConfig": "repository_crawler",
    "CrawlResult": "repository_crawler",
    "CrawlState": "repository_crawler",
    "CrawledFile": "repository_crawler",
    "FileSymbol": "repository_crawler",
    "FileDependency": "repository_crawler",
    "FileType": "repository_crawler",
    "crawl_repository": "repository_crawler",
    # legal
    "DocuSignConnector": "legal",
    "DocuSignCredentials": "legal",
    "DocuSignEnvironment": "legal",
    "Envelope": "legal",
    "EnvelopeCreateRequest": "legal",
    "EnvelopeStatus": "legal",
    "Recipient": "legal",
    "RecipientType": "legal",
    "Document": "legal",
    "SignatureTab": "legal",
    "WestlawConnector": "legal",
    "LexisConnector": "legal",
    # accounting
    "FASBConnector": "accounting.gaap",
    "IRSConnector": "accounting.irs",
    # tax
    "GenericTaxConnector": "tax",
    "TaxConnectorRegistry": "tax",
    "resolve_tax_connector": "tax",
    # devops
    "PagerDutyConnector": "devops",
    "PagerDutyCredentials": "devops",
    "PagerDutyError": "devops",
    "Incident": "devops",
    "IncidentCreateRequest": "devops",
    "IncidentNote": "devops",
    "IncidentPriority": "devops",
    "IncidentStatus": "devops",
    "IncidentUrgency": "devops",
    "OnCallSchedule": "devops",
    "Service": "devops",
    "ServiceStatus": "devops",
    "User": "devops",
    "WebhookPayload": "devops",
    # blockchain
    "ERC8004Connector": "blockchain",
    "BlockchainCredentials": "blockchain",
    "BlockchainEvidence": "blockchain",
    "BlockchainSearchResult": "blockchain",
    # knowledge
    "ObsidianConnector": "knowledge",
    "ObsidianConfig": "knowledge",
    "ObsidianNote": "knowledge",
    "NoteType": "knowledge",
    "create_obsidian_connector": "knowledge",
    # memory
    "ClaudeMemConnector": "memory",
    "ClaudeMemConfig": "memory",
    # conversation_ingestor
    "ConversationIngestorConnector": "conversation_ingestor",
    "Conversation": "conversation_ingestor",
    "ConversationMessage": "conversation_ingestor",
    "ConversationExport": "conversation_ingestor",
    "ClaimExtraction": "conversation_ingestor",
}

# A few names are re-exported under a different alias than the attribute
# defined in the backing submodule. Map public name -> source attribute.
_NAME_ALIASES: dict[str, str] = {
    "get_whisper_formats": "get_supported_formats",
}


def __getattr__(name: str) -> Any:
    mod = _NAME_TO_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module 'aragora.connectors' has no attribute {name!r}")
    submod = importlib.import_module(f"aragora.connectors.{mod}")
    source_attr = _NAME_ALIASES.get(name, name)
    value = getattr(submod, source_attr)
    globals()[name] = value  # cache so repeat access is fast and stable
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # Base classes
    "BaseConnector",
    "Evidence",
    # Connectors
    "LocalDocsConnector",
    "GitHubConnector",
    "WebConnector",
    "ArXivConnector",
    "ARXIV_CATEGORIES",
    "HackerNewsConnector",
    "WikipediaConnector",
    "RedditConnector",
    "TwitterConnector",
    "SQLConnector",
    "SQLQueryResult",
    "NewsAPIConnector",
    "HIGH_CREDIBILITY_SOURCES",
    "MEDIUM_CREDIBILITY_SOURCES",
    "SECConnector",
    "FORM_TYPES",
    "CourtListenerConnector",
    "GovInfoConnector",
    "NICEGuidanceConnector",
    "PubMedConnector",
    "SemanticScholarConnector",
    "CrossRefConnector",
    "ClinicalTablesConnector",
    "RxNavConnector",
    # Whisper Transcription
    "WhisperConnector",
    "TranscriptionResult",
    "TranscriptionSegment",
    "is_supported_audio",
    "is_supported_video",
    "is_supported_media",
    "get_whisper_formats",
    # Exceptions
    "ConnectorError",
    "ConnectorAuthError",
    "ConnectorRateLimitError",
    "ConnectorTimeoutError",
    "ConnectorNetworkError",
    "ConnectorAPIError",
    "ConnectorValidationError",
    "ConnectorNotFoundError",
    "ConnectorQuotaError",
    "ConnectorParseError",
    # Exception Utilities
    "is_retryable_error",
    "get_retry_delay",
    "classify_exception",
    "connector_error_handler",
    # Recovery
    "RecoveryStrategy",
    "RecoveryConfig",
    "RecoveryAction",
    "with_recovery",
    "create_recovery_chain",
    # Credentials
    "CredentialProvider",
    "EnvCredentialProvider",
    "AWSSecretsManagerProvider",
    "ChainedCredentialProvider",
    "CachedCredentialProvider",
    "get_credential_provider",
    # Repository Crawler
    "RepositoryCrawler",
    "CrawlConfig",
    "CrawlResult",
    "CrawlState",
    "CrawledFile",
    "FileSymbol",
    "FileDependency",
    "FileType",
    "crawl_repository",
    # Legal Connectors
    "DocuSignConnector",
    "DocuSignCredentials",
    "DocuSignEnvironment",
    "Envelope",
    "EnvelopeCreateRequest",
    "EnvelopeStatus",
    "Recipient",
    "RecipientType",
    "Document",
    "SignatureTab",
    "WestlawConnector",
    "LexisConnector",
    "FASBConnector",
    "IRSConnector",
    "GenericTaxConnector",
    "TaxConnectorRegistry",
    "resolve_tax_connector",
    # DevOps Connectors
    "PagerDutyConnector",
    "PagerDutyCredentials",
    "PagerDutyError",
    "Incident",
    "IncidentCreateRequest",
    "IncidentNote",
    "IncidentPriority",
    "IncidentStatus",
    "IncidentUrgency",
    "OnCallSchedule",
    "Service",
    "ServiceStatus",
    "User",
    "WebhookPayload",
    # Blockchain Connectors (ERC-8004)
    "ERC8004Connector",
    "BlockchainCredentials",
    "BlockchainEvidence",
    "BlockchainSearchResult",
    # Knowledge Connectors
    "ObsidianConnector",
    "ObsidianConfig",
    "ObsidianNote",
    "NoteType",
    "create_obsidian_connector",
    # Memory Connectors
    "ClaudeMemConnector",
    "ClaudeMemConfig",
    # Conversation Ingestor
    "ConversationIngestorConnector",
    "Conversation",
    "ConversationMessage",
    "ConversationExport",
    "ClaimExtraction",
]
