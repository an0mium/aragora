"""Feature handlers - audio, broadcast, documents, evidence, pulse, plugins, features, audit, control plane, folder upload, finding workflow, NL query, scheduler, connectors."""

from .audio import AudioHandler, PODCAST_AVAILABLE
from .audit_sessions import AuditSessionsHandler
from .broadcast import BroadcastHandler
from .connectors import ConnectorsHandler
from .document_query import DocumentQueryHandler
from .evidence_enrichment import EvidenceEnrichmentHandler
from .finding_workflow import FindingWorkflowHandler
from .control_plane import ControlPlaneHandler
from .documents import DocumentHandler
from .documents_batch import DocumentBatchHandler
from .evidence import EvidenceHandler
from .folder_upload import FolderUploadHandler
from .scheduler import SchedulerHandler
from .features import (
    FEATURE_REGISTRY,
    FeatureInfo,
    FeaturesHandler,
    feature_unavailable_response,
    get_all_features,
    get_available_features,
    get_unavailable_features,
    _check_feature_available,
    _check_pulse,
    _check_genesis,
    _check_elo,
    _check_z3,
    _check_lean,
    _check_laboratory,
    _check_calibration,
    _check_evolution,
    _check_red_team,
    _check_probes,
    _check_continuum,
    _check_consensus,
    _check_insights,
    _check_moments,
    _check_tournaments,
    _check_crux,
    _check_rhetorical,
    _check_trickster,
    _check_requirement,
)
from .integrations import (
    IntegrationsHandler,
    IntegrationConfig,
    IntegrationStatus,
    IntegrationType,
    VALID_INTEGRATION_TYPES,
    register_integration_routes,
)
from .plugins import PluginsHandler
from .pulse import PulseHandler
from .rlm import RLMHandler
from .transcription import TranscriptionHandler
from .smart_upload import (
    SmartUploadHandler,
    FileCategory,
    ProcessingAction,
    UploadResult,
    detect_file_category,
    get_processing_action,
    smart_upload,
    get_upload_status,
)
from .cloud_storage import (
    CloudStorageHandler,
    get_provider_status,
    get_all_provider_status,
    list_files as cloud_list_files,
    download_file as cloud_download_file,
)
from .gmail_ingest import GmailIngestHandler
from .gmail_query import GmailQueryHandler
from .routing_rules import RoutingRulesHandler, routing_rules_handler
from .advertising import AdvertisingHandler
from .analytics_platforms import AnalyticsPlatformsHandler
from .crm import CRMHandler
from .support import SupportHandler
from .ecommerce import EcommerceHandler

__all__ = [
    "AudioHandler",
    "AuditSessionsHandler",
    "BroadcastHandler",
    "ConnectorsHandler",
    "DocumentQueryHandler",
    "EvidenceEnrichmentHandler",
    "FindingWorkflowHandler",
    "ControlPlaneHandler",
    "DocumentHandler",
    "DocumentBatchHandler",
    "EvidenceHandler",
    "FolderUploadHandler",
    "SchedulerHandler",
    "FEATURE_REGISTRY",
    "FeatureInfo",
    "FeaturesHandler",
    "feature_unavailable_response",
    "get_all_features",
    "get_available_features",
    "get_unavailable_features",
    "_check_feature_available",
    "_check_pulse",
    "_check_genesis",
    "_check_elo",
    "_check_z3",
    "_check_lean",
    "_check_laboratory",
    "_check_calibration",
    "_check_evolution",
    "_check_red_team",
    "_check_probes",
    "_check_continuum",
    "_check_consensus",
    "_check_insights",
    "_check_moments",
    "_check_tournaments",
    "_check_crux",
    "_check_rhetorical",
    "_check_trickster",
    "_check_requirement",
    "PluginsHandler",
    "PODCAST_AVAILABLE",
    "PulseHandler",
    "RLMHandler",
    "TranscriptionHandler",
    "IntegrationsHandler",
    "IntegrationConfig",
    "IntegrationStatus",
    "IntegrationType",
    "VALID_INTEGRATION_TYPES",
    "register_integration_routes",
    "SmartUploadHandler",
    "FileCategory",
    "ProcessingAction",
    "UploadResult",
    "detect_file_category",
    "get_processing_action",
    "smart_upload",
    "get_upload_status",
    "CloudStorageHandler",
    "get_provider_status",
    "get_all_provider_status",
    "cloud_list_files",
    "cloud_download_file",
    "GmailIngestHandler",
    "GmailQueryHandler",
    "RoutingRulesHandler",
    "routing_rules_handler",
    # Connector API handlers
    "AdvertisingHandler",
    "AnalyticsPlatformsHandler",
    "CRMHandler",
    "SupportHandler",
    "EcommerceHandler",
]
