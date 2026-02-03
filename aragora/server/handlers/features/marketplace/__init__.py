"""Marketplace Handler package - Template discovery and deployment.

Provides API endpoints for discovering, browsing, and deploying workflow templates
across different industry verticals.

Submodules:
- models: Enums (TemplateCategory, DeploymentStatus) and dataclasses
- validation: Input validation functions and constants
- circuit_breaker: MarketplaceCircuitBreaker for resilience
- store: In-memory storage and template discovery/loading
- handler: MarketplaceHandler class and module-level helpers
"""

# Models
from .models import CATEGORY_INFO  # noqa: F401
from .models import DeploymentStatus  # noqa: F401
from .models import TemplateCategory  # noqa: F401
from .models import TemplateDeployment  # noqa: F401
from .models import TemplateMetadata  # noqa: F401
from .models import TemplateRating  # noqa: F401

# Circuit breaker
from .circuit_breaker import MarketplaceCircuitBreaker  # noqa: F401
from .circuit_breaker import _get_circuit_breaker  # noqa: F401
from .circuit_breaker import _get_marketplace_circuit_breaker  # noqa: F401
from .circuit_breaker import get_marketplace_circuit_breaker_status  # noqa: F401

# Validation
from .validation import DEFAULT_LIMIT  # noqa: F401
from .validation import MAX_CONFIG_KEYS  # noqa: F401
from .validation import MAX_CONFIG_SIZE  # noqa: F401
from .validation import MAX_DEPLOYMENT_NAME_LENGTH  # noqa: F401
from .validation import MAX_LIMIT  # noqa: F401
from .validation import MAX_OFFSET  # noqa: F401
from .validation import MAX_RATING  # noqa: F401
from .validation import MAX_REVIEW_LENGTH  # noqa: F401
from .validation import MAX_SEARCH_QUERY_LENGTH  # noqa: F401
from .validation import MAX_TEMPLATE_NAME_LENGTH  # noqa: F401
from .validation import MIN_LIMIT  # noqa: F401
from .validation import MIN_RATING  # noqa: F401
from .validation import SAFE_ID_PATTERN  # noqa: F401
from .validation import SAFE_TEMPLATE_ID_PATTERN  # noqa: F401
from .validation import _clamp_pagination  # noqa: F401
from .validation import _validate_category  # noqa: F401
from .validation import _validate_category_filter  # noqa: F401
from .validation import _validate_config  # noqa: F401
from .validation import _validate_deployment_id  # noqa: F401
from .validation import _validate_deployment_name  # noqa: F401
from .validation import _validate_deployment_name_internal  # noqa: F401
from .validation import _validate_id  # noqa: F401
from .validation import _validate_pagination  # noqa: F401
from .validation import _validate_rating  # noqa: F401
from .validation import _validate_rating_value  # noqa: F401
from .validation import _validate_review  # noqa: F401
from .validation import _validate_review_internal  # noqa: F401
from .validation import _validate_search_query  # noqa: F401
from .validation import _validate_template_id  # noqa: F401

# Store
from .store import _clear_marketplace_components  # noqa: F401
from .store import _clear_marketplace_state  # noqa: F401
from .store import _get_full_template  # noqa: F401
from .store import _get_templates_dir  # noqa: F401
from .store import _get_tenant_deployments  # noqa: F401
from .store import _load_templates  # noqa: F401
from .store import _parse_template_file  # noqa: F401
from .store import _templates_cache  # noqa: F401
from .store import _deployments  # noqa: F401
from .store import _ratings  # noqa: F401
from .store import _download_counts  # noqa: F401

# Handler
from .handler import MarketplaceHandler  # noqa: F401
from .handler import get_marketplace_handler  # noqa: F401
from .handler import handle_marketplace  # noqa: F401
