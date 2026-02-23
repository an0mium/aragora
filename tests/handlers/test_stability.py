"""Tests for aragora/server/handlers/_stability.py.

Comprehensive coverage of handler stability classifications:
1. HANDLER_STABILITY dictionary - completeness, valid values, categorization
2. get_handler_stability() - lookup, default fallback, edge cases
3. get_all_handler_stability() - string conversion, completeness
4. __all__ exports - correctness and importability
5. Stability enum integration - valid enum members only
6. Security/edge cases - injection, special characters, boundary conditions
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.config.stability import Stability
from aragora.server.handlers._stability import (
    HANDLER_STABILITY,
    __all__,
    get_all_handler_stability,
    get_handler_stability,
)


# =============================================================================
# HANDLER_STABILITY Dictionary Tests
# =============================================================================


class TestHandlerStabilityDict:
    """Test the HANDLER_STABILITY mapping."""

    def test_is_dict(self):
        assert isinstance(HANDLER_STABILITY, dict)

    def test_is_non_empty(self):
        assert len(HANDLER_STABILITY) > 0

    def test_has_over_100_entries(self):
        """Mapping should have a substantial number of handler entries."""
        assert len(HANDLER_STABILITY) >= 100

    def test_all_keys_are_strings(self):
        for key in HANDLER_STABILITY:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_all_values_are_stability_enum(self):
        for name, stability in HANDLER_STABILITY.items():
            assert isinstance(stability, Stability), (
                f"Value for {name!r} is {type(stability)}, expected Stability"
            )

    def test_all_keys_end_with_handler(self):
        """Handler class names should follow the naming convention."""
        for name in HANDLER_STABILITY:
            assert name.endswith("Handler"), f"Handler name {name!r} does not end with 'Handler'"

    def test_no_empty_keys(self):
        for key in HANDLER_STABILITY:
            assert key.strip() != "", "Empty key found in HANDLER_STABILITY"

    def test_no_duplicate_keys(self):
        """Dict keys are inherently unique, but verify count matches set size."""
        keys = list(HANDLER_STABILITY.keys())
        assert len(keys) == len(set(keys))

    def test_keys_are_pascal_case(self):
        """Handler names should be PascalCase."""
        for name in HANDLER_STABILITY:
            assert name[0].isupper(), f"Handler name {name!r} does not start with uppercase"

    def test_no_leading_or_trailing_whitespace_in_keys(self):
        for name in HANDLER_STABILITY:
            assert name == name.strip(), f"Handler name {name!r} has leading/trailing whitespace"


# =============================================================================
# Core Stable Handlers
# =============================================================================


class TestCoreStableHandlers:
    """Verify core handlers are classified as STABLE."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "DebatesHandler",
            "AgentConfigHandler",
            "AgentsHandler",
            "SystemHandler",
            "HealthHandler",
            "StatusPageHandler",
            "NomicHandler",
            "DocsHandler",
            "AnalyticsHandler",
            "ConsensusHandler",
            "MetricsHandler",
            "MemoryHandler",
            "AuthHandler",
            "DecisionHandler",
        ],
    )
    def test_core_handler_is_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "TournamentHandler",
            "ControlPlaneHandler",
            "CostDashboardHandler",
            "CostHandler",
            "CritiqueHandler",
            "BillingHandler",
            "BudgetHandler",
            "OAuthHandler",
            "PulseHandler",
            "GauntletHandler",
            "BeliefHandler",
            "SkillsHandler",
            "CalibrationHandler",
        ],
    )
    def test_extended_stable_handlers(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "DiscordHandler",
            "GoogleChatHandler",
            "TeamsHandler",
            "TelegramHandler",
            "WhatsAppHandler",
            "ZoomHandler",
            "SlackHandler",
        ],
    )
    def test_chat_platform_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "WorkflowHandler",
            "WorkflowTemplatesHandler",
            "WorkflowCategoriesHandler",
            "WorkflowPatternsHandler",
            "WorkflowPatternTemplatesHandler",
        ],
    )
    def test_workflow_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "GauntletSchemaHandler",
            "GauntletAllSchemasHandler",
            "GauntletTemplatesListHandler",
            "GauntletTemplateHandler",
            "GauntletReceiptExportHandler",
            "GauntletHeatmapExportHandler",
            "GauntletValidateReceiptHandler",
        ],
    )
    def test_gauntlet_sub_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE


# =============================================================================
# Experimental Handlers
# =============================================================================


class TestExperimentalHandlers:
    """Verify experimental handlers are classified correctly."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "CloudStorageHandler",
            "FindingWorkflowHandler",
            "EvidenceEnrichmentHandler",
            "SchedulerHandler",
            "AuditSessionsHandler",
            "A2AHandler",
            "AlertHandler",
            "AutonomousLearningHandler",
            "DependencyAnalysisHandler",
            "CodebaseAuditHandler",
        ],
    )
    def test_experimental_handler(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL

    @pytest.mark.parametrize(
        "handler_name",
        [
            "CreditsAdminHandler",
            "EmergencyAccessHandler",
            "FeatureFlagAdminHandler",
            "StorageHealthHandler",
            "AgentRecommendationHandler",
            "FeedbackHandler",
        ],
    )
    def test_admin_and_agent_experimental(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL

    @pytest.mark.parametrize(
        "handler_name",
        [
            "ActionCanvasHandler",
            "GoalCanvasHandler",
            "IdeaCanvasHandler",
            "OrchestrationCanvasHandler",
        ],
    )
    def test_canvas_handlers_are_experimental(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL


# =============================================================================
# Stability Level Distribution
# =============================================================================


class TestStabilityDistribution:
    """Test the distribution of stability levels across handlers."""

    def test_stable_handlers_exist(self):
        stable = [name for name, s in HANDLER_STABILITY.items() if s == Stability.STABLE]
        assert len(stable) > 0

    def test_experimental_handlers_exist(self):
        experimental = [
            name for name, s in HANDLER_STABILITY.items() if s == Stability.EXPERIMENTAL
        ]
        assert len(experimental) > 0

    def test_stable_is_most_common(self):
        """Stable should be the most common classification."""
        counts = {}
        for s in HANDLER_STABILITY.values():
            counts[s] = counts.get(s, 0) + 1
        assert counts.get(Stability.STABLE, 0) > counts.get(Stability.EXPERIMENTAL, 0)

    def test_no_preview_handlers(self):
        """Currently no handlers are classified as PREVIEW."""
        preview = [name for name, s in HANDLER_STABILITY.items() if s == Stability.PREVIEW]
        assert len(preview) == 0

    def test_no_deprecated_handlers(self):
        """Currently no handlers are classified as DEPRECATED."""
        deprecated = [name for name, s in HANDLER_STABILITY.items() if s == Stability.DEPRECATED]
        assert len(deprecated) == 0

    def test_only_two_stability_levels_used(self):
        """Only STABLE and EXPERIMENTAL are currently used."""
        levels = set(HANDLER_STABILITY.values())
        assert levels == {Stability.STABLE, Stability.EXPERIMENTAL}


# =============================================================================
# get_handler_stability() Tests
# =============================================================================


class TestGetHandlerStability:
    """Test the get_handler_stability function."""

    def test_returns_stable_for_known_stable_handler(self):
        assert get_handler_stability("DebatesHandler") == Stability.STABLE

    def test_returns_experimental_for_known_experimental_handler(self):
        assert get_handler_stability("A2AHandler") == Stability.EXPERIMENTAL

    def test_defaults_to_experimental_for_unknown_handler(self):
        assert get_handler_stability("NonExistentHandler") == Stability.EXPERIMENTAL

    def test_defaults_to_experimental_for_empty_string(self):
        assert get_handler_stability("") == Stability.EXPERIMENTAL

    def test_case_sensitive_lookup(self):
        """Lookup is case-sensitive; wrong case falls back to EXPERIMENTAL."""
        assert get_handler_stability("debateshandler") == Stability.EXPERIMENTAL
        assert get_handler_stability("DEBATESHANDLER") == Stability.EXPERIMENTAL

    def test_returns_stability_enum_type(self):
        result = get_handler_stability("HealthHandler")
        assert isinstance(result, Stability)

    def test_returns_stability_enum_for_unknown(self):
        result = get_handler_stability("BogusHandler")
        assert isinstance(result, Stability)

    def test_all_registered_handlers_return_correct_stability(self):
        """Every registered handler should return its mapped stability."""
        for name, expected in HANDLER_STABILITY.items():
            actual = get_handler_stability(name)
            assert actual == expected, (
                f"get_handler_stability({name!r}) returned {actual}, expected {expected}"
            )

    def test_whitespace_name_not_found(self):
        assert get_handler_stability("  DebatesHandler  ") == Stability.EXPERIMENTAL

    def test_unicode_name_not_found(self):
        assert get_handler_stability("D\u00e9batesHandler") == Stability.EXPERIMENTAL

    def test_none_like_string(self):
        assert get_handler_stability("None") == Stability.EXPERIMENTAL

    def test_numeric_string(self):
        assert get_handler_stability("12345") == Stability.EXPERIMENTAL

    def test_special_characters(self):
        assert get_handler_stability("Handler<script>") == Stability.EXPERIMENTAL

    def test_path_traversal_string(self):
        assert get_handler_stability("../../etc/passwd") == Stability.EXPERIMENTAL

    def test_sql_injection_string(self):
        assert get_handler_stability("'; DROP TABLE handlers;--") == Stability.EXPERIMENTAL

    def test_very_long_string(self):
        long_name = "X" * 10000
        assert get_handler_stability(long_name) == Stability.EXPERIMENTAL

    def test_newline_in_name(self):
        assert get_handler_stability("Debates\nHandler") == Stability.EXPERIMENTAL

    def test_tab_in_name(self):
        assert get_handler_stability("Debates\tHandler") == Stability.EXPERIMENTAL

    def test_null_byte_in_name(self):
        assert get_handler_stability("Debates\x00Handler") == Stability.EXPERIMENTAL


# =============================================================================
# get_all_handler_stability() Tests
# =============================================================================


class TestGetAllHandlerStability:
    """Test the get_all_handler_stability function."""

    def test_returns_dict(self):
        result = get_all_handler_stability()
        assert isinstance(result, dict)

    def test_same_length_as_handler_stability(self):
        result = get_all_handler_stability()
        assert len(result) == len(HANDLER_STABILITY)

    def test_all_keys_match(self):
        result = get_all_handler_stability()
        assert set(result.keys()) == set(HANDLER_STABILITY.keys())

    def test_all_values_are_strings(self):
        result = get_all_handler_stability()
        for name, value in result.items():
            assert isinstance(value, str), f"Value for {name!r} is {type(value)}, expected str"

    def test_stable_value_is_string_stable(self):
        result = get_all_handler_stability()
        assert result["DebatesHandler"] == "stable"

    def test_experimental_value_is_string_experimental(self):
        result = get_all_handler_stability()
        assert result["A2AHandler"] == "experimental"

    def test_values_are_valid_stability_strings(self):
        valid_strings = {s.value for s in Stability}
        result = get_all_handler_stability()
        for name, value in result.items():
            assert value in valid_strings, (
                f"Value {value!r} for {name!r} is not a valid Stability string"
            )

    def test_returns_new_dict_each_call(self):
        """Function should return a new dict, not a reference to the internal one."""
        result1 = get_all_handler_stability()
        result2 = get_all_handler_stability()
        assert result1 == result2
        assert result1 is not result2

    def test_modifying_result_does_not_affect_original(self):
        result = get_all_handler_stability()
        result["DebatesHandler"] = "deprecated"
        # Original should be unchanged
        fresh = get_all_handler_stability()
        assert fresh["DebatesHandler"] == "stable"

    def test_result_contains_all_registered_handlers(self):
        result = get_all_handler_stability()
        for name in HANDLER_STABILITY:
            assert name in result, f"{name!r} missing from get_all_handler_stability()"

    def test_each_value_matches_enum_value(self):
        result = get_all_handler_stability()
        for name, stability in HANDLER_STABILITY.items():
            assert result[name] == stability.value, (
                f"Mismatch for {name!r}: {result[name]} != {stability.value}"
            )


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestAllExports:
    """Test the __all__ list of module exports."""

    def test_all_is_list(self):
        assert isinstance(__all__, list)

    def test_all_has_three_items(self):
        assert len(__all__) == 3

    def test_handler_stability_in_all(self):
        assert "HANDLER_STABILITY" in __all__

    def test_get_handler_stability_in_all(self):
        assert "get_handler_stability" in __all__

    def test_get_all_handler_stability_in_all(self):
        assert "get_all_handler_stability" in __all__

    def test_all_items_are_strings(self):
        for item in __all__:
            assert isinstance(item, str)

    def test_no_duplicates_in_all(self):
        assert len(__all__) == len(set(__all__))

    def test_all_items_importable_from_module(self):
        from aragora.server.handlers import _stability

        for name in __all__:
            assert hasattr(_stability, name), (
                f"{name!r} is in __all__ but not accessible on the module"
            )


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImport:
    """Test that the module can be imported and used correctly."""

    def test_import_handler_stability(self):
        from aragora.server.handlers._stability import HANDLER_STABILITY as hs

        assert hs is HANDLER_STABILITY

    def test_import_get_handler_stability(self):
        from aragora.server.handlers._stability import (
            get_handler_stability as ghs,
        )

        assert callable(ghs)

    def test_import_get_all_handler_stability(self):
        from aragora.server.handlers._stability import (
            get_all_handler_stability as gahs,
        )

        assert callable(gahs)

    def test_stability_enum_imported_correctly(self):
        """The module uses Stability from aragora.config.stability."""
        assert Stability.STABLE.value == "stable"
        assert Stability.EXPERIMENTAL.value == "experimental"
        assert Stability.PREVIEW.value == "preview"
        assert Stability.DEPRECATED.value == "deprecated"


# =============================================================================
# Specific Handler Group Tests
# =============================================================================


class TestEmailHandlers:
    """Test email-related handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "EmailHandler",
            "EmailServicesHandler",
            "GmailIngestHandler",
            "GmailQueryHandler",
            "UnifiedInboxHandler",
            "EmailWebhooksHandler",
        ],
    )
    def test_email_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "EmailDebateHandler",
            "EmailTriageHandler",
        ],
    )
    def test_email_experimental_handlers(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL


class TestGatewayHandlers:
    """Test gateway-related handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "OpenClawGatewayHandler",
            "GatewayHealthHandler",
            "GatewayAgentsHandler",
            "GatewayCredentialsHandler",
        ],
    )
    def test_gateway_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    def test_gateway_config_is_experimental(self):
        assert HANDLER_STABILITY["GatewayConfigHandler"] == Stability.EXPERIMENTAL


class TestDocumentHandlers:
    """Test document-related handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "DocumentHandler",
            "DocumentBatchHandler",
            "DocumentQueryHandler",
            "FolderUploadHandler",
            "SmartUploadHandler",
        ],
    )
    def test_document_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE


class TestHealthAndAdminHandlers:
    """Test health and admin handler stability classifications."""

    def test_liveness_handler_is_stable(self):
        assert HANDLER_STABILITY["LivenessHandler"] == Stability.STABLE

    def test_readiness_handler_is_stable(self):
        assert HANDLER_STABILITY["ReadinessHandler"] == Stability.STABLE

    def test_admin_handler_is_stable(self):
        assert HANDLER_STABILITY["AdminHandler"] == Stability.STABLE

    def test_storage_health_is_experimental(self):
        assert HANDLER_STABILITY["StorageHealthHandler"] == Stability.EXPERIMENTAL


class TestSMEHandlers:
    """Test SME-related handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "BudgetControlsHandler",
            "ReceiptDeliveryHandler",
            "SlackWorkspaceHandler",
            "TeamsWorkspaceHandler",
            "SMESuccessDashboardHandler",
            "SMEWorkflowsHandler",
        ],
    )
    def test_sme_handlers_are_experimental(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL


class TestSocialHandlers:
    """Test social-related handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "ChannelHealthHandler",
            "DiscordOAuthHandler",
            "NotificationsHandler",
            "SharingHandler",
            "SlackOAuthHandler",
            "TeamsOAuthHandler",
        ],
    )
    def test_social_sub_handlers_are_experimental(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL


class TestAnalyticsHandlers:
    """Test analytics handler stability classifications."""

    @pytest.mark.parametrize(
        "handler_name",
        [
            "AnalyticsHandler",
            "AnalyticsDashboardHandler",
            "AnalyticsMetricsHandler",
            "EndpointAnalyticsHandler",
            "CrossPlatformAnalyticsHandler",
        ],
    )
    def test_analytics_handlers_are_stable(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.STABLE

    def test_analytics_platforms_is_experimental(self):
        assert HANDLER_STABILITY["AnalyticsPlatformsHandler"] == Stability.EXPERIMENTAL


class TestGovernanceHandlers:
    """Test governance-related handler stability classifications."""

    def test_outcome_handler_is_stable(self):
        assert HANDLER_STABILITY["OutcomeHandler"] == Stability.STABLE

    def test_readiness_check_handler_is_stable(self):
        assert HANDLER_STABILITY["ReadinessCheckHandler"] == Stability.STABLE


class TestSecurityAndComplianceHandlers:
    """Test security and compliance handler classifications."""

    def test_security_handler_is_stable(self):
        assert HANDLER_STABILITY["SecurityHandler"] == Stability.STABLE

    def test_policy_handler_is_stable(self):
        assert HANDLER_STABILITY["PolicyHandler"] == Stability.STABLE

    def test_privacy_handler_is_stable(self):
        assert HANDLER_STABILITY["PrivacyHandler"] == Stability.STABLE

    def test_sso_handler_is_stable(self):
        assert HANDLER_STABILITY["SSOHandler"] == Stability.STABLE

    def test_scim_handler_is_stable(self):
        assert HANDLER_STABILITY["SCIMHandler"] == Stability.STABLE

    def test_gdpr_deletion_handler_is_stable(self):
        assert HANDLER_STABILITY["GDPRDeletionHandler"] == Stability.STABLE

    def test_security_debate_handler_is_stable(self):
        assert HANDLER_STABILITY["SecurityDebateHandler"] == Stability.STABLE

    def test_audit_trail_handler_is_stable(self):
        assert HANDLER_STABILITY["AuditTrailHandler"] == Stability.STABLE


class TestIntegrationHandlers:
    """Test integration handler stability classifications."""

    def test_integrations_handler_is_stable(self):
        assert HANDLER_STABILITY["IntegrationsHandler"] == Stability.STABLE

    def test_external_integrations_handler_is_stable(self):
        assert HANDLER_STABILITY["ExternalIntegrationsHandler"] == Stability.STABLE

    def test_integration_management_handler_is_stable(self):
        assert HANDLER_STABILITY["IntegrationManagementHandler"] == Stability.STABLE

    def test_automation_handler_is_experimental(self):
        assert HANDLER_STABILITY["AutomationHandler"] == Stability.EXPERIMENTAL

    def test_integration_health_is_experimental(self):
        assert HANDLER_STABILITY["IntegrationHealthHandler"] == Stability.EXPERIMENTAL


class TestAccountingHandlers:
    """Test accounting handler stability classifications."""

    def test_expense_handler_is_stable(self):
        assert HANDLER_STABILITY["ExpenseHandler"] == Stability.STABLE

    def test_invoice_handler_is_stable(self):
        assert HANDLER_STABILITY["InvoiceHandler"] == Stability.STABLE

    @pytest.mark.parametrize(
        "handler_name",
        [
            "ARAutomationHandler",
            "APAutomationHandler",
            "ReconciliationHandler",
        ],
    )
    def test_automation_accounting_is_experimental(self, handler_name: str):
        assert HANDLER_STABILITY[handler_name] == Stability.EXPERIMENTAL


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_handler_stability_with_substring_of_real_name(self):
        """Partial handler name should not match."""
        assert get_handler_stability("Debates") == Stability.EXPERIMENTAL

    def test_get_handler_stability_with_prefix_of_real_name(self):
        assert get_handler_stability("DebatesHandlerExtra") == Stability.EXPERIMENTAL

    def test_handler_stability_dict_is_not_empty_after_import(self):
        """Verify the dict is populated at import time."""
        assert len(HANDLER_STABILITY) > 50

    def test_get_all_returns_consistent_results(self):
        """Multiple calls return the same data."""
        r1 = get_all_handler_stability()
        r2 = get_all_handler_stability()
        assert r1 == r2

    def test_stability_values_are_lowercase_strings(self):
        result = get_all_handler_stability()
        for name, value in result.items():
            assert value == value.lower(), f"Value {value!r} for {name!r} is not lowercase"

    def test_handler_names_contain_no_dots(self):
        """Handler names are class names without module paths."""
        for name in HANDLER_STABILITY:
            assert "." not in name, f"Handler name {name!r} contains a dot"

    def test_handler_names_contain_no_slashes(self):
        for name in HANDLER_STABILITY:
            assert "/" not in name, f"Handler name {name!r} contains a slash"
            assert "\\" not in name, f"Handler name {name!r} contains a backslash"

    def test_handler_names_are_ascii(self):
        for name in HANDLER_STABILITY:
            assert name.isascii(), f"Handler name {name!r} contains non-ASCII characters"


class TestSecurityEdgeCases:
    """Test security-related edge cases for the lookup function."""

    def test_html_injection_in_name(self):
        result = get_handler_stability("<b>DebatesHandler</b>")
        assert result == Stability.EXPERIMENTAL

    def test_json_injection_in_name(self):
        result = get_handler_stability('{"handler": "DebatesHandler"}')
        assert result == Stability.EXPERIMENTAL

    def test_template_injection_in_name(self):
        result = get_handler_stability("{{DebatesHandler}}")
        assert result == Stability.EXPERIMENTAL

    def test_url_encoded_name(self):
        result = get_handler_stability("Debates%48andler")
        assert result == Stability.EXPERIMENTAL

    def test_backslash_escape_in_name(self):
        result = get_handler_stability("Debates\\Handler")
        assert result == Stability.EXPERIMENTAL

    def test_emoji_in_name(self):
        result = get_handler_stability("DebatesHandler\U0001f680")
        assert result == Stability.EXPERIMENTAL


# =============================================================================
# Consistency Tests
# =============================================================================


class TestConsistency:
    """Test consistency between different access methods."""

    def test_get_handler_stability_matches_dict_for_all_entries(self):
        for name, expected in HANDLER_STABILITY.items():
            assert get_handler_stability(name) == expected

    def test_get_all_stability_matches_dict_for_all_entries(self):
        all_stabilities = get_all_handler_stability()
        for name, stability in HANDLER_STABILITY.items():
            assert all_stabilities[name] == stability.value

    def test_dict_and_function_agree_on_stable_count(self):
        dict_stable = sum(1 for s in HANDLER_STABILITY.values() if s == Stability.STABLE)
        all_stable = sum(1 for v in get_all_handler_stability().values() if v == "stable")
        assert dict_stable == all_stable

    def test_dict_and_function_agree_on_experimental_count(self):
        dict_exp = sum(1 for s in HANDLER_STABILITY.values() if s == Stability.EXPERIMENTAL)
        all_exp = sum(1 for v in get_all_handler_stability().values() if v == "experimental")
        assert dict_exp == all_exp


# =============================================================================
# Miscellaneous / Regression
# =============================================================================


class TestMiscellaneous:
    """Miscellaneous tests for completeness."""

    def test_debates_handler_in_map(self):
        assert "DebatesHandler" in HANDLER_STABILITY

    def test_erc8004_handler_is_stable(self):
        assert HANDLER_STABILITY["ERC8004Handler"] == Stability.STABLE

    def test_hybrid_debate_handler_is_stable(self):
        assert HANDLER_STABILITY["HybridDebateHandler"] == Stability.STABLE

    def test_external_agents_handler_is_stable(self):
        assert HANDLER_STABILITY["ExternalAgentsHandler"] == Stability.STABLE

    def test_audience_suggestions_handler_is_experimental(self):
        assert HANDLER_STABILITY["AudienceSuggestionsHandler"] == Stability.EXPERIMENTAL

    def test_debate_stats_handler_is_stable(self):
        assert HANDLER_STABILITY["DebateStatsHandler"] == Stability.STABLE

    def test_debate_share_handler_is_experimental(self):
        assert HANDLER_STABILITY["DebateShareHandler"] == Stability.EXPERIMENTAL

    def test_knowledge_mound_handler_is_stable(self):
        assert HANDLER_STABILITY["KnowledgeMoundHandler"] == Stability.STABLE

    def test_knowledge_chat_handler_is_stable(self):
        assert HANDLER_STABILITY["KnowledgeChatHandler"] == Stability.STABLE

    def test_receipt_handlers_are_stable(self):
        assert HANDLER_STABILITY["ReceiptsHandler"] == Stability.STABLE
        assert HANDLER_STABILITY["ReceiptExportHandler"] == Stability.STABLE

    def test_workspace_handler_is_stable(self):
        assert HANDLER_STABILITY["WorkspaceHandler"] == Stability.STABLE

    def test_marketplace_handler_is_stable(self):
        assert HANDLER_STABILITY["MarketplaceHandler"] == Stability.STABLE

    def test_pipeline_handlers_are_experimental(self):
        assert HANDLER_STABILITY["PipelineGraphHandler"] == Stability.EXPERIMENTAL
        assert HANDLER_STABILITY["PipelineTransitionsHandler"] == Stability.EXPERIMENTAL

    def test_notification_handlers_are_experimental(self):
        assert HANDLER_STABILITY["NotificationHistoryHandler"] == Stability.EXPERIMENTAL
        assert HANDLER_STABILITY["NotificationPreferencesHandler"] == Stability.EXPERIMENTAL

    def test_threat_intel_handler_is_experimental(self):
        assert HANDLER_STABILITY["ThreatIntelHandler"] == Stability.EXPERIMENTAL

    def test_moderation_handlers_are_experimental(self):
        assert HANDLER_STABILITY["ModerationHandler"] == Stability.EXPERIMENTAL
        assert HANDLER_STABILITY["ModerationAnalyticsHandler"] == Stability.EXPERIMENTAL

    def test_streaming_connector_handler_is_experimental(self):
        assert HANDLER_STABILITY["StreamingConnectorHandler"] == Stability.EXPERIMENTAL

    def test_unified_memory_handler_is_experimental(self):
        assert HANDLER_STABILITY["UnifiedMemoryHandler"] == Stability.EXPERIMENTAL
