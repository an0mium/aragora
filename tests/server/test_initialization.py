from __future__ import annotations

from types import SimpleNamespace

import pytest

from aragora.server import initialization as init_mod


@pytest.mark.asyncio
async def test_init_postgres_stores_sets_global_store_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aragora.storage.factory import StorageBackend

    shared_pool = object()
    setter_calls: dict[str, object] = {}

    monkeypatch.setattr(
        "aragora.storage.factory.get_storage_backend", lambda: StorageBackend.POSTGRES
    )
    monkeypatch.setattr("aragora.storage.pool_manager.is_pool_initialized", lambda: True)
    monkeypatch.setattr("aragora.storage.pool_manager.get_shared_pool", lambda: shared_pool)

    def make_module(module_path: str, class_name: str, setter_name: str | None) -> object:
        class FakeStore:
            def __init__(self, pool):
                self.pool = pool
                self.initialized = False

            async def initialize(self) -> None:
                self.initialized = True

        module = SimpleNamespace(**{class_name: FakeStore})
        if setter_name:
            setattr(
                module,
                setter_name,
                lambda store, key=module_path: setter_calls.setdefault(key, store),
            )
        return module

    stores_to_setters = {
        "aragora.storage.webhook_config_store": (
            "PostgresWebhookConfigStore",
            "set_webhook_config_store",
        ),
        "aragora.storage.integration_store": ("PostgresIntegrationStore", "set_integration_store"),
        "aragora.storage.gmail_token_store": ("PostgresGmailTokenStore", None),
        "aragora.storage.finding_workflow_store": ("PostgresFindingWorkflowStore", None),
        "aragora.storage.gauntlet_run_store": ("PostgresGauntletRunStore", None),
        "aragora.storage.job_queue_store": ("PostgresJobQueueStore", None),
        "aragora.storage.governance_store": ("PostgresGovernanceStore", None),
        "aragora.storage.marketplace_store": ("PostgresMarketplaceStore", None),
        "aragora.storage.federation_registry_store": ("PostgresFederationRegistryStore", None),
        "aragora.storage.approval_request_store": ("PostgresApprovalRequestStore", None),
        "aragora.storage.token_blacklist_store": ("PostgresBlacklist", None),
        "aragora.storage.user_store": ("PostgresUserStore", None),
        "aragora.storage.webhook_store": ("PostgresWebhookStore", "set_webhook_store"),
        "aragora.insights.postgres_store": ("PostgresInsightStore", None),
        "aragora.server.postgres_storage": ("PostgresDebateStorage", None),
        "aragora.pulse.postgres_store": ("PostgresScheduledDebateStore", None),
        "aragora.nomic.postgres_cycle_store": ("PostgresCycleLearningStore", None),
    }

    modules = {
        path: make_module(path, class_name, setter_name)
        for path, (class_name, setter_name) in stores_to_setters.items()
    }
    monkeypatch.setattr(init_mod.importlib, "import_module", lambda path: modules[path])

    results = await init_mod.init_postgres_stores()

    assert results["webhook_configs"] is True
    assert results["integrations"] is True
    assert results["webhooks"] is True
    assert setter_calls["aragora.storage.webhook_config_store"].pool is shared_pool
    assert setter_calls["aragora.storage.integration_store"].pool is shared_pool
    assert setter_calls["aragora.storage.webhook_store"].pool is shared_pool
