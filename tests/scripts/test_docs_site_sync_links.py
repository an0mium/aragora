from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SITE_ROOT = REPO_ROOT / "docs-site" / "docs"


def _read_docs_site(path: str) -> str:
    return (DOCS_SITE_ROOT / path).read_text(encoding="utf-8")


def test_documentation_index_rewrites_status_and_planning_links() -> None:
    content = _read_docs_site("contributing/documentation-index.md")

    expected_links = [
        "[Aragora Conductor Workflow](../guides/conductor-workflow)",
        "[Aragora Worker Prompt Pack](../guides/worker-prompt-pack)",
        "[Dev Swarm Coordination](./dev-swarm-coordination)",
        "[Conductor Control Plane Implementation Spec](./conductor-control-plane-implementation-spec)",
        "[Feature Discovery](./feature-discovery)",
        "[Feature Gap List](./feature-gap-list)",
        "[Next Steps (Canonical)](./next-steps-canonical)",
        "[Active 6-Week Execution Plan](./execution-next-6-weeks-2026-03-05)",
        "[Documentation Hygiene Register](./documentation-hygiene-and-gap-register)",
        "[Roadmap](./roadmap)",
    ]
    for link in expected_links:
        assert link in content

    unresolved_source_links = [
        "guides/CONDUCTOR_WORKFLOW.md",
        "guides/WORKER_PROMPT_PACK.md",
        "architecture/DEV_SWARM_COORDINATION.md",
        "plans/2026-03-07-conductor-control-plane.md",
        "status/FEATURE_DISCOVERY.md",
        "FEATURE_GAP_LIST.md",
        "status/NEXT_STEPS_CANONICAL.md",
        "status/EXECUTION_NEXT_6_WEEKS_2026-03-05.md",
        "status/DOCUMENTATION_HYGIENE_AND_GAP_REGISTER.md",
        "../ROADMAP.md",
    ]
    for link in unresolved_source_links:
        assert link not in content


def test_features_guide_points_to_current_state_docs_site_pages() -> None:
    content = _read_docs_site("guides/features.md")

    expected_links = [
        "[STATUS](../contributing/status)",
        "[FEATURE_DISCOVERY](../contributing/feature-discovery)",
        "[FEATURE_GAP_LIST](../contributing/feature-gap-list)",
        "[DOCUMENTATION_HYGIENE_AND_GAP_REGISTER](../contributing/documentation-hygiene-and-gap-register)",
    ]
    for link in expected_links:
        assert link in content

    unresolved_source_links = [
        "[FEATURE_DISCOVERY](FEATURE_DISCOVERY.md)",
        "[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md)",
        "[DOCUMENTATION_HYGIENE_AND_GAP_REGISTER](DOCUMENTATION_HYGIENE_AND_GAP_REGISTER.md)",
    ]
    for link in unresolved_source_links:
        assert link not in content


def test_commercial_overview_rewrites_current_proof_status_links() -> None:
    content = _read_docs_site("enterprise/commercial-overview.md")

    expected_links = [
        "[status/NEXT_STEPS_CANONICAL.md](../contributing/next-steps-canonical)",
        "[status/ACTIVE_EXECUTION_ISSUES.md](../contributing/active-execution-issues)",
        "[status/B0_BENCHMARK_TRUTH_STATUS.md](../contributing/b0-benchmark-truth-status)",
        "[status/TW03_RESCUE_PRODUCTIZATION_STATUS.md](../contributing/tw03-rescue-productization-status)",
    ]
    for link in expected_links:
        assert link in content

    unresolved_source_links = [
        "[status/NEXT_STEPS_CANONICAL.md](status/NEXT_STEPS_CANONICAL.md)",
        "[status/ACTIVE_EXECUTION_ISSUES.md](status/ACTIVE_EXECUTION_ISSUES.md)",
        "[status/B0_BENCHMARK_TRUTH_STATUS.md](status/B0_BENCHMARK_TRUTH_STATUS.md)",
        "[status/TW03_RESCUE_PRODUCTIZATION_STATUS.md](status/TW03_RESCUE_PRODUCTIZATION_STATUS.md)",
    ]
    for link in unresolved_source_links:
        assert link not in content


def test_docs_site_sync_creates_linked_status_and_planning_pages() -> None:
    expected_pages = [
        DOCS_SITE_ROOT / "contributing" / "2026-03-26-pmf-14-day-execution-plan.md",
        DOCS_SITE_ROOT / "contributing" / "active-execution-issues.md",
        DOCS_SITE_ROOT / "contributing" / "b0-benchmark-truth-status.md",
        DOCS_SITE_ROOT / "contributing" / "aragora-evolution-roadmap.md",
        DOCS_SITE_ROOT / "contributing" / "canonical-goals.md",
        DOCS_SITE_ROOT / "contributing" / "claude.md",
        DOCS_SITE_ROOT / "contributing" / "conductor-control-plane-implementation-spec.md",
        DOCS_SITE_ROOT / "contributing" / "dev-swarm-coordination.md",
        DOCS_SITE_ROOT / "contributing" / "documentation-hygiene-and-gap-register.md",
        DOCS_SITE_ROOT / "contributing" / "execution-next-6-weeks-2026-03-05.md",
        DOCS_SITE_ROOT / "contributing" / "extended-readme.md",
        DOCS_SITE_ROOT / "contributing" / "feature-discovery.md",
        DOCS_SITE_ROOT / "contributing" / "feature-gap-list.md",
        DOCS_SITE_ROOT / "contributing" / "mission-cadence-m0-m1.md",
        DOCS_SITE_ROOT / "contributing" / "next-steps-canonical.md",
        DOCS_SITE_ROOT / "contributing" / "pmf-dogfood-execution-plan.md",
        DOCS_SITE_ROOT / "contributing" / "pmf-scorecard.md",
        DOCS_SITE_ROOT / "contributing" / "roadmap.md",
        DOCS_SITE_ROOT / "contributing" / "roadmap-intake-register.md",
        DOCS_SITE_ROOT / "contributing" / "strategy-as-bounded-mission-cadence-design.md",
        DOCS_SITE_ROOT / "contributing" / "tw03-rescue-productization-status.md",
        DOCS_SITE_ROOT / "guides" / "conductor-workflow.md",
        DOCS_SITE_ROOT / "guides" / "marketplace.md",
        DOCS_SITE_ROOT / "guides" / "swarm-dogfood-operator.md",
        DOCS_SITE_ROOT / "guides" / "worker-prompt-pack.md",
        DOCS_SITE_ROOT / "enterprise" / "secrets.md",
    ]

    for page in expected_pages:
        assert page.exists(), f"Expected synced docs-site page missing: {page}"


def test_active_execution_links_to_synced_roadmap_intake_register() -> None:
    active = _read_docs_site("contributing/active-execution-issues.md")
    roadmap = _read_docs_site("contributing/roadmap.md")

    assert "[the intake register](./roadmap-intake-register#strategy-mission-queue)" in active
    assert "[Strategy Mission Queue](./roadmap-intake-register#strategy-mission-queue)" in active
    assert "[Roadmap intake register](./roadmap-intake-register)" in roadmap
    assert "ROADMAP_INTAKE_REGISTER.md" not in active
    assert "docs/status/ROADMAP_INTAKE_REGISTER.md" not in roadmap


def test_roadmap_intake_register_links_to_synced_superpowers_pages() -> None:
    content = _read_docs_site("contributing/roadmap-intake-register.md")

    assert (
        "[`docs/superpowers/specs/2026-06-26-strategy-as-bounded-mission-cadence-design.md`]"
        "(./strategy-as-bounded-mission-cadence-design)"
    ) in content
    assert (
        "[`docs/superpowers/plans/2026-06-26-mission-cadence-m0-m1.md`](./mission-cadence-m0-m1)"
    ) in content
    assert "../superpowers/" not in content


def test_synced_plan_preserves_python_f_string_braces_inside_fences() -> None:
    content = _read_docs_site("contributing/mission-cadence-m0-m1.md")

    assert 'f"unexpected columns: {header}"' in content
    assert 'f"missing mission rows: {ids}"' in content
    assert r"\{header\}" not in content
    assert r"\{ids\}" not in content


def test_cli_reference_preserves_generated_catalog_description() -> None:
    content = _read_docs_site("api/cli.md")

    assert "title: Aragora CLI Reference" in content
    assert "description: Generated Aragora CLI command catalog from live parser" in content


def test_disaster_recovery_links_resolve_by_source_directory_not_last_doc_map_entry() -> None:
    # DISASTER_RECOVERY.md is the basename of three DOC_MAP entries (deployment/,
    # runbooks/, enterprise/). A bare "DISASTER_RECOVERY.md" link must resolve
    # relative to the linking source's own directory, not to whichever DOC_MAP
    # entry for that basename happens to be defined last.
    deployment_async = _read_docs_site("deployment/async-gateway.md")
    deployment_volumes = _read_docs_site("deployment/container-volumes.md")
    enterprise_compliance = _read_docs_site("enterprise/compliance.md")
    production_readiness = _read_docs_site("operations/production-readiness.md")
    runbook_backup = _read_docs_site("operations/runbook-backup-automation.md")
    runbook_multi_region = _read_docs_site("operations/runbook-multi-region-setup.md")
    runbook_pg_migration = _read_docs_site("operations/runbook-postgresql-migration.md")
    runbook_pg_replication = _read_docs_site("operations/runbook-postgresql-replication.md")
    disaster_recovery_runbook = _read_docs_site("operations/disaster-recovery-runbook.md")

    # deployment/ siblings resolve to the deployment DR page. Previously these
    # silently mis-resolved to the runbooks DR page (the last DOC_MAP entry
    # sharing the "DISASTER_RECOVERY.md" basename).
    assert "[DISASTER_RECOVERY.md](./disaster-recovery)" in deployment_async
    assert "[DISASTER_RECOVERY.md](./disaster-recovery)" in deployment_volumes

    # PRODUCTION_READINESS.md's source lives in docs/deployment/, so its two bare
    # links are deployment/ sibling references too, not runbooks/ references.
    assert (
        production_readiness.count("[DISASTER_RECOVERY.md](../deployment/disaster-recovery)") == 2
    )

    # COMPLIANCE.md's source lives in docs/enterprise/, so its bare link is a
    # sibling reference to the (newly mapped) enterprise DR overview.
    assert "[DISASTER_RECOVERY.md](./disaster-recovery)" in enterprise_compliance

    # runbooks/ siblings correctly resolve to the runbooks DR page. Pin this down
    # so a future DOC_MAP reordering cannot silently break it the way the
    # deployment/ and enterprise/ links were broken.
    assert "[DISASTER_RECOVERY.md](./disaster-recovery-runbook)" in runbook_backup
    assert "[DISASTER_RECOVERY.md](./disaster-recovery-runbook)" in runbook_multi_region
    assert "[DISASTER_RECOVERY.md](./disaster-recovery-runbook)" in runbook_pg_migration
    assert "[DISASTER_RECOVERY.md](./disaster-recovery-runbook)" in runbook_pg_replication

    # runbooks/DISASTER_RECOVERY.md's own qualified link to its enterprise sibling
    # must resolve to the enterprise DR page, not back to its own runbooks page.
    assert (
        "[../enterprise/DISASTER_RECOVERY.md](../enterprise/disaster-recovery)"
        in disaster_recovery_runbook
    )

    for content in [
        deployment_async,
        deployment_volumes,
        enterprise_compliance,
        production_readiness,
        runbook_backup,
        runbook_multi_region,
        runbook_pg_migration,
        runbook_pg_replication,
        disaster_recovery_runbook,
    ]:
        assert "DISASTER_RECOVERY.md)" not in content


def test_enterprise_disaster_recovery_is_mapped_and_public_safe() -> None:
    # docs/enterprise/DISASTER_RECOVERY.md is a substantial standalone enterprise DR
    # overview that docs/runbooks/DISASTER_RECOVERY.md links to. The docs-site
    # route must stay valid without publishing the internal runbook body.
    page = DOCS_SITE_ROOT / "enterprise" / "disaster-recovery.md"
    assert page.exists(), f"Expected synced docs-site page missing: {page}"

    content = page.read_text(encoding="utf-8")
    assert "title: Enterprise Disaster Recovery Overview" in content
    assert "Recovery Objectives" in content
    assert "Operational response details are restricted" in content
    assert "Classification: Internal" not in content
    assert "Primary Region (us-east-1)" not in content
    assert "s3://aragora-backups" not in content
    assert "kubectl --context backup" not in content
    assert "Incident commander" not in content


def test_ambiguous_readme_basename_links_resolve_to_valid_targets() -> None:
    # README.md is also a multi-way-ambiguous basename (case-studies/README.md and
    # ADR/README.md both map to it), and several other source docs link to
    # non-docs-site README.md files (e.g. aragora/mcp/README.md, deploy/README.md)
    # that are intentionally outside DOC_MAP. The resolver must still fail closed
    # instead of guessing the ADR README, but known repo docs and docs-site pages
    # should rewrite to stable valid destinations instead of leaving broken
    # source-relative links in relocated generated docs.
    reference = _read_docs_site("api/reference.md")
    status = _read_docs_site("contributing/status.md")
    extended_readme = _read_docs_site("contributing/extended-readme.md")
    sdk_consolidation = _read_docs_site("guides/sdk-consolidation.md")
    sdk_quickstart = _read_docs_site("guides/sdk-quickstart.md")
    eu_ai_act_guide = _read_docs_site("security/eu-ai-act-guide.md")

    assert (
        "[MCP README]"
        "(https://github.com/synaptent/aragora/blob/main/aragora/mcp/README.md)" in reference
    )
    assert "[README](https://github.com/synaptent/aragora/blob/main/README.md)" in status
    assert "[README](https://github.com/synaptent/aragora/blob/main/README.md)" in extended_readme
    assert (
        "[algorithms/README.md]"
        "(https://github.com/synaptent/aragora/blob/main/docs/algorithms/README.md)"
        in extended_readme
    )
    assert (
        "[sdk/typescript/README.md]"
        "(https://github.com/synaptent/aragora/blob/main/sdk/typescript/README.md)"
        in sdk_consolidation
    )
    assert (
        "[`deploy/README.md`]"
        "(https://github.com/synaptent/aragora/blob/main/deploy/README.md)" in sdk_quickstart
    )
    assert "[Gauntlet Testing](../guides/gauntlet)" in eu_ai_act_guide

    for content in [
        reference,
        status,
        extended_readme,
        sdk_consolidation,
        sdk_quickstart,
        eu_ai_act_guide,
    ]:
        assert "analysis/adr" not in content
        for target in [
            "../../aragora/mcp/README.md",
            "../README.md",
            "algorithms/README.md",
            "../deploy/README.md",
            "../../aragora/gauntlet/README.md",
        ]:
            assert f"]({target})" not in content
