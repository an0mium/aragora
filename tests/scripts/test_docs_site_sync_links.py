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


def test_root_readme_link_resolves_to_its_own_page_not_adr() -> None:
    # docs/EXTENDED_README.md links to the repo-root README via "../README.md" three
    # times. Basename "README.md" is also used by ADR/README.md and
    # case-studies/README.md, so this guards against the rewriter's basename
    # fallback silently resolving to whichever of those is defined last in DOC_MAP.
    content = _read_docs_site("contributing/extended-readme.md")

    assert "[README](./readme)" in content
    assert "analysis/adr" not in content

    # A link with no DOC_MAP entry of its own ("algorithms/README.md") must be left
    # unrewritten rather than guessing via the ambiguous "README.md" basename.
    assert "[algorithms/README.md](algorithms/README.md)" in content


def test_root_readme_synced_with_proof_ladder_anchor() -> None:
    content = _read_docs_site("contributing/readme.md")

    assert "title: Aragora" in content
    assert '<a id="proof-ladder"></a>' in content


def test_root_readme_links_resolve_to_docs_site_pages_or_external_targets() -> None:
    content = _read_docs_site("contributing/readme.md")

    expected_links = [
        "[Quickstart](../getting-started/quickstart)",
        "[Cold Reviewer Guide](./cold-reviewer-guide)",
        "[Open Decision Receipt spec](./open-decision-receipt)",
        "[Boundaries and Scope](./boundaries-and-scope)",
        "[`docs/METRICS.md`](./metrics)",
        "[`docs/HONEST_ASSESSMENT.md`](./honest-assessment)",
        "[GA checklist](./ga-checklist)",
        "[CLI Reference](../api/cli)",
        "[SDK Guide](../guides/sdk)",
        "[API Reference](../api/reference)",
        "[Inspiration and credits](./credits)",
        "[LICENSE](https://github.com/synaptent/aragora/blob/main/LICENSE)",
    ]
    for link in expected_links:
        assert link in content

    unresolved_source_links = [
        "](docs/quickstart.md)",
        "](docs/COLD_REVIEWER_GUIDE.md)",
        "](docs/specs/OPEN_DECISION_RECEIPT.md)",
        "](docs/reference/CREDITS.md)",
        "](docs/strategy/BOUNDARIES_AND_SCOPE.md)",
        "](docs/METRICS.md)",
        "](docs/HONEST_ASSESSMENT.md)",
        "](docs/GA_CHECKLIST.md)",
        "](LICENSE)",
    ]
    for link in unresolved_source_links:
        assert link not in content


def test_public_entrypoint_source_relative_spec_links_resolve() -> None:
    cold_reviewer = _read_docs_site("contributing/cold-reviewer-guide.md")
    odr = _read_docs_site("contributing/open-decision-receipt.md")
    docs_index = _read_docs_site("contributing/documentation-index.md")

    assert "[Supported API Surface](../api/supported-surface)" in cold_reviewer
    assert "[`docs/specs/TAMPER_EVIDENT_TRAIL.md`](./tamper-evident-trail)" in odr
    assert "[`TAMPER_EVIDENT_TRAIL.md`](./tamper-evident-trail)" in odr
    assert "[`odr-native-mapping.md`](./odr-native-mapping)" in odr
    assert "[Supported API Surface](../api/supported-surface)" in docs_index

    for content in [cold_reviewer, odr, docs_index]:
        assert "api/SUPPORTED_SURFACE.md" not in content
        assert "TAMPER_EVIDENT_TRAIL.md)" not in content
        assert "odr-native-mapping.md)" not in content


def test_source_relative_disaster_recovery_links_resolve_by_source_directory() -> None:
    deployment = _read_docs_site("deployment/async-gateway.md")
    operations = _read_docs_site("operations/runbook-backup-automation.md")
    enterprise = _read_docs_site("enterprise/compliance.md")
    runbook = _read_docs_site("operations/disaster-recovery-runbook.md")

    assert "[DISASTER_RECOVERY.md](./disaster-recovery)" in deployment
    assert "[DISASTER_RECOVERY.md](./disaster-recovery-runbook)" in operations
    assert "[DISASTER_RECOVERY.md](./disaster-recovery)" in enterprise
    assert "[../enterprise/DISASTER_RECOVERY.md](../enterprise/disaster-recovery)" in runbook

    for content in [deployment, operations, enterprise, runbook]:
        assert "DISASTER_RECOVERY.md)" not in content


def test_source_code_and_deploy_links_resolve_to_served_docs_pages() -> None:
    api_reference = _read_docs_site("api/reference.md")
    eu_ai_act = _read_docs_site("security/eu-ai-act-guide.md")
    sdk_quickstart = _read_docs_site("guides/sdk-quickstart.md")

    assert "[MCP README](../guides/mcp-integration)" in api_reference
    assert "[Gauntlet Testing](../guides/gauntlet)" in eu_ai_act
    assert "[`deploy/README.md`](../deployment/docker)" in sdk_quickstart

    assert "../../aragora/mcp/README.md" not in api_reference
    assert "../../aragora/gauntlet/README.md" not in eu_ai_act
    assert "../deploy/README.md" not in sdk_quickstart
