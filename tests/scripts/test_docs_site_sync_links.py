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


def test_docs_site_sync_creates_linked_status_and_planning_pages() -> None:
    expected_pages = [
        DOCS_SITE_ROOT / "contributing" / "2026-03-26-pmf-14-day-execution-plan.md",
        DOCS_SITE_ROOT / "contributing" / "active-execution-issues.md",
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
        DOCS_SITE_ROOT / "contributing" / "next-steps-canonical.md",
        DOCS_SITE_ROOT / "contributing" / "pmf-dogfood-execution-plan.md",
        DOCS_SITE_ROOT / "contributing" / "pmf-scorecard.md",
        DOCS_SITE_ROOT / "contributing" / "roadmap.md",
        DOCS_SITE_ROOT / "guides" / "conductor-workflow.md",
        DOCS_SITE_ROOT / "guides" / "marketplace.md",
        DOCS_SITE_ROOT / "guides" / "swarm-dogfood-operator.md",
        DOCS_SITE_ROOT / "guides" / "worker-prompt-pack.md",
        DOCS_SITE_ROOT / "enterprise" / "secrets.md",
    ]

    for page in expected_pages:
        assert page.exists(), f"Expected synced docs-site page missing: {page}"
