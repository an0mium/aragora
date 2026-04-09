from __future__ import annotations

from aragora.swarm.campaign import (
    CampaignProjectStatus,
    load_campaign_manifest,
    save_campaign_manifest,
)
from aragora.swarm.initiative_campaign_bridge import sync_campaign_manifest_for_initiative
from aragora.swarm.initiative_models import InitiativeMilestone, InitiativeRecord, InitiativeSlice
from aragora.swarm.initiative_store import InitiativeStore


def _initiative() -> InitiativeRecord:
    return InitiativeRecord(
        initiative_id="bridge-initiative",
        title="Bridge initiative",
        goal="Keep initiative orchestration aligned with campaign execution.",
        rationale="Initiative CLI should materialize a runnable campaign manifest.",
        validations=["python3 -m pytest tests/swarm/test_initiative_campaign_bridge.py -q"],
        feature_flag_name="initiative_bridge",
        milestones=[
            InitiativeMilestone(
                milestone_id="m1",
                title="Execution lane",
                slice_ids=["slice-1", "slice-2"],
            )
        ],
        slices=[
            InitiativeSlice(
                slice_id="slice-1",
                title="Materialize manifest",
                description="Persist a campaign manifest for the initiative.",
                file_scope=["aragora/swarm/initiative_campaign_bridge.py"],
                acceptance_criteria=["campaign manifest is created"],
                validations=["python3 -m pytest tests/swarm/test_initiative_campaign_bridge.py -q"],
                estimated_complexity="small",
                status="active",
            ),
            InitiativeSlice(
                slice_id="slice-2",
                title="Wire CLI",
                description="Resolve manifests by initiative id.",
                dependencies=["slice-1"],
                file_scope=["aragora/cli/commands/swarm.py"],
                validations=["python3 -m pytest tests/swarm/test_initiative_integrator.py -q"],
            ),
        ],
    )


def test_sync_campaign_manifest_materializes_projects_from_initiative(tmp_path) -> None:
    store = InitiativeStore(state_dir=tmp_path)
    initiative = _initiative()
    store.save(initiative)

    manifest_path = sync_campaign_manifest_for_initiative(store, initiative)
    manifest = load_campaign_manifest(manifest_path)
    projects = manifest.project_map()

    assert manifest.campaign_id == initiative.initiative_id
    assert manifest.source_ref == str(store.path_for(initiative.initiative_id))
    assert set(projects) == {"slice-1", "slice-2"}
    assert projects["slice-1"].status == CampaignProjectStatus.ACTIVE.value
    assert projects["slice-1"].milestone == "Execution lane"
    assert projects["slice-1"].feature_flag == "initiative_bridge"
    assert projects["slice-1"].feature_flag_required is True
    assert projects["slice-2"].dependencies[0].project_id == "slice-1"
    assert "campaign manifest is created" in projects["slice-1"].spec.acceptance_criteria
    assert (
        "python3 -m pytest tests/swarm/test_initiative_campaign_bridge.py -q"
        in projects["slice-1"].spec.acceptance_criteria
    )


def test_sync_campaign_manifest_preserves_existing_runtime_state(tmp_path) -> None:
    store = InitiativeStore(state_dir=tmp_path)
    initiative = _initiative()
    store.save(initiative)

    manifest_path = sync_campaign_manifest_for_initiative(store, initiative)
    manifest = load_campaign_manifest(manifest_path)
    manifest.project_map()["slice-1"].pr_url = "https://example.com/pr/123"
    manifest.project_map()["slice-1"].retry_count = 2
    manifest.execution_state.total_cost_usd = 7.5
    save_campaign_manifest(manifest_path, manifest)

    refreshed_path = sync_campaign_manifest_for_initiative(store, initiative)
    refreshed = load_campaign_manifest(refreshed_path)

    assert refreshed.project_map()["slice-1"].pr_url == "https://example.com/pr/123"
    assert refreshed.project_map()["slice-1"].retry_count == 2
    assert refreshed.execution_state.total_cost_usd == 7.5
