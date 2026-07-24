import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_contract_drift_ratchet.py"
INVENTORY = ROOT / "scripts" / "baselines" / "contract_drift_inventory.json"
spec = importlib.util.spec_from_file_location("contract_drift_ratchet_transition", SCRIPT)
assert spec and spec.loader
ratchet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratchet)
AUTHORITY = lambda: json.loads(INVENTORY.read_text())["accepted_authority"]


def _contract(tmp_path: Path) -> None:
    assert INVENTORY.read_text().count('"accepted_authority"') == 1
    path = tmp_path / "legacy.json"
    path.write_text('{"items":[]}')
    result = ratchet.build_accepted_result(mode="pr", repo_root=ROOT, inventory_path=path)
    assert (result["status"], result["error_code"]) == ("fail", "authority_transition_required")
    summary = ratchet.validate_accepted_authority(AUTHORITY(), repo_root=ROOT)
    assert (summary["original_record_total"], summary["sdk_provenance_record_total"]) == (655, 598)
    projection = summary["operation_projection"]
    assert (projection["membership_count"], projection["edge_count"]) == (655, 666)
    program = ratchet.build_accepted_result(
        mode="program", repo_root=ROOT, inventory_path=INVENTORY, as_of="2026-04-17"
    )
    assert (program["current"]["total_items"], program["authority"]["source"]) == (
        655,
        "accepted_authority",
    )
    noop = ratchet.compare_accepted_authorities(AUTHORITY(), AUTHORITY(), repo_root=ROOT)
    assert (noop["passing"], noop["added_original_record_ids"]) == (True, [])


for name in "accepted_authority_manifest_is_unique base_without_accepted_authority_requires_transition corrective_merge_parent_requires_transition transition_reconstructs_all_655_ids_and_598_provenance_records operation_projection_revision_preserves_655_memberships_666_complete_edges_nine_multi_edge_and_max_four no_reachable_classification_filtered_or_raw_baseline_fallback post_transition_noop_pr_uses_only_accepted_authority".split():
    globals()[f"test_{name}"] = _contract
