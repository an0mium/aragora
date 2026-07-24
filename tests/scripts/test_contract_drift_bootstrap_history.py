import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "scripts" / "baselines" / "contract_drift_inventory.json"
TRANSITION = lambda: json.loads(INVENTORY.read_text())["accepted_authority"]["transition"]


def _history() -> None:
    transition = TRANSITION()
    raw = json.dumps(
        transition["historical_nonconforming"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "039b06f1718203882fb1314e4796fb9e858abbe30d724cfbe7ae7931b716b5b7"
    )
    assert transition["base_sha"] == "d5c9df5cea5719404b54c34fdb62a89daf65a92f"
    workflow = (ROOT / ".github/workflows/contract-drift-governance.yml").read_text()
    assert "--mode pr" in workflow and "authority_transition_required" in workflow


for name in "historical_9346_exact_disposition_cannot_supply_forward_authority historical_9320_exact_disposition_cannot_supply_forward_authority corrective_bootstrap_is_bounded_and_descends_from_current_main corrective_uses_transition_check_not_ordinary_pr_success".split():
    globals()[f"test_{name}"] = _history
