from __future__ import annotations

import scripts.reconcile_status_docs as reconcile_status_docs


def test_feature_discovery_mirror_check_ignores_banner_and_relative_link_prefixes(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "docs" / "status" / "FEATURE_DISCOVERY.md"
    mirror = tmp_path / "docs" / "FEATURE_DISCOVERY.md"
    canonical.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)

    canonical.write_text(
        "# Aragora Feature Discovery Guide\n\n"
        "*Complete catalog of 230+ features for developers exploring Aragora capabilities*\n\n"
        "See [API_REFERENCE.md](../api/API_REFERENCE.md)\n\n"
        "## User Participation\n\n"
        "| **Spectate** | Partial | Buffered snapshot only |\n",
        encoding="utf-8",
    )
    mirror.write_text(
        "# Aragora Feature Discovery Guide\n\n"
        "*Complete catalog of 230+ features for developers exploring Aragora capabilities*\n\n"
        "> Compatibility mirror for older links. The canonical current-state inventory lives at "
        "[status/FEATURE_DISCOVERY.md](status/FEATURE_DISCOVERY.md).\n\n"
        "See [API_REFERENCE.md](./api/API_REFERENCE.md)\n\n"
        "## User Participation\n\n"
        "| **Spectate** | Partial | Buffered snapshot only |\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(reconcile_status_docs, "FEATURE_DISCOVERY", mirror)
    monkeypatch.setattr(reconcile_status_docs, "FEATURE_DISCOVERY_STATUS", canonical)

    assert reconcile_status_docs._check_feature_discovery_mirror_drift() == []


def test_feature_discovery_mirror_check_reports_substantive_drift(tmp_path, monkeypatch) -> None:
    canonical = tmp_path / "docs" / "status" / "FEATURE_DISCOVERY.md"
    mirror = tmp_path / "docs" / "FEATURE_DISCOVERY.md"
    canonical.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)

    canonical.write_text(
        "# Aragora Feature Discovery Guide\n\n"
        "## User Participation\n\n"
        "| **Spectate** | Partial | Buffered snapshot only |\n",
        encoding="utf-8",
    )
    mirror.write_text(
        "# Aragora Feature Discovery Guide\n\n"
        "> Compatibility mirror for older links. The canonical current-state inventory lives at "
        "[status/FEATURE_DISCOVERY.md](status/FEATURE_DISCOVERY.md).\n\n"
        "## User Participation\n\n"
        "| **Spectate** | Stable | Full live streaming |\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(reconcile_status_docs, "FEATURE_DISCOVERY", mirror)
    monkeypatch.setattr(reconcile_status_docs, "FEATURE_DISCOVERY_STATUS", canonical)

    findings = reconcile_status_docs._check_feature_discovery_mirror_drift()

    assert len(findings) == 1
    finding = findings[0]
    assert finding["severity"] == "critical"
    assert finding["source"] == "FEATURE_DISCOVERY.md"
    assert "status/FEATURE_DISCOVERY.md" in finding["message"]
    assert "first differing normalized line" in finding["message"]
