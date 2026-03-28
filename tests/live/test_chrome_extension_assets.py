from __future__ import annotations

import json
from pathlib import Path


EXTENSION_DIR = Path("aragora/live/public/chrome-extension")


def test_extension_bundle_files_exist() -> None:
    expected_files = {
        "manifest.json",
        "popup.html",
        "popup.css",
        "popup.js",
    }

    assert EXTENSION_DIR.exists()
    assert expected_files.issubset(
        {path.name for path in EXTENSION_DIR.iterdir() if path.is_file()}
    )


def test_manifest_declares_popup_background_permissions_and_hosts() -> None:
    manifest_path = EXTENSION_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["manifest_version"] == 3
    assert manifest["background"]["service_worker"] == "popup.js"
    assert manifest["action"]["default_popup"] == "popup.html"

    permissions = set(manifest["permissions"])
    assert {"activeTab", "contextMenus", "storage"}.issubset(permissions)

    host_permissions = set(manifest["host_permissions"])
    assert "https://api.aragora.ai/*" in host_permissions
    assert "http://localhost/*" in host_permissions


def test_popup_assets_reference_review_flow_and_gauntlet_endpoints() -> None:
    popup_html = (EXTENSION_DIR / "popup.html").read_text()
    popup_js = (EXTENSION_DIR / "popup.js").read_text()

    assert "Send Selection to Aragora Review" in popup_html
    assert 'id="settings-form"' in popup_html
    assert 'id="findings-list"' in popup_html
    assert '<script src="popup.js"></script>' in popup_html

    assert "chrome.contextMenus.create" in popup_js
    assert "/api/gauntlet/run" in popup_js
    assert "/api/v1/gauntlet/" in popup_js
    assert "/api/gauntlet/" in popup_js
    assert "Authorization" in popup_js
    assert "vulnerability_details" in popup_js
