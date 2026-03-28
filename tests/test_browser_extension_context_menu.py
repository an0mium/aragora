from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = REPO_ROOT / "aragora" / "live" / "browser-extension"


def _read_file(name: str) -> str:
    return (EXTENSION_DIR / name).read_text(encoding="utf-8")


def test_extension_bundle_files_exist() -> None:
    expected_files = {
        "manifest.json",
        "background.js",
        "content.js",
        "popup.html",
        "popup.css",
        "popup.js",
    }

    assert EXTENSION_DIR.exists()
    assert expected_files.issubset({path.name for path in EXTENSION_DIR.iterdir()})


def test_manifest_registers_popup_service_worker_and_content_script() -> None:
    manifest = json.loads(_read_file("manifest.json"))

    assert manifest["manifest_version"] == 3
    assert manifest["background"]["service_worker"] == "background.js"
    assert manifest["action"]["default_popup"] == "popup.html"
    assert {"activeTab", "contextMenus", "storage"}.issubset(set(manifest["permissions"]))
    assert "https://*/*" in manifest["host_permissions"]

    content_script = manifest["content_scripts"][0]
    assert content_script["matches"] == ["<all_urls>"]
    assert content_script["js"] == ["content.js"]


def test_background_script_handles_context_menu_selection_and_api_submission() -> None:
    background_script = _read_file("background.js")

    assert "chrome.contextMenus.onClicked.addListener" in background_script
    assert "chrome.tabs.sendMessage" in background_script
    assert '"aragora:get-selection"' in background_script
    assert "chrome.storage.local.set" in background_script
    assert "/api/v2/debates" in background_script
    assert 'Authorization: `Bearer ${String(settings.apiKey || "").trim()}`' in background_script


def test_popup_assets_render_saved_selection_and_latest_result() -> None:
    popup_html = _read_file("popup.html")
    popup_js = _read_file("popup.js")
    content_script = _read_file("content.js")

    assert 'id="selection-preview"' in popup_html
    assert 'id="debate-id"' in popup_html
    assert 'id="result-answer"' in popup_html
    assert 'id="save-settings"' in popup_html

    assert "chrome.storage.onChanged.addListener" in popup_js
    assert "window.setInterval" in popup_js
    assert "finalAnswer" in popup_js
    assert "fetch(`${normalizeApiUrl(apiUrl)}/api/v2/debates/${debateId}`" in popup_js

    assert '"aragora:get-selection"' in content_script
    assert '"selectionchange"' in content_script
