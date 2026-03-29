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
    assert manifest["name"] == "Aragora Adversarial Review"
    assert "adversarial review" in manifest["description"]
    assert "popup" in manifest["description"]
    assert manifest["background"]["service_worker"] == "background.js"
    assert manifest["action"]["default_title"] == "Aragora"
    assert manifest["action"]["default_popup"] == "popup.html"
    assert {"activeTab", "contextMenus", "storage"}.issubset(set(manifest["permissions"]))
    assert "https://*/*" in manifest["host_permissions"]
    assert "http://localhost/*" in manifest["host_permissions"]
    assert "http://127.0.0.1/*" in manifest["host_permissions"]

    content_script = manifest["content_scripts"][0]
    assert content_script["matches"] == ["<all_urls>"]
    assert content_script["js"] == ["content.js"]


def test_background_script_handles_context_menu_selection_and_api_submission() -> None:
    background_script = _read_file("background.js")

    assert 'const MENU_ID = "aragora-send-selection"' in background_script
    assert 'title: "Send selection to Aragora"' in background_script
    assert 'contexts: ["selection"]' in background_script
    assert "chrome.contextMenus.onClicked.addListener" in background_script
    assert "chrome.tabs.sendMessage" in background_script
    assert '"aragora:get-selection"' in background_script
    assert "chrome.storage.local.set" in background_script
    assert "/api/v2/debates" in background_script
    assert 'Authorization: `Bearer ${String(settings.apiKey || "").trim()}`' in background_script
    assert 'source: "browser_extension_context_menu"' in background_script
    assert 'source_title: source.pageTitle || ""' in background_script
    assert 'source_url: source.pageUrl || ""' in background_script
    assert 'status: "submitting"' in background_script
    assert 'status: "error"' in background_script
    assert 'status: createdDebate.status || "running"' in background_script
    assert 'error: "Add an Aragora API key in the popup before sending text."' in background_script


def test_popup_assets_render_saved_selection_and_latest_result() -> None:
    popup_html = _read_file("popup.html")
    popup_js = _read_file("popup.js")
    content_script = _read_file("content.js")

    assert "Adversarial Review" in popup_html
    assert 'id="status-pill"' in popup_html
    assert 'id="selection-preview"' in popup_html
    assert 'id="source-link"' in popup_html
    assert 'id="refresh-result"' in popup_html
    assert 'id="api-url"' in popup_html
    assert 'id="api-key"' in popup_html
    assert 'id="agents"' in popup_html
    assert 'id="rounds"' in popup_html
    assert 'id="debate-id"' in popup_html
    assert 'id="result-status"' in popup_html
    assert 'id="result-confidence"' in popup_html
    assert 'id="result-answer"' in popup_html
    assert 'id="result-error"' in popup_html
    assert "Latest review" in popup_html
    assert 'id="save-settings"' in popup_html

    assert "chrome.storage.onChanged.addListener" in popup_js
    assert "window.setInterval" in popup_js
    assert "resolveFinalAnswer" in popup_js
    assert "humanizeStatus" in popup_js
    assert "buildResultState" in popup_js
    assert "result?.answer" in popup_js
    assert "result?.consensus?.summary" in popup_js
    assert "result?.consensus?.final_answer" in popup_js
    assert "finalAnswer" in popup_js
    assert "fetch(`${normalizeApiUrl(apiUrl)}/api/v2/debates/${debateId}`" in popup_js
    assert "result?.final_answer" in popup_js
    assert "result?.finalAnswer" in popup_js
    assert "elements.resultConfidence.textContent =" in popup_js
    assert "elements.resultAnswer.textContent = answer" in popup_js
    assert "elements.selectionPreview.textContent =" in popup_js

    assert '"aragora:get-selection"' in content_script
    assert '"selectionchange"' in content_script
    assert '"contextmenu"' in content_script
    assert "MAX_SELECTION_LENGTH = 9000" in content_script
    assert "window.__aragoraSelection = buildSelectionPayload()" in content_script
