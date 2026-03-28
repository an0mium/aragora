importScripts("api-client.js");

const MENU_ID = "aragora-send-selection";
const STATE_KEY = "aragoraPopupState";
const DEFAULT_SETTINGS = {
  apiUrl: "https://api.aragora.ai",
  apiKey: "",
  agents: "",
  rounds: 3,
  consensus: "majority",
};
const QUESTION_LIMIT = 5000;
const SELECTION_LIMIT = 9000;
const TERMINAL_STATUSES = new Set(["completed", "consensus_reached", "done"]);
const ERROR_STATUSES = new Set(["failed", "error", "cancelled"]);
const normalizeApiUrl = AragoraExtensionApi.normalizeApiUrl;
const readErrorMessage = AragoraExtensionApi.readErrorMessage;
const buildAdversarialReviewPayload = AragoraExtensionApi.buildAdversarialReviewPayload;
const buildDebateSnapshot = AragoraExtensionApi.buildDebateSnapshot;

function registerContextMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create(
      {
        id: MENU_ID,
        title: "Send selection to Aragora",
        contexts: ["selection"],
      },
      () => {
        if (chrome.runtime.lastError) {
          console.warn("Failed to register Aragora context menu:", chrome.runtime.lastError.message);
        }
      }
    );
  });
}

async function ensureDefaultSettings() {
  const settings = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  await chrome.storage.sync.set({ ...DEFAULT_SETTINGS, ...settings });
}

function sanitizeSelectionText(value) {
  return String(value || "")
    .replace(/\u0000/g, "")
    .trim()
    .slice(0, SELECTION_LIMIT);
}

async function setBadge(text, color) {
  await chrome.action.setBadgeText({ text });

  if (color) {
    await chrome.action.setBadgeBackgroundColor({ color });
  }
}

async function writePopupState(nextState) {
  const current = await chrome.storage.local.get(STATE_KEY);
  const mergedState = {
    ...(current[STATE_KEY] || {}),
    ...nextState,
    updatedAt: new Date().toISOString(),
  };

  await chrome.storage.local.set({ [STATE_KEY]: mergedState });
  return mergedState;
}

async function getSelectionFromContentScript(tabId) {
  if (typeof tabId !== "number") {
    return null;
  }

  try {
    return await chrome.tabs.sendMessage(tabId, { type: "aragora:get-selection" });
  } catch (error) {
    console.warn("Could not read selection from content script:", error);
    return null;
  }
}

async function getSettings() {
  return chrome.storage.sync.get(DEFAULT_SETTINGS);
}

async function createDebate(selectionText, source, settings) {
  const apiUrl = normalizeApiUrl(settings.apiUrl);
  const response = await fetch(`${apiUrl}/api/v2/debates`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${String(settings.apiKey || "").trim()}`,
    },
    body: JSON.stringify(buildAdversarialReviewPayload(selectionText, source, settings)),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

function buildStoredResult(snapshot) {
  return {
    debateId: snapshot.debateId,
    status: snapshot.status,
    finalAnswer: snapshot.finalAnswer || "",
    message: snapshot.message || null,
    confidence: snapshot.confidence,
    task: snapshot.task || "",
  };
}

async function applyBadgeForSnapshot(snapshot) {
  if (ERROR_STATUSES.has(snapshot.status)) {
    await setBadge("ERR", "#b42318");
    return;
  }

  if (TERMINAL_STATUSES.has(snapshot.status) || snapshot.finalAnswer) {
    await setBadge("OK", "#0f766e");
    return;
  }

  await setBadge("RUN", "#1d4ed8");
}

async function handleContextMenuClick(info, tab) {
  let selectionText = sanitizeSelectionText(info.selectionText);
  const source = {
    pageTitle: tab?.title || "",
    pageUrl: info.pageUrl || tab?.url || "",
  };

  if (!selectionText) {
    const contentSelection = await getSelectionFromContentScript(tab?.id);
    selectionText = sanitizeSelectionText(contentSelection?.selectedText);

    if (contentSelection?.pageTitle && !source.pageTitle) {
      source.pageTitle = contentSelection.pageTitle;
    }

    if (contentSelection?.pageUrl && !source.pageUrl) {
      source.pageUrl = contentSelection.pageUrl;
    }
  }

  if (!selectionText) {
    await writePopupState({
      status: "error",
      error: "No selected text was available to send.",
      debateId: null,
      result: null,
      selectionText: "",
      source,
    });
    await setBadge("ERR", "#b42318");
    return;
  }

  const settings = await getSettings();
  if (!String(settings.apiKey || "").trim()) {
    await writePopupState({
      status: "error",
      error: "Add an Aragora API key in the popup before sending text.",
      debateId: null,
      result: null,
      selectionText,
      source,
    });
    await setBadge("ERR", "#b42318");
    return;
  }

  await writePopupState({
    status: "submitting",
    error: null,
    debateId: null,
    result: null,
    selectionText,
    source,
    submittedAt: new Date().toISOString(),
  });
  await setBadge("...", "#0f766e");

  try {
    const createdDebate = await createDebate(selectionText, source, settings);
    const snapshot = buildDebateSnapshot(createdDebate, "running");
    const debateId = snapshot.debateId;

    if (!debateId) {
      throw new Error("Aragora did not return a debate ID.");
    }

    await writePopupState({
      status: snapshot.status,
      debateId,
      error: null,
      result: buildStoredResult(snapshot),
      selectionText,
      source,
    });
    await applyBadgeForSnapshot(snapshot);
  } catch (error) {
    await writePopupState({
      status: "error",
      debateId: null,
      result: null,
      error: error instanceof Error ? error.message : String(error),
      selectionText,
      source,
    });
    await setBadge("ERR", "#b42318");
  }
}

chrome.runtime.onInstalled.addListener(() => {
  registerContextMenu();
  void ensureDefaultSettings();
  void setBadge("", null);
});

chrome.runtime.onStartup.addListener(() => {
  registerContextMenu();
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== MENU_ID) {
    return;
  }

  void handleContextMenuClick(info, tab);
});
