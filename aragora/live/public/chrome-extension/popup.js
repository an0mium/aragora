(function () {
  const MENU_ID = "aragora-send-selection-review";
  const SETTINGS_KEY = "aragoraExtensionSettings";
  const REVIEW_KEY = "aragoraLatestReview";
  const DEFAULT_SETTINGS = {
    apiUrl: "https://api.aragora.ai",
    apiToken: "",
    persona: "security",
    profile: "quick",
  };
  const DEFAULT_REVIEW = {
    status: "idle",
    message: 'Right-click selected text and choose "Send Selection to Aragora Review".',
    findings: [],
    riskSummary: { critical: 0, high: 0, medium: 0, low: 0, total: 0 },
  };
  const MAX_POLL_ATTEMPTS = 24;
  const POLL_INTERVAL_MS = 2500;
  const POPUP_CONTEXT = typeof document !== "undefined";
  const EXTENSION_CONTEXT = typeof chrome !== "undefined" && Boolean(chrome.storage);

  function getStorage(areaName) {
    return chrome.storage[areaName];
  }

  function storageGet(areaName, defaults) {
    return new Promise((resolve, reject) => {
      getStorage(areaName).get(defaults, (result) => {
        if (chrome.runtime && chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve(result);
      });
    });
  }

  function storageSet(areaName, value) {
    return new Promise((resolve, reject) => {
      getStorage(areaName).set(value, () => {
        if (chrome.runtime && chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve();
      });
    });
  }

  function delay(ms) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }

  function normalizeApiUrl(url) {
    return (url || DEFAULT_SETTINGS.apiUrl).trim().replace(/\/+$/, "");
  }

  async function getSettings() {
    const stored = await storageGet("sync", { [SETTINGS_KEY]: DEFAULT_SETTINGS });
    return {
      ...DEFAULT_SETTINGS,
      ...(stored[SETTINGS_KEY] || {}),
      apiUrl: normalizeApiUrl((stored[SETTINGS_KEY] || {}).apiUrl),
    };
  }

  async function saveSettings(nextSettings) {
    const merged = {
      ...DEFAULT_SETTINGS,
      ...nextSettings,
      apiUrl: normalizeApiUrl(nextSettings.apiUrl),
      apiToken: (nextSettings.apiToken || "").trim(),
    };
    await storageSet("sync", { [SETTINGS_KEY]: merged });
    return merged;
  }

  async function getLatestReview() {
    const stored = await storageGet("local", { [REVIEW_KEY]: DEFAULT_REVIEW });
    return {
      ...DEFAULT_REVIEW,
      ...(stored[REVIEW_KEY] || {}),
    };
  }

  async function updateLatestReview(patch) {
    const current = await getLatestReview();
    const nextReview = {
      ...current,
      ...patch,
      updatedAt: new Date().toISOString(),
    };
    await storageSet("local", { [REVIEW_KEY]: nextReview });
    return nextReview;
  }

  async function clearLatestReview() {
    await storageSet("local", { [REVIEW_KEY]: { ...DEFAULT_REVIEW, updatedAt: null } });
  }

  function extractErrorMessage(payload, fallbackMessage) {
    if (!payload) {
      return fallbackMessage;
    }

    if (typeof payload === "string") {
      return payload;
    }

    if (payload.error) {
      return payload.error;
    }

    if (payload.message) {
      return payload.message;
    }

    if (payload.detail) {
      return payload.detail;
    }

    return fallbackMessage;
  }

  async function aragoraRequest(settings, path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("Accept", "application/json");
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    if (settings.apiToken) {
      headers.set("Authorization", `Bearer ${settings.apiToken}`);
    }

    const response = await fetch(`${normalizeApiUrl(settings.apiUrl)}${path}`, {
      ...options,
      headers,
    });
    const rawText = await response.text();
    let payload = null;

    if (rawText) {
      try {
        payload = JSON.parse(rawText);
      } catch (_error) {
        payload = rawText;
      }
    }

    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, `${response.status} ${response.statusText}`));
    }

    return payload || {};
  }

  function setBadge(text, color) {
    if (!chrome.action) {
      return;
    }

    chrome.action.setBadgeText({ text });
    chrome.action.setBadgeBackgroundColor({ color });
  }

  function truncateText(value, maxLength) {
    if (!value) {
      return "";
    }
    return value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;
  }

  function getSourceLabel(source) {
    if (!source) {
      return "No page captured";
    }

    if (source.pageTitle && source.pageUrl) {
      return `${source.pageTitle} · ${source.pageUrl}`;
    }

    return source.pageUrl || source.pageTitle || "No page captured";
  }

  function normalizeFinding(finding, index) {
    const severity = String(
      finding.severity_level || finding.severity || "LOW"
    ).toUpperCase();
    return {
      id: finding.id || `${finding.category || "finding"}-${index}`,
      title: finding.title || finding.description || `Finding ${index + 1}`,
      description: finding.description || "No description provided.",
      severity,
      category: finding.category || "general",
    };
  }

  function normalizeReceipt(receipt, source, fallbackSummary) {
    const findings = Array.isArray(receipt.vulnerability_details)
      ? receipt.vulnerability_details.map(normalizeFinding)
      : Array.isArray(receipt.findings)
        ? receipt.findings.map(normalizeFinding)
        : [];
    const riskSummary = receipt.risk_summary || {};
    const totalFindings = Number(riskSummary.total || findings.length || 0);

    return {
      status: "completed",
      gauntletId: receipt.gauntlet_id || null,
      receiptId: receipt.receipt_id || null,
      verdict: String(receipt.verdict || "UNKNOWN").toUpperCase(),
      confidence: receipt.confidence ?? null,
      robustnessScore: receipt.robustness_score ?? null,
      message: totalFindings
        ? `Aragora surfaced ${totalFindings} finding${totalFindings === 1 ? "" : "s"}.`
        : "Aragora completed the review without surfaced findings.",
      verdictReasoning:
        receipt.verdict_reasoning ||
        receipt.summary ||
        "No detailed reasoning was returned in the receipt.",
      inputSummary: receipt.input_summary || truncateText(fallbackSummary, 280),
      findings,
      riskSummary: {
        critical: Number(riskSummary.critical || 0),
        high: Number(riskSummary.high || 0),
        medium: Number(riskSummary.medium || 0),
        low: Number(riskSummary.low || 0),
        total: totalFindings,
      },
      source,
      completedAt: receipt.timestamp || new Date().toISOString(),
    };
  }

  async function pollForReceipt(settings, gauntletId, source, fallbackSummary) {
    for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt += 1) {
      const statusPayload = await aragoraRequest(settings, `/api/v1/gauntlet/${gauntletId}`, {
        method: "GET",
      });
      const status = String(statusPayload.status || "").toLowerCase();

      if (status === "completed") {
        const receipt = await aragoraRequest(
          settings,
          `/api/gauntlet/${gauntletId}/receipt?signed=false`,
          { method: "GET" }
        );
        const normalized = normalizeReceipt(receipt, source, fallbackSummary);
        await updateLatestReview(normalized);
        setBadge(normalized.riskSummary.total ? "RISK" : "PASS", normalized.riskSummary.total ? "#b45309" : "#166534");
        return normalized;
      }

      if (status === "failed" || status === "error") {
        const errorMessage = extractErrorMessage(
          statusPayload.result || statusPayload,
          "Aragora did not complete the review."
        );
        const failedReview = await updateLatestReview({
          status: "error",
          gauntletId,
          source,
          message: errorMessage,
          inputSummary: truncateText(fallbackSummary, 280),
        });
        setBadge("ERR", "#b91c1c");
        return failedReview;
      }

      await updateLatestReview({
        status: status || "pending",
        gauntletId,
        source,
        inputSummary: truncateText(fallbackSummary, 280),
        message:
          status === "running"
            ? "Aragora is adversarially testing the selected text."
            : "Review queued. Waiting for results.",
      });
      setBadge("RUN", "#0f766e");
      await delay(POLL_INTERVAL_MS);
    }

    const timedOutReview = await updateLatestReview({
      status: "error",
      gauntletId,
      source,
      message: "Timed out while waiting for Aragora review results.",
      inputSummary: truncateText(fallbackSummary, 280),
    });
    setBadge("ERR", "#b91c1c");
    return timedOutReview;
  }

  async function runReview(selectionText, source) {
    const settings = await getSettings();
    const trimmedSelection = (selectionText || "").trim();

    if (!trimmedSelection) {
      await updateLatestReview({
        status: "error",
        source,
        message: "Select text before running an Aragora review.",
      });
      setBadge("ERR", "#b91c1c");
      return null;
    }

    if (!settings.apiToken) {
      await updateLatestReview({
        status: "error",
        source,
        message: "Add a bearer token in the popup before running Aragora reviews.",
        inputSummary: truncateText(trimmedSelection, 280),
      });
      setBadge("ERR", "#b91c1c");
      return null;
    }

    await updateLatestReview({
      status: "starting",
      source,
      findings: [],
      verdict: null,
      gauntletId: null,
      receiptId: null,
      inputSummary: truncateText(trimmedSelection, 280),
      message: "Submitting the selected text to Aragora gauntlet.",
    });
    setBadge("RUN", "#0f766e");

    try {
      const startResponse = await aragoraRequest(settings, "/api/gauntlet/run", {
        method: "POST",
        body: JSON.stringify({
          input_content: trimmedSelection,
          input_type: "text",
          persona: settings.persona,
          profile: settings.profile,
        }),
      });

      const gauntletId = startResponse.gauntlet_id;
      await updateLatestReview({
        status: startResponse.status || "pending",
        gauntletId,
        source,
        message: "Review queued. Waiting for adversarial findings.",
      });

      return await pollForReceipt(settings, gauntletId, source, trimmedSelection);
    } catch (error) {
      const failedReview = await updateLatestReview({
        status: "error",
        source,
        inputSummary: truncateText(trimmedSelection, 280),
        message: error instanceof Error ? error.message : "Aragora request failed.",
      });
      setBadge("ERR", "#b91c1c");
      return failedReview;
    }
  }

  async function ensureContextMenu() {
    return new Promise((resolve) => {
      chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create(
          {
            id: MENU_ID,
            title: "Send Selection to Aragora Review",
            contexts: ["selection"],
          },
          () => {
            resolve();
          }
        );
      });
    });
  }

  function attachBackgroundHandlers() {
    chrome.runtime.onInstalled.addListener(() => {
      void ensureContextMenu();
      void clearLatestReview();
      setBadge("", "#0f766e");
    });

    chrome.runtime.onStartup.addListener(() => {
      void ensureContextMenu();
    });

    chrome.contextMenus.onClicked.addListener((info, tab) => {
      if (info.menuItemId !== MENU_ID) {
        return;
      }

      const source = {
        pageTitle: tab && tab.title ? tab.title : "",
        pageUrl: info.pageUrl || (tab && tab.url ? tab.url : ""),
      };

      void runReview(info.selectionText || "", source);
    });

    void ensureContextMenu();
  }

  function getSeverityClass(severity) {
    const normalized = String(severity || "low").toLowerCase();
    if (normalized === "critical" || normalized === "high") {
      return normalized;
    }
    if (normalized === "medium") {
      return "medium";
    }
    return "low";
  }

  function formatDate(value) {
    if (!value) {
      return "Waiting for a review run.";
    }

    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return "Waiting for a review run.";
    }

    return `Updated ${parsed.toLocaleString()}`;
  }

  function bindPopup() {
    const settingsForm = document.getElementById("settings-form");
    const apiUrlInput = document.getElementById("api-url");
    const apiTokenInput = document.getElementById("api-token");
    const personaSelect = document.getElementById("persona");
    const profileSelect = document.getElementById("profile");
    const settingsStatus = document.getElementById("settings-status");
    const clearResultsButton = document.getElementById("clear-results");
    const statusPill = document.getElementById("status-pill");
    const resultUpdatedAt = document.getElementById("result-updated-at");
    const verdictValue = document.getElementById("verdict-value");
    const findingsCount = document.getElementById("findings-count");
    const riskBreakdown = document.getElementById("risk-breakdown");
    const resultMessage = document.getElementById("result-message");
    const gauntletId = document.getElementById("gauntlet-id");
    const sourceUrl = document.getElementById("source-url");
    const verdictReasoning = document.getElementById("verdict-reasoning");
    const findingsCaption = document.getElementById("findings-caption");
    const findingsList = document.getElementById("findings-list");

    function renderSettings(settings) {
      apiUrlInput.value = settings.apiUrl || DEFAULT_SETTINGS.apiUrl;
      apiTokenInput.value = settings.apiToken || "";
      personaSelect.value = settings.persona || DEFAULT_SETTINGS.persona;
      profileSelect.value = settings.profile || DEFAULT_SETTINGS.profile;
    }

    function renderFindings(review) {
      findingsList.textContent = "";
      if (!Array.isArray(review.findings) || review.findings.length === 0) {
        const emptyState = document.createElement("li");
        emptyState.className = "finding-item";
        emptyState.textContent = "No detailed findings are stored for the latest result.";
        findingsList.appendChild(emptyState);
        findingsCaption.textContent = "No findings yet";
        return;
      }

      findingsCaption.textContent = `${review.findings.length} finding${review.findings.length === 1 ? "" : "s"} shown`;
      review.findings.forEach((finding) => {
        const item = document.createElement("li");
        item.className = "finding-item";

        const topLine = document.createElement("div");
        topLine.className = "finding-topline";

        const title = document.createElement("p");
        title.className = "finding-title";
        title.textContent = finding.title;

        const severity = document.createElement("span");
        severity.className = `finding-severity ${getSeverityClass(finding.severity)}`;
        severity.textContent = finding.severity;

        topLine.appendChild(title);
        topLine.appendChild(severity);

        const description = document.createElement("p");
        description.className = "finding-description";
        description.textContent = finding.description;

        const category = document.createElement("p");
        category.className = "finding-category";
        category.textContent = `Category: ${finding.category}`;

        item.appendChild(topLine);
        item.appendChild(description);
        item.appendChild(category);
        findingsList.appendChild(item);
      });
    }

    function renderReview(review) {
      const reviewStatus = String(review.status || "idle").toLowerCase();
      const verdict = String(review.verdict || reviewStatus || "idle").toUpperCase();
      statusPill.textContent = verdict === "IDLE" ? "Idle" : verdict;
      statusPill.className = "status-pill";
      if (verdict === "PASS") {
        statusPill.classList.add("is-pass");
      } else if (verdict === "CONDITIONAL" || reviewStatus === "running" || reviewStatus === "pending") {
        statusPill.classList.add("is-conditional");
      } else if (verdict === "FAIL" || reviewStatus === "error") {
        statusPill.classList.add("is-fail");
      }

      resultUpdatedAt.textContent = formatDate(review.updatedAt || review.completedAt);
      verdictValue.textContent = verdict === "IDLE" ? "Not run" : verdict;
      findingsCount.textContent = String((review.riskSummary && review.riskSummary.total) || 0);
      riskBreakdown.textContent = `${(review.riskSummary && review.riskSummary.critical) || 0} critical / ${(review.riskSummary && review.riskSummary.high) || 0} high`;
      resultMessage.textContent = review.message || DEFAULT_REVIEW.message;
      gauntletId.textContent = review.gauntletId || "Not started";
      sourceUrl.textContent = getSourceLabel(review.source);
      verdictReasoning.textContent =
        review.verdictReasoning ||
        "Results from the most recent gauntlet receipt will appear here.";
      renderFindings(review);
    }

    async function refreshPopup() {
      const [settings, review] = await Promise.all([getSettings(), getLatestReview()]);
      renderSettings(settings);
      renderReview(review);

      if (
        review.gauntletId &&
        ["starting", "pending", "running"].includes(String(review.status || "").toLowerCase())
      ) {
        settingsStatus.textContent = "Refreshing pending review from Aragora.";
        try {
          await pollForReceipt(
            settings,
            review.gauntletId,
            review.source || null,
            review.inputSummary || ""
          );
          const latest = await getLatestReview();
          renderReview(latest);
          settingsStatus.textContent = "Latest review refreshed.";
        } catch (error) {
          settingsStatus.textContent = error instanceof Error ? error.message : "Refresh failed.";
        }
      }
    }

    settingsForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      settingsStatus.textContent = "Saving extension settings.";
      try {
        const settings = await saveSettings({
          apiUrl: apiUrlInput.value,
          apiToken: apiTokenInput.value,
          persona: personaSelect.value,
          profile: profileSelect.value,
        });
        renderSettings(settings);
        settingsStatus.textContent = "Saved. Future context-menu reviews will use these settings.";
      } catch (error) {
        settingsStatus.textContent = error instanceof Error ? error.message : "Save failed.";
      }
    });

    clearResultsButton.addEventListener("click", async () => {
      await clearLatestReview();
      const review = await getLatestReview();
      renderReview(review);
      settingsStatus.textContent = "Stored review state cleared.";
      setBadge("", "#0f766e");
    });

    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName === "local" && changes[REVIEW_KEY]) {
        renderReview({ ...DEFAULT_REVIEW, ...(changes[REVIEW_KEY].newValue || {}) });
      }
      if (areaName === "sync" && changes[SETTINGS_KEY]) {
        renderSettings({ ...DEFAULT_SETTINGS, ...(changes[SETTINGS_KEY].newValue || {}) });
      }
    });

    void refreshPopup();
  }

  if (!EXTENSION_CONTEXT) {
    return;
  }

  if (POPUP_CONTEXT) {
    document.addEventListener("DOMContentLoaded", bindPopup);
  } else {
    attachBackgroundHandlers();
  }
})();
