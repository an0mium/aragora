(function attachAragoraExtensionApi(globalScope) {
  const DEFAULT_SETTINGS = {
    apiUrl: "https://api.aragora.ai",
    apiKey: "",
    agents: "",
    rounds: 3,
    consensus: "majority",
  };
  const ADVERSARIAL_REVIEW_PROMPT =
    "Perform an adversarial review of the highlighted text. Identify factual gaps, unsafe assumptions, contradictions, manipulation risks, and the strongest counterarguments. End with a concise verdict.";

  function normalizeApiUrl(apiUrl) {
    return String(apiUrl || DEFAULT_SETTINGS.apiUrl).trim().replace(/\/+$/, "");
  }

  async function readErrorMessage(response) {
    const fallback = `Request failed with status ${response.status}`;

    try {
      const body = await response.json();
      return body.detail || body.error || body.message || fallback;
    } catch (_error) {
      try {
        const text = await response.text();
        return text || fallback;
      } catch (_textError) {
        return fallback;
      }
    }
  }

  function buildAdversarialReviewPayload(selectionText, source, settings) {
    const sourceTitle = String(source?.pageTitle || "").trim();
    const sourceUrl = String(source?.pageUrl || "").trim();
    const normalizedSelection = String(selectionText || "").replace(/\u0000/g, "").trim();
    const sourceContext = [
      sourceTitle ? `Source title: ${sourceTitle}` : "",
      sourceUrl ? `Source URL: ${sourceUrl}` : "",
      "Highlighted text:",
      normalizedSelection,
    ]
      .filter(Boolean)
      .join("\n\n");

    const payload = {
      question: sourceTitle
        ? `${ADVERSARIAL_REVIEW_PROMPT} Focus on the excerpt from "${sourceTitle}".`
        : ADVERSARIAL_REVIEW_PROMPT,
      rounds: Number(settings.rounds) || DEFAULT_SETTINGS.rounds,
      consensus: settings.consensus || DEFAULT_SETTINGS.consensus,
      auto_select: !String(settings.agents || "").trim(),
      context: sourceContext.slice(0, 10000),
      metadata: {
        source: "browser_extension_context_menu",
        workflow: "adversarial_review",
        source_title: sourceTitle,
        source_url: sourceUrl,
        selection_length: normalizedSelection.length,
      },
    };

    const agents = String(settings.agents || "").trim();
    if (agents) {
      payload.agents = agents;
    }

    return payload;
  }

  function getFinalAnswer(debate) {
    return (
      debate?.final_answer ||
      debate?.finalAnswer ||
      debate?.consensus?.final_answer ||
      debate?.consensus?.finalAnswer ||
      debate?.consensus?.summary ||
      debate?.answer ||
      ""
    );
  }

  function buildDebateSnapshot(debate, fallbackStatus) {
    const confidence = Number(
      debate?.consensus?.confidence ?? debate?.confidence ?? debate?.result?.confidence
    );

    return {
      debateId: debate?.debate_id || debate?.id || debate?.result?.debateId || null,
      status: String(debate?.status || fallbackStatus || "running").trim().toLowerCase(),
      message: debate?.message || debate?.detail || null,
      finalAnswer: getFinalAnswer(debate),
      confidence: Number.isNaN(confidence) ? null : confidence,
      task: debate?.task || debate?.question || "",
    };
  }

  globalScope.AragoraExtensionApi = {
    ADVERSARIAL_REVIEW_PROMPT,
    DEFAULT_SETTINGS,
    buildAdversarialReviewPayload,
    buildDebateSnapshot,
    normalizeApiUrl,
    readErrorMessage,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
