"""Memory viewer mixin (MemoryViewerMixin).

Extracted from memory.py to reduce file size.
Contains HTML viewer rendering for memory browser UI.
"""

from __future__ import annotations

from ..base import HandlerResult


class MemoryViewerMixin:
    """Mixin providing memory viewer HTML rendering."""

    def _html_response(self, html: str, status: int = 200) -> HandlerResult:
        """Create an HTML response."""
        return HandlerResult(
            status_code=status,
            content_type="text/html",
            body=html.encode("utf-8"),
        )

    def _render_viewer(self) -> HandlerResult:
        """Render the memory viewer HTML UI."""
        html = _VIEWER_HTML
        return self._html_response(html)


# HTML template for the memory viewer (extracted for readability)
_VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Aragora Memory Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-1: #f6f1e9;
      --bg-2: #e7efe9;
      --ink: #111214;
      --muted: #5c6166;
      --accent: #0f766e;
      --accent-2: #c46f2b;
      --panel: #ffffff;
      --line: #e3e0da;
      --shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fef8ee 0%, var(--bg-1) 45%, var(--bg-2) 100%);
      min-height: 100vh;
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }
    header {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 24px;
    }
    header h1 {
      font-size: 32px;
      margin: 0;
      letter-spacing: -0.02em;
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(280px, 1.4fr) minmax(260px, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      animation: rise 0.5s ease both;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
    }
    input[type="text"],
    input[type="number"] {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      font-family: "Space Grotesk", sans-serif;
      width: 100%;
    }
    .mono {
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    .tier-row, .toggle-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .tier-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      background: #faf8f3;
    }
    .primary {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .primary:hover {
      background: #0b5d56;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 520px;
      overflow: auto;
      padding-right: 6px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      text-align: left;
      background: #fff;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .item:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .item .meta {
      display: flex;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .item .preview {
      font-size: 13px;
      margin-top: 6px;
    }
    .detail {
      border-top: 1px dashed var(--line);
      padding-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .detail h3 {
      margin: 0;
      font-size: 14px;
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .detail pre {
      white-space: pre-wrap;
      background: #f9f7f2;
      border-radius: 10px;
      padding: 12px;
      margin: 0;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>Memory Viewer</h1>
      <p>Progressive disclosure search across continuum memory, with optional external sources.</p>
    </header>
    <div class="grid">
      <section class="panel">
        <h2>Search</h2>
        <label for="query">Query</label>
        <input id="query" type="text" placeholder="Search memory content" />
        <label>Tiers</label>
        <div class="tier-row">
          <label class="tier-pill"><input type="checkbox" class="tier" value="fast" checked /> fast</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="medium" checked /> medium</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="slow" checked /> slow</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="glacial" checked /> glacial</label>
        </div>
        <label for="limit">Limit</label>
        <input id="limit" type="number" value="20" min="1" max="100" />
        <label for="minImportance">Min Importance</label>
        <input id="minImportance" type="number" value="0" step="0.05" min="0" max="1" />
        <div class="toggle-row">
          <label class="tier-pill"><input id="useHybrid" type="checkbox" /> hybrid search</label>
          <label class="tier-pill"><input id="includeExternal" type="checkbox" /> external</label>
        </div>
        <div class="toggle-row">
          <label class="tier-pill"><input id="extSupermemory" type="checkbox" /> supermemory</label>
          <label class="tier-pill"><input id="extClaudeMem" type="checkbox" /> claude-mem</label>
        </div>
        <button class="primary" id="searchBtn">Search</button>
        <div id="status" class="mono"></div>
      </section>

      <section class="panel">
        <h2>Index</h2>
        <div id="results" class="list"></div>
        <div class="detail">
          <h3>External</h3>
          <div id="external" class="list"></div>
        </div>
      </section>

      <section class="panel">
        <h2>Timeline</h2>
        <div id="timeline" class="list"></div>
        <div class="detail">
          <h3>Entry</h3>
          <pre id="entry">Select a memory to view full content.</pre>
        </div>
      </section>
    </div>
  </div>

  <script>
    const apiBase = "/api/v1/memory";
    const resultsEl = document.getElementById("results");
    const externalEl = document.getElementById("external");
    const timelineEl = document.getElementById("timeline");
    const entryEl = document.getElementById("entry");
    const statusEl = document.getElementById("status");

    const qs = (id) => document.getElementById(id);
    const tiers = () => Array.from(document.querySelectorAll(".tier"))
      .filter((el) => el.checked)
      .map((el) => el.value);

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.style.color = isError ? "#b91c1c" : "#0f766e";
    }

    function clear(el) {
      while (el.firstChild) el.removeChild(el.firstChild);
    }

    function makeItem(item, onClick) {
      const button = document.createElement("button");
      button.className = "item";
      button.type = "button";
      button.addEventListener("click", onClick);

      const meta = document.createElement("div");
      meta.className = "meta";
      const tier = item.tier ? item.tier.toUpperCase() : (item.source || "external");
      meta.textContent = `${tier} | importance ${item.importance ?? "n/a"} | tokens ${item.token_estimate ?? "-"}`;

      const preview = document.createElement("div");
      preview.className = "preview";
      preview.textContent = item.preview || "(no preview)";

      button.appendChild(meta);
      button.appendChild(preview);
      return button;
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      return res.json();
    }

    async function searchIndex() {
      const query = qs("query").value.trim();
      if (!query) {
        setStatus("Enter a query.", true);
        return;
      }
      setStatus("Searching...");

      const params = new URLSearchParams();
      params.set("q", query);
      params.set("limit", qs("limit").value);
      params.set("min_importance", qs("minImportance").value);
      const tierValues = tiers();
      if (tierValues.length) {
        params.set("tier", tierValues.join(","));
      }
      if (qs("useHybrid").checked) {
        params.set("use_hybrid", "true");
      }
      if (qs("includeExternal").checked) {
        params.set("include_external", "true");
        const ext = [];
        if (qs("extSupermemory").checked) ext.push("supermemory");
        if (qs("extClaudeMem").checked) ext.push("claude-mem");
        if (ext.length) {
          params.set("external", ext.join(","));
        }
      }

      try {
        const data = await fetchJson(`${apiBase}/search-index?${params.toString()}`);
        renderResults(data.results || []);
        renderExternal(data.external_results || []);
        setStatus(`Found ${data.count} results${data.external_results?.length ? " + external" : ""}.`);
        if (data.results && data.results.length) {
          loadTimeline(data.results[0].id);
        }
      } catch (err) {
        setStatus(err.message || "Search failed.", true);
      }
    }

    function renderResults(items) {
      clear(resultsEl);
      if (!items.length) {
        resultsEl.textContent = "No results yet.";
        return;
      }
      items.forEach((item) => {
        const node = makeItem(item, () => loadTimeline(item.id));
        resultsEl.appendChild(node);
      });
    }

    function renderExternal(items) {
      clear(externalEl);
      if (!items.length) {
        externalEl.textContent = "No external results.";
        return;
      }
      items.forEach((item) => {
        const node = makeItem(item, () => showExternal(item));
        externalEl.appendChild(node);
      });
    }

    function showExternal(item) {
      const lines = [];
      lines.push(`source: ${item.source}`);
      if (item.metadata && Object.keys(item.metadata).length) {
        lines.push(`metadata: ${JSON.stringify(item.metadata, null, 2)}`);
      }
      lines.push("");
      lines.push(item.preview || "");
      entryEl.textContent = lines.join("\\n");
    }

    async function loadTimeline(anchorId) {
      if (!anchorId) return;
      const params = new URLSearchParams();
      params.set("anchor_id", anchorId);
      params.set("before", "3");
      params.set("after", "3");
      const tierValues = tiers();
      if (tierValues.length) {
        params.set("tier", tierValues.join(","));
      }
      try {
        const data = await fetchJson(`${apiBase}/search-timeline?${params.toString()}`);
        renderTimeline(data);
        if (data.anchor) {
          loadEntry(data.anchor.id);
        }
      } catch (err) {
        setStatus(err.message || "Timeline failed.", true);
      }
    }

    function renderTimeline(data) {
      clear(timelineEl);
      if (!data || !data.anchor) {
        timelineEl.textContent = "Timeline unavailable.";
        return;
      }
      const items = [...(data.before || []), data.anchor, ...(data.after || [])];
      items.forEach((item) => {
        const node = makeItem(item, () => loadEntry(item.id));
        timelineEl.appendChild(node);
      });
    }

    async function loadEntry(entryId) {
      if (!entryId) return;
      try {
        const data = await fetchJson(`${apiBase}/entries?ids=${encodeURIComponent(entryId)}`);
        const entry = data.entries && data.entries.length ? data.entries[0] : null;
        if (!entry) {
          entryEl.textContent = "Entry not found.";
          return;
        }
        const lines = [];
        lines.push(`id: ${entry.id}`);
        lines.push(`tier: ${entry.tier}`);
        lines.push(`importance: ${entry.importance}`);
        if (entry.red_line) {
          lines.push(`red_line: ${entry.red_line_reason || "true"}`);
        }
        lines.push("");
        lines.push(entry.content || "");
        entryEl.textContent = lines.join("\\n");
      } catch (err) {
        entryEl.textContent = err.message || "Failed to load entry.";
      }
    }

    document.getElementById("searchBtn").addEventListener("click", searchIndex);
  </script>
</body>
</html>"""
