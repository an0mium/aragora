"""
Static HTML Exporter - Self-contained interactive debate viewer.

Generates a single HTML file with embedded CSS and JavaScript that:
- Visualizes the debate graph structure
- Provides step-through timeline replay
- Shows provenance for each claim
- Works completely offline

No external dependencies - all assets are embedded.
"""

import json
from pathlib import Path

from aragora.export.artifact import DebateArtifact

# Embedded CSS styles for the HTML viewer
# Extracted from _generate_styles() for better organization and maintainability
_STYLES_CSS = """:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #8b5cf6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg: #0f172a;
    --bg-light: #1e293b;
    --card: #1e293b;
    --card-hover: #334155;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #334155;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}

#app {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 1rem;
}

header h1 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.meta {
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
}

.task-section {
    background: var(--card);
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 2rem;
    border-left: 4px solid var(--primary);
}

.task-section h2 {
    font-size: 0.9rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.tab {
    padding: 0.75rem 1.5rem;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 0.5rem 0.5rem 0 0;
    font-size: 0.95rem;
    transition: all 0.2s;
}

.tab:hover {
    color: var(--text);
    background: var(--card);
}

.tab.active {
    color: var(--primary);
    background: var(--card);
    font-weight: 600;
}

.tab-panel {
    display: none;
}

.tab-panel.active {
    display: block;
}

/* Graph View */
.graph-container {
    background: var(--card);
    border-radius: 0.5rem;
    padding: 1.5rem;
    min-height: 400px;
    position: relative;
}

.graph-canvas {
    width: 100%;
    height: 400px;
    position: relative;
}

.graph-node {
    position: absolute;
    padding: 0.75rem 1rem;
    background: var(--bg);
    border: 2px solid var(--primary);
    border-radius: 0.5rem;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    max-width: 200px;
}

.graph-node:hover {
    background: var(--card-hover);
    transform: scale(1.05);
}

.graph-node.proposal { border-color: var(--primary); }
.graph-node.critique { border-color: var(--warning); }
.graph-node.synthesis { border-color: var(--success); }
.graph-node.root { border-color: var(--secondary); }

.graph-node .agent {
    font-weight: 600;
    color: var(--primary);
}

.graph-node .type {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Timeline View */
.timeline {
    position: relative;
    padding-left: 2rem;
}

.timeline::before {
    content: '';
    position: absolute;
    left: 0.5rem;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
}

.timeline-item {
    position: relative;
    padding: 1rem;
    margin-bottom: 1rem;
    background: var(--card);
    border-radius: 0.5rem;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -1.75rem;
    top: 1.25rem;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--primary);
}

.timeline-item.critique::before { background: var(--warning); }
.timeline-item.synthesis::before { background: var(--success); }

.timeline-item .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.timeline-item .agent {
    font-weight: 600;
    color: var(--primary);
}

.timeline-item .round {
    font-size: 0.85rem;
    color: var(--text-muted);
}

.timeline-item .content {
    font-size: 0.95rem;
    white-space: pre-wrap;
    max-height: 200px;
    overflow-y: auto;
}

.timeline-controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    align-items: center;
}

.timeline-controls button {
    padding: 0.5rem 1rem;
    background: var(--primary);
    border: none;
    color: white;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: background 0.2s;
}

.timeline-controls button:hover {
    background: var(--primary-dark);
}

.timeline-controls button:disabled {
    background: var(--border);
    cursor: not-allowed;
}

.timeline-slider {
    flex: 1;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--border);
    border-radius: 2px;
    outline: none;
}

.timeline-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--primary);
    border-radius: 50%;
    cursor: pointer;
}

/* Provenance View */
.provenance-list {
    display: grid;
    gap: 1rem;
}

.provenance-item {
    background: var(--card);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 3px solid var(--secondary);
}

.provenance-item .evidence-id {
    font-family: monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.provenance-item .source {
    display: flex;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.provenance-item .source-type {
    background: var(--bg);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
}

.provenance-item .content {
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

.chain-visualization {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.chain-link {
    padding: 0.25rem 0.5rem;
    background: var(--bg);
    border-radius: 0.25rem;
    font-family: monospace;
}

/* Verification View */
.verification-list {
    display: grid;
    gap: 1rem;
}

.verification-item {
    background: var(--card);
    padding: 1rem;
    border-radius: 0.5rem;
}

.verification-item.verified { border-left: 3px solid var(--success); }
.verification-item.refuted { border-left: 3px solid var(--danger); }
.verification-item.timeout { border-left: 3px solid var(--warning); }

.verification-item .status {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    font-weight: 600;
}

.verification-item .status.verified { background: rgba(34, 197, 94, 0.2); color: var(--success); }
.verification-item .status.refuted { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
.verification-item .status.timeout { background: rgba(245, 158, 11, 0.2); color: var(--warning); }

/* Stats */
.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 2rem;
}

.stat {
    background: var(--card);
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary);
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Consensus Badge */
.consensus-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: 600;
    margin-top: 1rem;
}

.consensus-badge.reached {
    background: rgba(34, 197, 94, 0.2);
    color: var(--success);
}

.consensus-badge.not-reached {
    background: rgba(239, 68, 68, 0.2);
    color: var(--danger);
}

/* Footer */
footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.85rem;
}

footer a {
    color: var(--primary);
    text-decoration: none;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--text-muted);
}

/* Modal */
.modal-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    justify-content: center;
    align-items: center;
}

.modal-overlay.active {
    display: flex;
}

.modal {
    background: var(--card);
    padding: 2rem;
    border-radius: 1rem;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    position: relative;
}

.modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.5rem;
    cursor: pointer;
}

.modal h3 {
    margin-bottom: 1rem;
}

.modal pre {
    background: var(--bg);
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    font-size: 0.85rem;
}

@media (max-width: 768px) {
    #app {
        padding: 1rem;
    }

    header h1 {
        font-size: 1.5rem;
    }

    .tabs {
        flex-wrap: wrap;
    }

    .tab {
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
    }

    .stats {
        grid-template-columns: repeat(2, 1fr);
    }
}"""


class StaticHTMLExporter:
    """Generates self-contained HTML debate viewers."""

    def __init__(self, artifact: DebateArtifact):
        self.artifact = artifact

    def generate(self) -> str:
        """Generate the complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>aragora Debate: {self._escape(self.artifact.task[:50])}...</title>
    {self._generate_styles()}
</head>
<body>
    <div id="app">
        {self._generate_header()}
        {self._generate_task_section()}
        {self._generate_tabs()}
        <div id="tab-content">
            {self._generate_graph_view()}
            {self._generate_timeline_view()}
            {self._generate_provenance_view()}
            {self._generate_verification_view()}
        </div>
        {self._generate_stats()}
        {self._generate_footer()}
    </div>
    {self._generate_scripts()}
</body>
</html>"""

    def _escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _generate_styles(self) -> str:
        """Generate embedded CSS from module-level constant."""
        return f"<style>{_STYLES_CSS}</style>"

    def _generate_header(self) -> str:
        """Generate header section."""
        consensus = self.artifact.consensus_proof
        consensus_text = "Consensus Reached" if consensus and consensus.reached else "No Consensus"
        confidence = f"{consensus.confidence:.0%}" if consensus else "N/A"

        return f"""
<header>
    <h1>aragora Debate</h1>
    <div class="meta">
        Multi-Agent Deliberation | {self.artifact.created_at[:10]} |
        ID: {self.artifact.artifact_id}
    </div>
    <div class="consensus-badge {"reached" if consensus and consensus.reached else "not-reached"}">
        {"&#10003;" if consensus and consensus.reached else "&#9888;"} {consensus_text} ({confidence})
    </div>
</header>"""

    def _generate_task_section(self) -> str:
        """Generate task description section."""
        return f"""
<div class="task-section">
    <h2>Task</h2>
    <p>{self._escape(self.artifact.task)}</p>
</div>"""

    def _generate_tabs(self) -> str:
        """Generate tab navigation."""
        return """
<div class="tabs">
    <button class="tab active" data-tab="graph">Graph</button>
    <button class="tab" data-tab="timeline">Timeline</button>
    <button class="tab" data-tab="provenance">Provenance</button>
    <button class="tab" data-tab="verification">Verification</button>
</div>"""

    def _generate_graph_view(self) -> str:
        """Generate graph visualization panel."""
        if not self.artifact.graph_data:
            return """
<div class="tab-panel active" id="panel-graph">
    <div class="graph-container">
        <div class="empty-state">
            <p>No graph data available for this debate.</p>
        </div>
    </div>
</div>"""

        # Generate nodes from graph data
        nodes_html = self._render_graph_nodes()

        return f"""
<div class="tab-panel active" id="panel-graph">
    <div class="graph-container">
        <div class="graph-canvas" id="graph-canvas">
            {nodes_html}
        </div>
    </div>
</div>"""

    def _render_graph_nodes(self) -> str:
        """Render graph nodes as HTML elements."""
        if not self.artifact.graph_data or "nodes" not in self.artifact.graph_data:
            return ""

        nodes = self.artifact.graph_data.get("nodes", {})
        html_parts = []

        # Simple layout algorithm - arrange by type and position
        x_offset = 50
        y_positions = {
            "root": 50,
            "proposal": 150,
            "critique": 250,
            "synthesis": 350,
            "conclusion": 450,
        }

        node_x: dict[str, tuple[int, int]] = {}
        type_counts: dict[str, int] = {}

        for node_id, node in nodes.items():
            node_type = node.get("node_type", "proposal")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

            x = x_offset + (type_counts[node_type] - 1) * 220
            y = y_positions.get(node_type, 200)

            node_x[node_id] = (x, y)

            agent = node.get("agent_id", "unknown")
            content = node.get("content", "")[:50]

            html_parts.append(f"""
<div class="graph-node {node_type}"
     style="left: {x}px; top: {y}px;"
     data-node-id="{node_id}"
     data-content="{self._escape(node.get("content", "")[:500])}"
     title="{self._escape(content)}...">
    <div class="agent">{self._escape(agent)}</div>
    <div class="type">{node_type}</div>
</div>""")

        return "".join(html_parts)

    def _generate_timeline_view(self) -> str:
        """Generate timeline replay panel."""
        if not self.artifact.trace_data:
            return """
<div class="tab-panel" id="panel-timeline">
    <div class="empty-state">
        <p>No trace data available for timeline replay.</p>
    </div>
</div>"""

        events = self.artifact.trace_data.get("events", [])
        timeline_items = []

        for event in events:
            event_type = event.get("event_type", "")
            if event_type not in ["agent_proposal", "agent_critique", "agent_synthesis"]:
                continue

            agent = event.get("agent", "unknown")
            content = event.get("content", {})
            round_num = event.get("round_num", 0)

            if event_type == "agent_proposal":
                text = content.get("content", "")[:300]
                item_class = ""
            elif event_type == "agent_critique":
                issues = content.get("issues", [])
                text = "Issues: " + ", ".join(issues[:3])
                item_class = "critique"
            else:
                text = content.get("content", "")[:300]
                item_class = "synthesis"

            timeline_items.append(f"""
<div class="timeline-item {item_class}">
    <div class="header">
        <span class="agent">{self._escape(agent)}</span>
        <span class="round">Round {round_num}</span>
    </div>
    <div class="content">{self._escape(text)}...</div>
</div>""")

        return f"""
<div class="tab-panel" id="panel-timeline">
    <div class="timeline-controls">
        <button id="btn-prev" disabled>&larr; Prev</button>
        <input type="range" class="timeline-slider" id="timeline-slider"
               min="0" max="{len(timeline_items) - 1}" value="0">
        <button id="btn-next">Next &rarr;</button>
        <button id="btn-play">&#9658; Play</button>
    </div>
    <div class="timeline" id="timeline-container">
        {"".join(timeline_items)}
    </div>
</div>"""

    def _generate_provenance_view(self) -> str:
        """Generate provenance chain panel."""
        if not self.artifact.provenance_data:
            return """
<div class="tab-panel" id="panel-provenance">
    <div class="empty-state">
        <p>No provenance data available.</p>
    </div>
</div>"""

        chain = self.artifact.provenance_data.get("chain", {})
        records = chain.get("records", [])

        provenance_items = []
        for record in records[-10:]:  # Show last 10 records
            provenance_items.append(f"""
<div class="provenance-item">
    <div class="evidence-id">{record.get("id", "unknown")}</div>
    <div class="source">
        <span class="source-type">{record.get("source_type", "unknown")}</span>
        <span>{self._escape(record.get("source_id", "")[:30])}</span>
    </div>
    <div class="content">{self._escape(record.get("content", "")[:200])}...</div>
    <div class="chain-visualization">
        <span>Hash:</span>
        <span class="chain-link">{record.get("content_hash", "")[:12]}...</span>
        {f'<span>&larr;</span><span class="chain-link">{record.get("previous_hash", "")[:12]}...</span>' if record.get("previous_hash") else ""}
    </div>
</div>""")

        return f"""
<div class="tab-panel" id="panel-provenance">
    <h3 style="margin-bottom: 1rem;">Evidence Chain ({len(records)} records)</h3>
    <div class="provenance-list">
        {"".join(provenance_items) if provenance_items else '<div class="empty-state">No provenance records.</div>'}
    </div>
</div>"""

    def _generate_verification_view(self) -> str:
        """Generate verification results panel."""
        verifications = self.artifact.verification_results

        if not verifications:
            return """
<div class="tab-panel" id="panel-verification">
    <div class="empty-state">
        <p>No formal verification results available.</p>
    </div>
</div>"""

        verification_items = []
        for v in verifications:
            status_class = (
                v.status.lower()
                if v.status.lower() in ["verified", "refuted", "timeout"]
                else "timeout"
            )
            verification_items.append(f"""
<div class="verification-item {status_class}">
    <div>
        <span class="status {status_class}">{v.status.upper()}</span>
        <span style="margin-left: 0.5rem; color: var(--text-muted);">via {v.method}</span>
    </div>
    <p style="margin-top: 0.5rem;">{self._escape(v.claim_text[:200])}</p>
    {f'<pre style="margin-top: 0.5rem;">{self._escape(v.proof_trace[:200] if v.proof_trace else "")}</pre>' if v.proof_trace else ""}
</div>""")

        return f"""
<div class="tab-panel" id="panel-verification">
    <h3 style="margin-bottom: 1rem;">Formal Verification Results</h3>
    <div class="verification-list">
        {"".join(verification_items)}
    </div>
</div>"""

    def _generate_stats(self) -> str:
        """Generate statistics section."""
        return f"""
<div class="stats">
    <div class="stat">
        <div class="stat-value">{self.artifact.rounds}</div>
        <div class="stat-label">Rounds</div>
    </div>
    <div class="stat">
        <div class="stat-value">{self.artifact.message_count}</div>
        <div class="stat-label">Messages</div>
    </div>
    <div class="stat">
        <div class="stat-value">{self.artifact.critique_count}</div>
        <div class="stat-label">Critiques</div>
    </div>
    <div class="stat">
        <div class="stat-value">{self.artifact.duration_seconds:.0f}s</div>
        <div class="stat-label">Duration</div>
    </div>
    <div class="stat">
        <div class="stat-value">{len(self.artifact.agents)}</div>
        <div class="stat-label">Agents</div>
    </div>
    <div class="stat">
        <div class="stat-value">{len(self.artifact.verification_results)}</div>
        <div class="stat-label">Verifications</div>
    </div>
</div>"""

    def _generate_footer(self) -> str:
        """Generate footer section."""
        return f"""
<footer>
    <p>
        Generated by <a href="https://github.com/an0mium/aragora">aragora</a> {self.artifact.generator} |
        Artifact: {self.artifact.artifact_id} |
        Hash: {self.artifact.content_hash}
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem;">
        This is a self-contained document. All data and interactions work offline.
    </p>
</footer>"""

    def _generate_scripts(self) -> str:
        """Generate embedded JavaScript."""
        artifact_json = json.dumps(self.artifact.to_dict())

        return f"""
<script>
// Embedded artifact data
const artifactData = {artifact_json};

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        // Update active tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Show corresponding panel
        const tabName = tab.dataset.tab;
        document.querySelectorAll('.tab-panel').forEach(panel => {{
            panel.classList.remove('active');
        }});
        document.getElementById('panel-' + tabName).classList.add('active');
    }});
}});

// Timeline controls
const timelineItems = document.querySelectorAll('.timeline-item');
const slider = document.getElementById('timeline-slider');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const btnPlay = document.getElementById('btn-play');
let currentIndex = 0;
let isPlaying = false;
let playInterval;

function updateTimeline() {{
    timelineItems.forEach((item, i) => {{
        item.style.opacity = i <= currentIndex ? '1' : '0.3';
        item.style.transform = i === currentIndex ? 'scale(1.02)' : 'scale(1)';
    }});
    slider.value = currentIndex;
    btnPrev.disabled = currentIndex === 0;
    btnNext.disabled = currentIndex >= timelineItems.length - 1;
}}

if (btnPrev) {{
    btnPrev.addEventListener('click', () => {{
        if (currentIndex > 0) {{
            currentIndex--;
            updateTimeline();
        }}
    }});
}}

if (btnNext) {{
    btnNext.addEventListener('click', () => {{
        if (currentIndex < timelineItems.length - 1) {{
            currentIndex++;
            updateTimeline();
        }}
    }});
}}

if (slider) {{
    slider.addEventListener('input', (e) => {{
        currentIndex = parseInt(e.target.value);
        updateTimeline();
    }});
}}

if (btnPlay) {{
    btnPlay.addEventListener('click', () => {{
        isPlaying = !isPlaying;
        btnPlay.innerHTML = isPlaying ? '&#9724; Pause' : '&#9658; Play';

        if (isPlaying) {{
            playInterval = setInterval(() => {{
                if (currentIndex < timelineItems.length - 1) {{
                    currentIndex++;
                    updateTimeline();
                }} else {{
                    isPlaying = false;
                    btnPlay.innerHTML = '&#9658; Play';
                    clearInterval(playInterval);
                }}
            }}, 1500);
        }} else {{
            clearInterval(playInterval);
        }}
    }});
}}

// Graph node interaction
document.querySelectorAll('.graph-node').forEach(node => {{
    node.addEventListener('click', () => {{
        const content = node.dataset.content;
        const nodeId = node.dataset.nodeId;

        // Create and show modal (using safe DOM methods to prevent XSS)
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay active';

        const modal = document.createElement('div');
        modal.className = 'modal';

        const closeBtn = document.createElement('button');
        closeBtn.className = 'modal-close';
        closeBtn.innerHTML = '&times;';

        const heading = document.createElement('h3');
        heading.textContent = 'Node ' + nodeId;

        const pre = document.createElement('pre');
        pre.textContent = content;

        modal.appendChild(closeBtn);
        modal.appendChild(heading);
        modal.appendChild(pre);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        closeBtn.addEventListener('click', () => {{
            overlay.remove();
        }});

        overlay.addEventListener('click', (e) => {{
            if (e.target === overlay) overlay.remove();
        }});
    }});
}});

// Initialize timeline
updateTimeline();

console.log('aragora debate viewer loaded. Artifact:', artifactData.artifact_id);
</script>"""

    def save(self, path: Path) -> Path:
        """Save the HTML to a file."""
        html = self.generate()
        path.write_text(html)
        return path


def export_to_html(artifact: DebateArtifact, output_path: Path) -> Path:
    """Convenience function to export artifact to HTML."""
    exporter = StaticHTMLExporter(artifact)
    return exporter.save(output_path)
