"""Export utilities for argument graphs."""

from pathlib import Path
from typing import Optional
import hashlib
import json
import threading
import time

from aragora.visualization.mapper import ArgumentCartographer


# In-memory export cache with TTL
# Key: (debate_id, format, graph_hash) -> (content, timestamp)
_export_cache: dict[tuple[str, str, str], tuple[str, float]] = {}
_export_cache_lock = threading.Lock()
_EXPORT_CACHE_TTL = 300.0  # 5 minutes
_MAX_CACHE_ENTRIES = 100


def _get_graph_hash(cartographer: ArgumentCartographer) -> str:
    """Get a hash of the current graph state for caching."""
    stats = cartographer.get_statistics()
    # Include key metrics that affect output
    hash_input = f"{stats['node_count']}:{stats['edge_count']}:{stats['rounds']}:{','.join(sorted(stats['agents']))}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _get_cached_export(debate_id: str, format_name: str, graph_hash: str) -> Optional[str]:
    """Get cached export if valid. Thread-safe."""
    key = (debate_id, format_name, graph_hash)
    with _export_cache_lock:
        if key in _export_cache:
            content, timestamp = _export_cache[key]
            if time.time() - timestamp < _EXPORT_CACHE_TTL:
                return content
            else:
                del _export_cache[key]
    return None


def _cache_export(debate_id: str, format_name: str, graph_hash: str, content: str) -> None:
    """Cache an export. Thread-safe."""
    global _export_cache

    with _export_cache_lock:
        # Evict old entries if at limit
        if len(_export_cache) >= _MAX_CACHE_ENTRIES:
            now = time.time()
            # Remove expired entries first
            expired = [k for k, (_, ts) in _export_cache.items() if now - ts > _EXPORT_CACHE_TTL]
            for k in expired:
                del _export_cache[k]

            # If still at limit, remove oldest
            if len(_export_cache) >= _MAX_CACHE_ENTRIES:
                oldest_key = min(_export_cache.keys(), key=lambda k: _export_cache[k][1])
                del _export_cache[oldest_key]

        _export_cache[(debate_id, format_name, graph_hash)] = (content, time.time())


def clear_export_cache() -> int:
    """Clear the export cache. Returns number of entries cleared. Thread-safe."""
    global _export_cache
    with _export_cache_lock:
        count = len(_export_cache)
        _export_cache = {}
        return count


def save_debate_visualization(
    cartographer: ArgumentCartographer,
    output_dir: Path,
    debate_id: str,
    formats: Optional[list] = None,
    use_cache: bool = True,
) -> dict:
    """
    Save debate visualization to multiple formats.

    Args:
        cartographer: The cartographer with the debate graph
        output_dir: Directory to save files
        debate_id: ID for naming files
        formats: List of formats to export ("mermaid", "json", "html")
        use_cache: Whether to use caching for export content

    Returns:
        Dictionary mapping format to output file path
    """
    formats = formats or ["mermaid", "json"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get graph hash for caching
    graph_hash = _get_graph_hash(cartographer) if use_cache else ""

    results = {}

    if "mermaid" in formats:
        mermaid_path = output_dir / f"{debate_id}_graph.mermaid"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "mermaid", graph_hash)
        if content is None:
            content = cartographer.export_mermaid()
            if use_cache:
                _cache_export(debate_id, "mermaid", graph_hash, content)
        mermaid_path.write_text(content)
        results["mermaid"] = str(mermaid_path)

    if "json" in formats:
        json_path = output_dir / f"{debate_id}_graph.json"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "json", graph_hash)
        if content is None:
            content = cartographer.export_json(include_full_content=True)
            if use_cache:
                _cache_export(debate_id, "json", graph_hash, content)
        json_path.write_text(content)
        results["json"] = str(json_path)

    if "html" in formats:
        html_path = output_dir / f"{debate_id}_graph.html"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "html", graph_hash)
        if content is None:
            content = generate_standalone_html(cartographer)
            if use_cache:
                _cache_export(debate_id, "html", graph_hash, content)
        html_path.write_text(content)
        results["html"] = str(html_path)

    return results


def generate_standalone_html(cartographer: ArgumentCartographer) -> str:
    """Generate a standalone HTML file with embedded Mermaid diagram."""
    mermaid_code = cartographer.export_mermaid()
    stats = cartographer.get_statistics()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora Debate: {cartographer.topic or 'Untitled'}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #16213e;
            padding: 15px 25px;
            border-radius: 8px;
            border: 1px solid #0f3460;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .mermaid {{
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ {cartographer.topic or 'Aragora Debate'}</h1>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{stats['node_count']}</div>
                <div class="stat-label">Arguments</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats['edge_count']}</div>
                <div class="stat-label">Connections</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats['rounds']}</div>
                <div class="stat-label">Rounds</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(stats['agents'])}</div>
                <div class="stat-label">Agents</div>
            </div>
        </div>
        
        <div class="mermaid">
{mermaid_code}
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4CAF50"></div>
                <span>Proposal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF5722"></div>
                <span>Critique</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9C27B0"></div>
                <span>Evidence</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF9800"></div>
                <span>Concession</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2196F3"></div>
                <span>Consensus</span>
            </div>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
