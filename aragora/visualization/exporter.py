"""Export utilities for argument graphs."""

from pathlib import Path
from typing import Optional
import json

from aragora.visualization.mapper import ArgumentCartographer


def save_debate_visualization(
    cartographer: ArgumentCartographer,
    output_dir: Path,
    debate_id: str,
    formats: Optional[list] = None,
) -> dict:
    """
    Save debate visualization to multiple formats.
    
    Args:
        cartographer: The cartographer with the debate graph
        output_dir: Directory to save files
        debate_id: ID for naming files
        formats: List of formats to export ("mermaid", "json", "html")
    
    Returns:
        Dictionary mapping format to output file path
    """
    formats = formats or ["mermaid", "json"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if "mermaid" in formats:
        mermaid_path = output_dir / f"{debate_id}_graph.mermaid"
        mermaid_path.write_text(cartographer.export_mermaid())
        results["mermaid"] = str(mermaid_path)
    
    if "json" in formats:
        json_path = output_dir / f"{debate_id}_graph.json"
        json_path.write_text(cartographer.export_json(include_full_content=True))
        results["json"] = str(json_path)
    
    if "html" in formats:
        html_path = output_dir / f"{debate_id}_graph.html"
        html_content = generate_standalone_html(cartographer)
        html_path.write_text(html_content)
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
                <div class="stat-value">{stats['total_nodes']}</div>
                <div class="stat-label">Arguments</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats['total_edges']}</div>
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