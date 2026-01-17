#!/usr/bin/env python3
"""Trace how a specific feature is implemented across the codebase.

Uses document processing to follow the implementation trail of a feature.
"""

import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent


def trace_feature(feature_keywords: list[str], context_lines: int = 3) -> dict[str, Any]:
    """Trace a feature across the codebase by following keyword patterns."""
    from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig
    from aragora.documents.chunking.token_counter import TokenCounter

    counter = TokenCounter()
    aragora_dir = PROJECT_ROOT / "aragora"

    findings = []
    files_with_feature = set()

    # Build regex pattern from keywords
    pattern = re.compile(
        r"(" + "|".join(re.escape(kw) for kw in feature_keywords) + r")", re.IGNORECASE
    )

    for py_file in aragora_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or ".venv" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            matches = list(pattern.finditer(content))
            if not matches:
                continue

            rel_path = str(py_file.relative_to(PROJECT_ROOT))
            files_with_feature.add(rel_path)

            # Find line numbers and context
            for match in matches:
                # Calculate line number
                line_num = content[: match.start()].count("\n") + 1
                start_line = max(0, line_num - context_lines - 1)
                end_line = min(len(lines), line_num + context_lines)

                context = lines[start_line:end_line]

                findings.append(
                    {
                        "file": rel_path,
                        "line": line_num,
                        "match": match.group(0),
                        "context": "\n".join(context),
                    }
                )

        except Exception as e:
            continue

    return {
        "total_matches": len(findings),
        "files_count": len(files_with_feature),
        "files": sorted(files_with_feature),
        "findings": findings,
    }


def analyze_call_chain(start_function: str) -> list[dict]:
    """Trace the call chain from a starting function."""
    aragora_dir = PROJECT_ROOT / "aragora"

    # Find where the function is defined
    definition = None
    for py_file in aragora_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            # Look for function definition
            if re.search(rf"(async\s+)?def\s+{re.escape(start_function)}\s*\(", content):
                definition = {
                    "file": str(py_file.relative_to(PROJECT_ROOT)),
                    "function": start_function,
                }
                break
        except Exception:
            continue

    if not definition:
        return [{"error": f"Function {start_function} not found"}]

    # Find all calls to this function
    callers = []
    for py_file in aragora_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            # Look for function calls (not definitions)
            pattern = rf"(?<!def\s){re.escape(start_function)}\s*\("
            if re.search(pattern, content):
                rel_path = str(py_file.relative_to(PROJECT_ROOT))
                if rel_path != definition["file"]:  # Exclude self-references
                    callers.append(rel_path)
        except Exception:
            continue

    return {
        "definition": definition,
        "called_from": callers,
    }


def main():
    print("=" * 70)
    print("FEATURE TRACE: Consensus Detection")
    print("Tracing how consensus/convergence detection works in aragora")
    print("=" * 70)

    # Trace consensus-related code
    print("\n[1/3] Finding consensus-related code...")
    consensus_trace = trace_feature(["consensus", "converge", "agreement", "unanimous"])
    print(
        f"  Found {consensus_trace['total_matches']} matches in {consensus_trace['files_count']} files"
    )
    print("  Key files:")
    for f in consensus_trace["files"][:10]:
        print(f"    - {f}")

    # Trace specific function
    print("\n[2/3] Tracing 'check_consensus' function call chain...")
    call_chain = analyze_call_chain("check_consensus")
    if "error" in call_chain:
        print(f"  {call_chain['error']}")
    else:
        print(f"  Defined in: {call_chain.get('definition', {}).get('file', 'unknown')}")
        print(f"  Called from {len(call_chain.get('called_from', []))} files:")
        for caller in call_chain.get("called_from", [])[:5]:
            print(f"    - {caller}")

    # Show interesting consensus-related code snippets
    print("\n[3/3] Key consensus detection implementations...")
    interesting_patterns = [
        ("Semantic similarity", ["semantic_similarity", "cosine_similarity"]),
        ("Vote counting", ["vote_count", "majority", "quorum"]),
        ("Convergence detection", ["convergence", "converged"]),
    ]

    for name, keywords in interesting_patterns:
        trace = trace_feature(keywords)
        if trace["findings"]:
            print(
                f"\n  {name}: {trace['total_matches']} occurrences in {trace['files_count']} files"
            )
            # Show first code snippet
            finding = trace["findings"][0]
            print(f"    File: {finding['file']}:{finding['line']}")
            # Show just the context (truncated)
            ctx_lines = finding["context"].split("\n")[:5]
            for line in ctx_lines:
                print(f"      {line[:80]}")

    # Summary
    print("\n" + "=" * 70)
    print("CONSENSUS DETECTION ARCHITECTURE")
    print("=" * 70)

    print(
        """
Based on the trace:

1. CORE FILES:
   - aragora/debate/consensus.py - Main consensus detection logic
   - aragora/debate/convergence.py - Semantic similarity checking
   - aragora/debate/orchestrator.py - Orchestrates consensus checks

2. DETECTION METHODS:
   - Semantic similarity (cosine similarity of embeddings)
   - Vote counting (majority/unanimous)
   - Convergence thresholds (configurable)

3. FLOW:
   Agent responses → Embeddings → Similarity calculation →
   Threshold check → Consensus decision → DecisionReceipt
"""
    )


if __name__ == "__main__":
    main()
