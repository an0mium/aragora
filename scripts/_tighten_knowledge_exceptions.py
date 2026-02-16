#!/usr/bin/env python3
"""Tighten except Exception catches in aragora/knowledge/ directory.

Replaces bare `except Exception` with specific exception tuples based on context.
Uses heuristics based on file path and surrounding code to determine appropriate types.
"""

import re
import os
import sys

# Mapping of context patterns to replacement exception tuples
# Key: regex pattern to match in surrounding lines (within 5 lines above)
# Value: replacement tuple string

REDIS_TUPLE = "(OSError, ConnectionError, RuntimeError)"
DB_TUPLE = "(RuntimeError, ValueError, OSError)"
ADAPTER_INNER_TUPLE = "(RuntimeError, ValueError, AttributeError, KeyError)"
ADAPTER_OUTER_TUPLE = "(RuntimeError, ValueError, OSError, AttributeError)"
API_TUPLE = "(OSError, ConnectionError, RuntimeError, ValueError)"
JSON_TUPLE = "(ValueError, KeyError, TypeError)"
NETWORK_TUPLE = "(OSError, ConnectionError, TimeoutError, RuntimeError)"
HEALTH_TUPLE = "(RuntimeError, ValueError, TypeError, AttributeError)"
CALLBACK_TUPLE = "(RuntimeError, ValueError, TypeError, AttributeError)"
STORE_TUPLE = "(RuntimeError, ValueError, OSError, AttributeError)"
QUERY_TUPLE = "(RuntimeError, ValueError, OSError, AttributeError, KeyError)"

def get_context_lines(lines, idx, window=8):
    """Get surrounding lines for context analysis."""
    start = max(0, idx - window)
    end = min(len(lines), idx + window)
    return "\n".join(lines[start:end])

def determine_replacement(lines, idx, filepath):
    """Determine the replacement exception tuple based on context."""
    context = get_context_lines(lines, idx)
    line = lines[idx]

    # Check indentation level - deeper = inner (per-item) catch
    indent = len(line) - len(line.lstrip())

    # Determine if this is a noqa-worthy catch (inner per-item isolation)
    is_inner = indent >= 16  # Deep nesting typically means per-item processing

    # Check for specific patterns in context
    ctx_lower = context.lower()

    # Redis operations
    if any(kw in ctx_lower for kw in ['redis', 'cache_key', 'cache.get', 'cache.set', 'r.get', 'r.set', 'r.hget', 'r.expire']):
        return REDIS_TUPLE, is_inner

    # JSON parsing
    if any(kw in ctx_lower for kw in ['json.loads', 'json.dumps', 'json_data', 'json.decoder', 'jsondecodeerror']):
        return JSON_TUPLE, is_inner

    # Health checks
    if any(kw in ctx_lower for kw in ['health_check', 'is_healthy', 'health_status', 'readiness']):
        return HEALTH_TUPLE, True  # Health checks always noqa

    # Event callbacks
    if any(kw in ctx_lower for kw in ['_emit_event', 'event_callback', 'callback(', '_event_callback']):
        return CALLBACK_TUPLE, True  # Callbacks always noqa

    # Network/API operations
    if any(kw in ctx_lower for kw in ['connect', 'request', 'fetch', 'http', 'grpc', 'timeout', 'socket']):
        return NETWORK_TUPLE, is_inner

    # Database/store operations
    if any(kw in ctx_lower for kw in ['postgres', 'sqlite', 'database', 'cursor', 'execute', 'query(', 'pool']):
        return DB_TUPLE, is_inner

    # File path based heuristics
    basename = os.path.basename(filepath)
    dirpath = os.path.dirname(filepath)

    # Vector abstraction layer
    if 'vector_abstraction' in dirpath:
        return API_TUPLE, is_inner

    # Resilience layer
    if 'resilience' in dirpath:
        if 'circuit' in basename:
            return "(RuntimeError, OSError, ConnectionError, TimeoutError)", is_inner
        return STORE_TUPLE, is_inner

    # API layer
    if '/api/' in dirpath:
        return QUERY_TUPLE, is_inner

    # Culture layer
    if 'culture' in dirpath:
        return ADAPTER_OUTER_TUPLE, is_inner

    # Adapter files
    if 'adapter' in basename or 'adapters' in dirpath:
        if is_inner:
            return ADAPTER_INNER_TUPLE, True
        return ADAPTER_OUTER_TUPLE, False

    # Ops files
    if '/ops/' in dirpath:
        if is_inner:
            return ADAPTER_INNER_TUPLE, True
        return ADAPTER_OUTER_TUPLE, False

    # Default for mound files
    if 'mound' in dirpath:
        if is_inner:
            return ADAPTER_INNER_TUPLE, True
        return STORE_TUPLE, False

    # Default for top-level knowledge files
    if is_inner:
        return ADAPTER_INNER_TUPLE, True
    return STORE_TUPLE, False


def process_file(filepath):
    """Process a single file, replacing except Exception catches."""
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    replacements = 0
    new_lines = []

    for i, line in enumerate(lines):
        # Match except Exception patterns
        match = re.match(r'^(\s*)except Exception(\s+as\s+\w+)?:', line)
        if match:
            indent = match.group(1)
            as_clause = match.group(2) or ""

            # Skip the intentional savepoint rollback catch
            if 'transaction.py' in filepath and 'ROLLBACK TO SAVEPOINT' in get_context_lines(lines, i, 3):
                new_lines.append(line)
                continue

            # Determine replacement
            tuple_str, needs_noqa = determine_replacement(lines, i, filepath)

            # Check if there's already a noqa comment
            noqa_suffix = ""
            if needs_noqa:
                noqa_suffix = "  # noqa: BLE001 - adapter isolation"

            new_line = f"{indent}except {tuple_str}{as_clause}:{noqa_suffix}"
            new_lines.append(new_line)
            replacements += 1
        else:
            new_lines.append(line)

    if replacements > 0:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))

    return replacements


def main():
    """Process all knowledge files."""
    import subprocess

    # Find all files with except Exception
    result = subprocess.run(
        ['grep', '-rl', 'except Exception', 'aragora/knowledge/', '--include=*.py'],
        capture_output=True, text=True, cwd='/Users/armand/Development/aragora'
    )

    files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

    total_replacements = 0
    total_files = 0

    for filepath in sorted(files):
        full_path = os.path.join('/Users/armand/Development/aragora', filepath)
        count = process_file(full_path)
        if count > 0:
            total_files += 1
            total_replacements += count
            print(f"  {filepath}: {count} replacements")

    print(f"\nTotal: {total_replacements} replacements across {total_files} files")


if __name__ == '__main__':
    main()
