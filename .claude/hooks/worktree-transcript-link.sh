#!/usr/bin/env bash
# worktree-transcript-link.sh — Stop hook that symlinks transcript files
# for worktree sessions so plugins (e.g. claude-mem) can find them.
#
# Problem: When Claude Code runs in a worktree (e.g. .claude/worktrees/foo),
# it creates a project dir based on the worktree path, but the transcript
# .jsonl lives under the original project dir. Plugins that read the
# transcript via the worktree project path get "file not found".
#
# Solution: Before plugins run, symlink the transcript from the original
# project dir into the worktree project dir.
#
# Safe for multiple concurrent sessions — each has a unique session UUID.

set -euo pipefail

CWD="$(pwd)"

# Only act if we're in a worktree
if [[ "$CWD" != *"/.worktrees/"* ]] && [[ "$CWD" != *"/.claude/worktrees/"* ]]; then
  exit 0
fi

CLAUDE_DIR="${HOME}/.claude/projects"

# Derive project dir names from paths (same mangling Claude Code uses: / → -)
worktree_project_dir="${CLAUDE_DIR}/$(echo "$CWD" | sed 's|^/||; s|/|-|g')"

# Find the main repo root by walking up from CWD to find the parent of .claude/worktrees
main_repo="$CWD"
while [[ "$main_repo" == *"/.claude/worktrees/"* ]] || [[ "$main_repo" == *"/.worktrees/"* ]]; do
  # Strip everything from /.claude/worktrees/ or /.worktrees/ onwards
  if [[ "$main_repo" == *"/.claude/worktrees/"* ]]; then
    main_repo="${main_repo%%/.claude/worktrees/*}"
  elif [[ "$main_repo" == *"/.worktrees/"* ]]; then
    main_repo="${main_repo%%/.worktrees/*}"
  fi
done

main_project_dir="${CLAUDE_DIR}/$(echo "$main_repo" | sed 's|^/||; s|/|-|g')"

# If main and worktree project dirs are the same, nothing to do
if [[ "$worktree_project_dir" == "$main_project_dir" ]]; then
  exit 0
fi

# Ensure worktree project dir exists
mkdir -p "$worktree_project_dir"

# Symlink any .jsonl transcript files that exist in main but not in worktree
if [[ -d "$main_project_dir" ]]; then
  for jsonl in "$main_project_dir"/*.jsonl; do
    [[ -f "$jsonl" ]] || continue
    basename="$(basename "$jsonl")"
    target="$worktree_project_dir/$basename"
    if [[ ! -e "$target" ]]; then
      ln -sf "$jsonl" "$target"
    fi
  done
fi
