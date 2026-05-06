# Agent Flywheel Local Pilot Runbook

This runbook validates Agent Flywheel-style tools on the local workstation first.
Local validation comes before AWS because the active Claude, Codex, Gemini, and
OpenRouter CLI credentials are part of the substrate being tested.

## Rules

- Do not run a broad workstation installer blindly.
- Inspect install scripts, license files, shell-profile edits, daemon behavior,
  network calls, and cleanup paths before executing anything.
- Pin exact commits for any external repo cloned for inspection.
- Keep external clones and logs outside this repository, preferably under
  `~/.aragora/flywheel-lab/`.
- Use a toy repository or disposable Aragora worktree for exercises.
- Do not use Homebrew, global shell-profile mutation, launch agents, or daemon
  installation without explicit operator approval.
- Do not run paid model calls as part of this pilot.
- Do not install the full stack on GitHub runners.

## Local Lab Layout

Recommended scratch layout:

```text
~/.aragora/flywheel-lab/
  repos/
  installs/
  logs/
  toy-repo/
  manifest.json
```

Record each external repo as:

```json
{
  "repo": "https://github.com/Dicklesworthstone/ntm",
  "commit": "<pinned sha>",
  "license_checked": true,
  "install_surface": "user-space venv only",
  "notes": "no shell profile mutation"
}
```

## First Validation Pass

1. Run the Aragora read-only probe:

   ```bash
   python3 scripts/flywheel_tools_probe.py --json --no-help
   ```

2. Pick at most one session-oriented tool for a manual lab exercise. Prefer
   `ntm` or session search before Agent Mail, destructive-command wrappers, or
   full ACFS bootstrap.

3. Run the selected tool only inside the lab scratch directory or toy repo.

4. Capture:

   - exact repo URL and commit
   - install command used
   - changed files outside `~/.aragora/flywheel-lab/`, if any
   - whether local Claude/Codex/Gemini auth was detected or required
   - useful patterns Aragora should reimplement

## AWS Deferral

AWS is a later portability check. It should validate clean bootstrap and
repeatability after local value is proven. AWS should not replace local testing
because a fresh instance will not have the workstation's active agent CLI auth.

## GitHub Runner Boundary

GitHub runners may later use narrow, noninteractive adapters such as a JSON
probe or task-graph export. They should not run full tmux swarm bootstrap or
credential-dependent local-agent workflows by default.
