# Gastown + Moltbot Parity Matrix (Extension Targets)

Status legend:
- present: capability exists in Aragora core as a first-class feature
- partial: related capability exists but lacks the specific workflow/UX/semantics
- missing: no clear equivalent in Aragora core today

This matrix treats Gastown and Moltbot parity as extension layers on top of the
Aragora enterprise decision control plane.

## Core invariants (Aragora must remain)
- Enterprise decision control plane
- Multi-agent vetted decisionmaking and audit trails
- Evidence-based outputs and defensible receipts
- Multi-channel delivery

## Gastown parity (developer orchestration extension)

| Feature | Requirement (Gastown) | Aragora status | Notes / gap summary |
| --- | --- | --- | --- |
| Mayor coordinator | Primary AI coordinator with workspace context | partial | Aragora has multi-agent debate orchestration but no explicit single "Mayor" operator role or CLI session concept.
| Town workspace | Root workspace for projects/agents/config | partial | Aragora has project structures, but no dedicated workspace manager layer.
| Rigs (project containers) | Per-repo containers with agent contexts | partial | Aragora has agents + debates, but not per-repo "rig" abstraction with shared agent pools.
| Crew members | User workspace in a rig | partial | Aragora supports human-in-the-loop but lacks explicit crew workspace semantics.
| Polecats | Ephemeral worker agents | present/partial | Aragora can spawn agents; needs explicit lifecycle + task-bound ephemeral UX.
| Hooks | Git worktree-based persistent storage | missing | No git-hook/worktree persistence layer in Aragora core.
| Convoys | Work tracking units | missing | Aragora has audit trails but not convoy task packaging or status UI.
| Beads ledger | Git-backed issue tracking | missing | No Beads-equivalent issue ledger in core.
| Mailboxes/handoffs | Built-in agent mail + handoff | partial | Some debate/message flows exist, but not durable agent mailboxes.
| Dashboard | Convoy and hook state visibility | partial | Aragora has debate streaming; needs workspace/convoy dashboards.
| CLI workflows | "gt" command flows | partial | Aragora has CLI, but not Gastown workflows.

## Moltbot parity (consumer/device extension)

| Feature | Requirement (Moltbot) | Aragora status | Notes / gap summary |
| --- | --- | --- | --- |
| Local-first assistant | Runs on user device | missing | Aragora is server/control-plane oriented.
| Multi-channel inbox | WhatsApp/Telegram/etc as primary UI | partial | Aragora supports channels, but not consumer "inbox" model.
| Voice wake/talk | On-device speech I/O | missing | Not documented as a first-class capability.
| Live Canvas (A2UI) | Real-time visual canvas UI | missing | No equivalent surface.
| Gateway control plane | Local gateway for routing | partial | Aragora has control-plane concepts; needs local gateway service.
| Onboarding wizard | Guided setup CLI/UI | missing | Not documented as a first-class flow.
| Device nodes | Companion apps / device capabilities | missing | No device-node model in core.
| Skill marketplace | User-installed skills | partial | Aragora has skills but not consumer skill store UX.
| Security pairing | Allowlist / pairing | partial | RBAC exists for enterprise; consumer pairing UX is missing.

## Cross-cutting extension requirements

| Capability | Purpose | Aragora status | Notes |
| --- | --- | --- | --- |
| Agent Fabric | High-scale scheduling + isolation | partial | Orchestration exists; needs large-N agent runtime management.
| Policy engine | Tool access, approvals, sandboxing | partial | RBAC exists, but device-level policy and approvals needed.
| Audit + replay | Every device action logged | present | Aragora already emphasizes audit trails; extend to device actions.
| Cost + budget controls | Prevent runaway usage | partial | Budget controls exist in roadmap; enforce at agent fabric level.
| Safe computer use | UI/browser/shell with approvals | missing | Requires new sandboxed execution surface.

