# When To Use Aragora Vs Execution Substrates

This is the practical decision table for operating Aragora alongside tools like
Codex, Claude Code, OpenCode, and Pi.

## Default Rule

Use the **simplest layer that preserves the needed truthfulness**.

If the task only needs raw execution, use a worker runtime.
If the task needs receipts, review, provenance, or truthful blocker handling,
use Aragora.

## Decision Table

| Situation | Best default | Why |
|---|---|---|
| Small code edit with clear scope | Single strong coding agent | Lowest coordination overhead |
| One owner, 2-4 bounded parallel subtasks | Lead agent plus bounded subagents | Keeps ownership clear while getting real parallelism |
| Vague natural-language request | Lead agent first | Someone has to frame, slice, and own integration |
| High-stakes review or merge decision | Aragora | Receipts, dissent, gates, and blocker truth matter |
| Unattended multi-step execution | Aragora | Queue, watch, integrate, and truthful terminalization are the point |
| Pure terminal productivity for one developer | Codex / Claude Code / OpenCode / Pi | Fastest path, lowest ceremony |
| Model-routing experiments | OpenCode/Pi or direct worker harnesses | Good substrate for worker-level routing |
| Building Aragora itself | Lead agent plus bounded subagents, optionally under Aragora for proof runs | Orchestration overhead must stay bounded |

## Recommended Operating Modes

### 1. Single-agent mode

Use when:

- the task is small
- scope is obvious
- integration risk is low

Best tools:

- Codex
- Claude Code
- other direct coding agents

### 2. Lead agent plus bounded subagents

Use when:

- the prompt is vague
- you need a decomposition pass
- there are a few independent sidecar tasks

This is the default mode for building Aragora itself.

Why:

- one agent owns framing and integration
- a few workers handle isolated slices
- coordination stays legible

### 3. Aragora control-plane mode

Use when:

- the work is consequential
- a receipt is valuable
- review and publish behavior must be explicit
- unattended execution needs truthful stopping behavior

Why:

- Aragora's value starts where worker runtimes stop
- it owns governance, not just execution

### 4. Worker-runtime mode

Use OpenCode, Pi, Codex, Claude Code, or similar tools directly when:

- you mainly need execution speed
- auditability is not the main bottleneck
- the task does not need a control plane

These tools are better treated as substrates than as strategic enemies.

## A Note On "Whole Orchestras"

Large heterogeneous swarms sound appealing, but they fail quickly when:

- the prompt is vague
- scopes overlap
- state propagation lies
- verification evidence is weak
- no one clearly owns integration

So the default should not be "spawn a huge orchestra."

The better sequence is:

1. one lead agent frames the task
2. a few bounded workers handle independent slices
3. Aragora governs when the work becomes consequential enough to need receipts, gates, and truthfulness

## Strategic Implication

Aragora should not try to beat OpenCode or Pi at being execution substrates.

Aragora should sit above them:

- selecting when deeper governance is needed
- preserving receipts and provenance
- making disagreement useful
- turning "needs human" into a precise, low-cost next action
