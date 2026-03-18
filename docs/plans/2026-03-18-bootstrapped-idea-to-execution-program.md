# Bootstrapped Idea-to-Execution Program

Last updated: 2026-03-18

Primary execution epic: [#989](https://github.com/synaptent/aragora/issues/989)

Related execution issues:
- [#991](https://github.com/synaptent/aragora/issues/991) Close the Ralph autonomy gap from spec to repair PR
- [#994](https://github.com/synaptent/aragora/issues/994) Add interactive stage transitions from ideas to goals to specs
- [#993](https://github.com/synaptent/aragora/issues/993) Productize the unified local-first DAG workbench
- [#990](https://github.com/synaptent/aragora/issues/990) Dogfood the pipeline to build more of Aragora itself

## Purpose

Record the near-term program for turning Aragora's existing idea/goals/actions/orchestration substrate into the primary product surface, and then using that same substrate to build more of Aragora with less repeated human prompting and copy-paste.

This document is the bridge between the thesis in `docs/CANONICAL_GOALS.md`, the long-range architecture in `docs/plans/ARAGORA_EVOLUTION_ROADMAP.md`, and the live execution order in `docs/status/NEXT_STEPS_CANONICAL.md`.

## Thesis

Aragora's next moat is not more disconnected backend primitives. It is a local-first workbench where:
- vague human ideas are upgraded into connected idea graphs
- idea graphs are upgraded into goals and principles through structured questioning
- goals are upgraded into executable task/spec DAGs
- those specs are submitted directly into the Ralph loop for implementation, review, repair, PR creation, and merge-gate waiting

The same pipeline should also increasingly be used to plan and dispatch Aragora's own work. Human input should be concentrated on priority, risk, and policy choices rather than low-leverage technical restatement.

## What Already Exists

The repo already contains substantial substrate for this program:
- idea, goal, and pipeline canvas models and storage
- `IdeaToExecutionPipeline` and stage transition infrastructure
- workflow DAG execution and orchestration components
- provenance and receipt infrastructure
- Ralph/swarm outer-loop execution with recent validation of repair and PR-gate flows

The main missing layer is productization:
- guided stage-transition UX
- a unified local-first workbench instead of separate partial surfaces
- autonomous handoff from approved specs into Ralph with no manual babysitting
- dogfooding the pipeline so roadmap work becomes executable pipeline artifacts

## Strategic Goals

1. Make Aragora a reliable spec-to-merge engine, not just a debate/orchestration demo.
2. Let idea-rich, execution-poor users reach executable specs without having to become expert spec writers.
3. Unify ideas, goals, actions, and orchestration as one DAG/provenance model.
4. Use heterogeneous models as a product advantage, with different models doing planning, implementation, review, and repair.
5. Produce benchmarked evidence and self-bootstrapping execution lanes instead of anecdotal progress.

## Operating Doctrine

- Local-first is a feature, not a temporary limitation.
- AI should ask the minimum useful questions at each stage transition.
- Humans should intervene where they add high-value judgment, not where the system is merely missing glue.
- Stage outputs must be editable, approval-gated, and provenance-preserving.
- Canonical docs and GitHub issues should increasingly be generated from pipeline artifacts rather than maintained as disconnected narratives.

## Program Lanes

| Lane | Issue | Outcome |
|------|-------|---------|
| Ralph autonomy hardening | [#991](https://github.com/synaptent/aragora/issues/991) | A clean benchmark proves spec -> repair PR -> merge-gate wait with no manual manifest or branch intervention |
| Interactive stage transitions | [#994](https://github.com/synaptent/aragora/issues/994) | Idea graphs become editable goal DAGs and then executable spec DAGs through structured questioning and approvals |
| Unified DAG workbench | [#993](https://github.com/synaptent/aragora/issues/993) | One local-first workbench shows ideas, goals, actions, orchestration, review, repair, and provenance as one flow |
| Self-bootstrapping lane | [#990](https://github.com/synaptent/aragora/issues/990) | Aragora uses its own pipeline to generate, maintain, and dispatch more of Aragora's roadmap work |

## Success Bar

This program is successful when:
- a user can go from a local idea graph to a reviewable executable spec without manual copy-paste between tools
- that spec can be submitted directly to Ralph
- the workbench can show both upstream provenance and downstream execution/review/repair state
- at least one Aragora implementation lane is driven by pipeline-generated specs and issue updates

## Relationship To Other Canonical Docs

- `docs/CANONICAL_GOALS.md` defines the why.
- `docs/plans/ARAGORA_EVOLUTION_ROADMAP.md` defines the long-range architecture and moat.
- `docs/status/NEXT_STEPS_CANONICAL.md` defines short-horizon execution order.
- `docs/status/ACTIVE_EXECUTION_ISSUES.md` maps this program to the live GitHub backlog.

This document exists to make the March 2026 bootstrapping program explicit and durable.
