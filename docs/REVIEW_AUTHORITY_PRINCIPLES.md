# Review Authority Principles

Review authority in Aragora rests on four factors: competence, independence, accountability, and stake. An approval matters only when the approver can judge the change, is not merely echoing the system being judged, bears responsibility for the outcome, and is exposed to the consequences of being wrong.

This document describes the principles underlying current review authority. It does not authorize any change to current human-settlement requirements. Any future adjustment to approver authority requires a separate design proposal, explicit founder sign-off, and demonstrated calibration evidence from post-merge outcomes, not citation of this document.

Competence in this repo has two forms. Object-level competence is direct judgment about code, tests, and failure modes in a specific diff. Governance competence is judgment about the evidence around a diff: whether the scope is bounded, whether validation is credible, whether competing analyses agree, and whether the claimed risk matches the actual change. Aragora uses both forms. The system prepares evidence, but settlement still depends on a reviewer who can cash that evidence out into a real decision.

The founder role in this workflow is governance competence plus accountability and stake, not object-level competence on every line of every PR. That role is still real. It is the place where bounded evidence, receipts, and competing analysis become an accepted or rejected risk.

Current AI reviewers in this codebase have not demonstrated sufficient calibration for authority expansion. They are useful for summarizing diffs, surfacing risks, checking consistency, and preparing packets, but they do not currently replace human settlement in Aragora's live review path. This document is not a path to reducing human settlement.

These principles fit the repo's existing pillars rather than adding new ones. Receipts bind decisions to the exact reviewed state. Evidence-first review requires concrete validation and current-head truth before a merge decision is meaningful. Bounded scope keeps approvals legible enough that an approver can understand what is being accepted. The goal is not symbolic human presence. The goal is a reviewer with enough competence, independence, accountability, and stake to make the approval mean something.
