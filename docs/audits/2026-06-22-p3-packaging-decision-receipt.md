# P3 Packaging Decision Receipt (#8263)

Dogfood decision receipt for the decision-gated packaging issue
[#8263](https://github.com/synaptent/aragora/issues/8263) ("one installable
story — unify PyPI naming + console entry points"). The decision gate required
running an Aragora debate on the packaging split question and attaching the
resulting **decision receipt** before implementing. This document records that
receipt; it is a post-hoc record because the chosen option already shipped on
`main` (see "Outcome" below).

Companion design doc:
[`../architecture/PACKAGING_AND_DISTRIBUTION.md`](../architecture/PACKAGING_AND_DISTRIBUTION.md)
(full package strategy + traced dependency audit table).

## The decision question (verbatim framing from #8263)

- **Option A** — one name (`aragora`) ships the full package including the CLI
  and server console entry points (`aragora = aragora.cli.main:main`), keeping
  `aragora-debate/` as a separate standalone wedge.
- **Option B** — a minimal wedge package (receipts + offline verifier, aligned
  with ODR-3 #8226) plus a full meta-package depending on it.

Selection criterion: which option best delivers a single clean documented
`pip install` → CLI → zero-key decision-receipt path while honoring ODR-3
([#8226](https://github.com/synaptent/aragora/issues/8226)), which ships a
separate standalone `aragora-verify` offline receipt verifier.

## Dogfood debate

The decision was stress-tested with Aragora's own debate engine
(`python3 -m aragora.cli.main ask ... --consensus judge`), heterogeneous
model quorum, judge consensus.

| Field | Value |
|---|---|
| receipt_id | `debate-9ea6b178-a438-4964-9836-3ba84230bd03` |
| debate_id | `9ea6b178-a438-4964-9836-3ba84230bd03` |
| timestamp | `2026-06-22T22:11:52Z` |
| agents (heterogeneous) | `grok`, `mistral-api`, `deepseek` |
| agents_failed | none |
| rounds_used | 1 |
| consensus | judge; `consensus_reached = true` |
| winning position | `grok` |
| dissenting_views | 0 |
| verdict | **PASS** |
| confidence | 0.80 |
| robustness_score | 0.80 |
| input_hash | `05e1b2f374b5708e925f11c05ec1f00bd230aa3b0b5af350fb26f70a85904fc8` |
| content_hash | `275a182b60fc8e1c` |
| artifact_hash / checksum | `b665b43f8af31673911b3676a115182c3f3e971b0d6012970433b3a7a8559655` |
| schema_version | 1.1 |
| receipt sha256 | `5fa4f45c10e4925df8ff7b95757ba861beccb62772d1072da53ec2a7fea0e6b1` |

### Synthesis (winning position, abridged)

The judge synthesis favored the **unified path**: "A single encompassing
approach carries clear strengths ... this unity reduces the cognitive load" of
moving from intent to a tangible result, while "the presence of more focused
companions for those who already know their precise need preserves autonomy
without forcing the majority through extra thresholds." The modular alternative
was recognized as valid where needs genuinely diverge ("a solitary walker may
prefer a simple staff rather than an entire traveler's kit"), but it "risks
introducing hesitation at the very moment when momentum toward purpose matters
most." The debate explicitly respected the concern that a unified offering must
**not entangle the independent core verifier** — i.e. ODR-3's standalone
`aragora-verify` must stay separable.

The receipt JSON is reproducible from the loop and verifiable offline:

```
aragora receipt verify <path>/9ea6b178-a438-4964-9836-3ba84230bd03_275a182b60fc8e1c.json
```

## Decision

**Option A (unified `aragora` distribution), with standalone wedges preserved.**

This is effectively a synthesis that keeps Option A's single clean install story
while retaining the separable wedges that Option B and ODR-3 care about:

1. The root distribution is renamed `aragora-debate` → `aragora` and ships the
   full package (auto-discovery `include = ["aragora*"]`) with a console entry
   point `aragora = aragora.cli.main:main`.
2. `aragora-debate/` remains an unchanged standalone wedge.
3. ODR-3's `aragora-verify` stays a **separate** near-zero-dependency package;
   the root `aragora` distribution does not absorb the offline verifier, so the
   "receipt is trustworthy outside Aragora" property is preserved. ODR-3 leads
   on receipt-surface naming.

## Outcome (already shipped on main)

The chosen Option A landed via **#8517** (merged on `main`, commit `4cf2fd74db`,
"feat(packaging): root distribution becomes installable `aragora` (P3)"). This
receipt is therefore a post-hoc record of the decision gate, not a
pre-implementation hold. Verified on `origin/main`:

- `[project].name == "aragora"`
- `[project.scripts].aragora == "aragora.cli.main:main"`
- `[tool.setuptools.packages.find].include == ["aragora*"]`
- `aragora-debate/pyproject.toml` `[project].name == "aragora-debate"` (wedge intact)

## Cross-references

- Issue: #8263 (decision-gated packaging) — this receipt satisfies its decision gate.
- ODR-3 coordination: #8226 (standalone `aragora-verify`) — dependency audit shared there.
- Source audit baseline: [`2026-06-12-codebase-health-audit.md`](2026-06-12-codebase-health-audit.md).
- Implementation: PR #8517 (root distribution rename, merged).
