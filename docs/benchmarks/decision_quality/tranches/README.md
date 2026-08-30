# Decision Quality Corpus Tranches

These files are construction inputs for the outcome-backed decision-quality
benchmark. They are not independently eligible for a counted benchmark run.

`software-development-1` contributes the four software-engineering development
cases required by the planned 24-case corpus. Every evidence URL is pinned to
an immutable Git commit or release tag, and the outcome answer key remains in a
separate hash-bound sidecar.

Current canonical digests:

- corpus: `aae58206475930742377b9a75f2f62f7e394e52f127fa97960d00eb8a651dd9c`
- outcome sidecar: `dbce998d194fa3bb9fef6167902bc27a1445ab38e0d4d21378360503d8a97bb6`

The tranche passes the decision-quality corpus validator with
`--allow-partial`. That flag skips only the final 24-case balance requirement.
It does not relax source cutoffs, outcome separation, digest binding, or
per-case semantics.

Do not run model inference from a tranche. Counted inference begins only after
all 24 cases, the scoring contract, prompts, roster, and both corpus digests are
merged and frozen together.
