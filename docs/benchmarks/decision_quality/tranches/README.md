# Decision Quality Corpus Tranches

These files are construction inputs for the outcome-backed decision-quality
benchmark. They are not independently eligible for a counted benchmark run.

`software-development-1` contributes the four software-engineering development
cases required by the planned 24-case corpus. Every evidence URL is pinned to
an immutable Git commit or release tag, and the outcome answer key remains in a
separate hash-bound sidecar.

`business-operations-1` contributes four business/operations development
cases. Its acquisition scenarios deliberately balance two completed and two
terminated transactions. Pre-cutoff packets combine the signed transaction
with a public litigation or regulator signal; outcomes remain in the separate
hash-bound sidecar.

Current canonical digests:

- corpus: `aae58206475930742377b9a75f2f62f7e394e52f127fa97960d00eb8a651dd9c`
- outcome sidecar: `dbce998d194fa3bb9fef6167902bc27a1445ab38e0d4d21378360503d8a97bb6`

Business/operations tranche:

- corpus: `734f515a6cff55e88faa8de2d4ff5bf32e42385bbe8ee109b68eb6df54ef8661`
- outcome sidecar: `896a4e7f6b49c6cc7f0474e75a8b835619ef462fe142a53cd957e6e4d4ec9277`

The tranche passes the decision-quality corpus validator with
`--allow-partial`. That flag skips only the final 24-case balance requirement.
It does not relax source cutoffs, outcome separation, digest binding, or
per-case semantics.

Together the two tranches provide eight of the planned 24 cases: eight
development cases, four each in software engineering and business/operations.

Do not run model inference from a tranche. Counted inference begins only after
all 24 cases, the scoring contract, prompts, roster, and both corpus digests are
merged and frozen together.
