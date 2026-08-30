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

`policy-compliance-1` contributes four nonpartisan policy/compliance
development cases. The tranche balances two rules published within the stated
horizon and two standards or rules published after the stated horizon. All
model-visible evidence is an official pre-cutoff publication; later outcome
evidence remains in the hash-bound sidecar.

`science-forecasting-1` contributes four science/forecasting development
cases. It balances two observed outcomes that met the forecast threshold and
two delayed missions that did not. Model-visible evidence is limited to
official pre-cutoff NASA or NOAA material; the measured outcomes remain in the
hash-bound sidecar.

`software-engineering-holdout-1` contributes the two software-engineering
holdout cases. One roadmap commitment was fulfilled and one announced language
semantic change did not ship on schedule. Both cases use immutable upstream
release or source-history evidence, and their outcomes remain in the separate
hash-bound sidecar.

`business-operations-holdout-1` contributes the two business/operations
holdout cases. One regulatory return-to-service milestone was met and one
aircraft delivery milestone was formally delayed. Both cases use immutable SEC
filings, with the outcome answer key kept in the separate hash-bound sidecar.

Current canonical digests:

- corpus: `aae58206475930742377b9a75f2f62f7e394e52f127fa97960d00eb8a651dd9c`
- outcome sidecar: `dbce998d194fa3bb9fef6167902bc27a1445ab38e0d4d21378360503d8a97bb6`

Business/operations tranche:

- corpus: `734f515a6cff55e88faa8de2d4ff5bf32e42385bbe8ee109b68eb6df54ef8661`
- outcome sidecar: `896a4e7f6b49c6cc7f0474e75a8b835619ef462fe142a53cd957e6e4d4ec9277`

Policy/compliance tranche:

- corpus: `4247cf544b5c12f939517b2267f915559df85b059a164c457a40af74ffd88ea3`
- outcome sidecar: `0972e27803a6b4b73ed15f553ba8847692c319ee2b882e7b50c84dcf5daea52e`

Science/forecasting tranche:

- corpus: `3f5ea13fac70579c8422b08f820a96a3ebd266e7cd2db174d381a24f47bc491e`
- outcome sidecar: `f86c8ee9b8ee6694595ebdc11a758569262a20176771bf7f9d7067cdf323e090`

Software-engineering holdout tranche:

- corpus: `4d13fea53e1313c2db332bb932ab57abac5170a9779eba779cef4c3c712af2c6`
- outcome sidecar: `bae34db3f7928ade185e2975849e609139c59d7c64b496909a6b4137c3d957d4`

Business/operations holdout tranche:

- corpus: `26473525c2ab4fbd7298d307292b42859540532aa62e24be08bf934208cd9b1d`
- outcome sidecar: `e0e44873391ddd2836bc6f4158ed5a82f1c4b1900d132e228ca918eef2657a47`

Each tranche passes the decision-quality corpus validator with
`--allow-partial`. That flag skips only the final 24-case balance requirement.
It does not relax source cutoffs, outcome separation, digest binding, or
per-case semantics.

Together the six tranches provide twenty of the planned 24 cases: four
development cases in each required domain, both software-engineering holdouts,
and both business/operations holdouts.

Do not run model inference from a tranche. Counted inference begins only after
all 24 cases, the scoring contract, prompts, roster, and both corpus digests are
merged and frozen together.
