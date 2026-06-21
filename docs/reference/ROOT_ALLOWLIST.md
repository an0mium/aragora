# Repo-root allowlist

The repository root must stay curated: only the files listed below may be tracked
directly at the root of `synaptent/aragora`. Everything else belongs in a
subdirectory. The list is enforced by `scripts/ci/check_root_allowlist.py`, which
exits non-zero (naming the offender) when a tracked root file is not listed here.

This guard was introduced with the P1 repo-root hygiene cleanup (HEALTH-1 #8258),
which untracked stale clutter (marketing screenshots, a stale generated
`openapi.json` whose canonical copy lives at `docs/api/openapi.json`, an
inbox-triage dump, and two stray root test files). The only binary assets that
remain at the root are the brand images `aragora_logo.png` and `favicon.png`.

## How to update

When you legitimately need a new root-level file, add its exact name between the
markers below (keep the list sorted). To remove clutter instead, run
`git rm --cached <file>` and add a matching pattern to `.gitignore`. Regenerate
the candidate set with `git ls-files | grep -v /`.

## Allowed root files

<!-- ROOT_ALLOWLIST_BEGIN -->
```text
.dockerignore
.env.example
.env.production.example
.gitattributes
.gitguardian.yaml
.gitignore
.gitleaks.toml
.importlinter
.mypy-baseline
.pre-commit-config.yaml
.trivy.yaml
.trivyignore
AGENTS.md
CHANGELOG.md
CLAUDE.md
CODE_OF_CONDUCT.md
CONTRIBUTING.md
DEVELOPMENT.md
Dockerfile
INSTALL.md
Idea-to-Execution-Pipeline-Research.md
LICENSE
MANIFEST.in
Makefile
NEXT_STEPS.md
README.md
ROADMAP.md
SECURITY.md
SECURITY_AUDIT_INPUT_VALIDATION.md
THIRD_PARTY_LICENSES.md
TROUBLESHOOTING.md
action.yml
alembic.ini
aragora_logo.png
docker-compose.dev.yml
docker-compose.production.yml
docker-compose.quickstart.yml
docker-compose.simple.yml
docker-compose.sme.yml
docker-compose.yml
favicon.png
github-app-manifest.json
k8s
pyproject.toml
requirements.txt
run_staging.py
sitecustomize.py
trivy.yaml
uv.lock
```
<!-- ROOT_ALLOWLIST_END -->
