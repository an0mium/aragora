from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERCEL_WORKFLOWS = [
    Path(".github/workflows/deploy-frontend.yml"),
    Path(".github/workflows/deploy-secure.yml"),
]


def test_production_vercel_deploys_use_tgz_archive_upload() -> None:
    deploy_lines = 0
    violations: list[str] = []

    for rel_path in VERCEL_WORKFLOWS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "npx vercel deploy" not in line or "--prod" not in line:
                continue
            deploy_lines += 1
            if "--archive=tgz" not in line:
                violations.append(f"{rel_path}:{line_number}: {line.strip()}")

    assert deploy_lines == 2
    assert not violations, "Vercel production deploys must use --archive=tgz: " + "; ".join(
        violations
    )
