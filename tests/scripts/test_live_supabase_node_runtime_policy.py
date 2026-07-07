import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = REPO_ROOT / "aragora" / "live"
NODE_RUNTIME_FILES = (
    LIVE_DIR / "Dockerfile",
    REPO_ROOT / "deploy" / "Dockerfile.frontend",
    REPO_ROOT / "docker-compose.dev.yml",
)
FRONTEND_DOCKERFILES = (
    LIVE_DIR / "Dockerfile",
    REPO_ROOT / "deploy" / "Dockerfile.frontend",
)
SUPABASE_PACKAGES = (
    "node_modules/@supabase/auth-js",
    "node_modules/@supabase/functions-js",
    "node_modules/@supabase/postgrest-js",
    "node_modules/@supabase/realtime-js",
    "node_modules/@supabase/storage-js",
    "node_modules/@supabase/supabase-js",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _node_engine_floor(engine: str) -> tuple[int, int, int]:
    match = re.search(r">=\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", engine)
    assert match, f"unsupported node engine expression: {engine!r}"
    return (
        int(match.group(1)),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
    )


def _node_image_versions(path: Path) -> list[tuple[int, int, int]]:
    text = path.read_text(encoding="utf-8")
    return [
        (
            int(match.group(1)),
            int(match.group(2) or 0),
            int(match.group(3) or 0),
        )
        for match in re.finditer(
            r"^\s*(?:FROM|image:)\s+node:(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[.\-]|$)",
            text,
            re.MULTILINE,
        )
    ]


def test_live_supabase_version_is_exactly_pinned() -> None:
    package_json = _load_json(LIVE_DIR / "package.json")
    package_lock = _load_json(LIVE_DIR / "package-lock.json")

    spec = package_json["dependencies"]["@supabase/supabase-js"]
    locked = package_lock["packages"]["node_modules/@supabase/supabase-js"]["version"]

    assert re.fullmatch(r"\d+\.\d+\.\d+", spec), (
        "pin @supabase/supabase-js exactly so minor updates cannot silently raise "
        "the live Node runtime floor"
    )
    assert locked == spec


def test_live_npm_enforces_declared_node_engine() -> None:
    package_json = _load_json(LIVE_DIR / "package.json")
    package_lock = _load_json(LIVE_DIR / "package-lock.json")
    npmrc = (LIVE_DIR / ".npmrc").read_text(encoding="utf-8")

    assert "engine-strict=true" in npmrc.splitlines()
    assert package_lock["packages"][""]["engines"] == package_json["engines"]


def test_live_realtime_phoenix_dependency_is_locked() -> None:
    package_lock = _load_json(LIVE_DIR / "package-lock.json")
    packages = package_lock["packages"]
    realtime_dependencies = packages["node_modules/@supabase/realtime-js"]["dependencies"]
    phoenix_version = packages["node_modules/@supabase/phoenix"]["version"]

    assert realtime_dependencies["@supabase/phoenix"] == phoenix_version


def test_frontend_dockerfiles_install_from_lockfile_with_sdk_context() -> None:
    for dockerfile in FRONTEND_DOCKERFILES:
        text = dockerfile.read_text(encoding="utf-8")

        assert "COPY sdk/typescript/ /sdk/typescript/" in text
        assert (
            "COPY aragora/live/package.json aragora/live/package-lock.json aragora/live/.npmrc ./"
        ) in text
        assert "RUN npm ci --legacy-peer-deps" in text
        assert "npm install --legacy-peer-deps" not in text
        assert "npm ci --only=production" not in text
        assert "sed -i" not in text
        assert "replaceAll(" not in text


def test_live_sdk_lock_path_matches_docker_context() -> None:
    package_json = _load_json(LIVE_DIR / "package.json")
    package_lock = _load_json(LIVE_DIR / "package-lock.json")

    assert package_json["dependencies"]["@aragora/sdk"] == "file:../../sdk/typescript"
    assert package_lock["packages"][""]["dependencies"]["@aragora/sdk"] == (
        "file:../../sdk/typescript"
    )
    assert package_lock["packages"]["node_modules/@aragora/sdk"]["resolved"] == (
        "../../sdk/typescript"
    )


def test_pre_deploy_rejects_sub_node20_frontend_runtime() -> None:
    text = (REPO_ROOT / "scripts" / "pre_deploy_check.sh").read_text(encoding="utf-8")

    assert '[[ "${NODE_VERSION%%.*}" -ge 20 ]]' in text
    assert 'print_fail "Node.js version $NODE_VERSION (required: 20+)"' in text
    assert "recommended: 20+" not in text


def test_deploy_frontend_workflow_uses_repo_root_docker_context() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "deploy-frontend.yml").read_text(encoding="utf-8")

    assert "context: ." in text
    assert "file: aragora/live/Dockerfile" in text
    assert "context: aragora/live" not in text


def test_live_supabase_node_engines_fit_docker_runtime() -> None:
    package_json = _load_json(LIVE_DIR / "package.json")
    package_lock = _load_json(LIVE_DIR / "package-lock.json")

    runtime_floors = {
        path: min(_node_image_versions(path))
        for path in NODE_RUNTIME_FILES
        if _node_image_versions(path)
    }
    assert runtime_floors.keys() == set(NODE_RUNTIME_FILES), (
        "live Node runtime files must use numeric node image tags so engine floors can be checked"
    )
    runtime_floor = min(runtime_floors.values())

    package_floor = _node_engine_floor(package_json["engines"]["node"])
    assert runtime_floor >= package_floor

    packages = package_lock["packages"]
    for package_name in SUPABASE_PACKAGES:
        engine = packages[package_name]["engines"]["node"]
        assert runtime_floor >= _node_engine_floor(engine), (
            f"{package_name} requires Node {engine}, but live runtime floor is {runtime_floor}"
        )
