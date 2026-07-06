import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = REPO_ROOT / "aragora" / "live"
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


def _node_engine_floor(engine: str) -> int:
    match = re.search(r">=\s*(\d+)", engine)
    assert match, f"unsupported node engine expression: {engine!r}"
    return int(match.group(1))


def _docker_node_majors(dockerfile: Path) -> list[int]:
    text = dockerfile.read_text(encoding="utf-8")
    return [
        int(match.group(1))
        for match in re.finditer(r"^FROM\s+node:(\d+)(?:[.\-]|$)", text, re.MULTILINE)
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


def test_live_supabase_node_engines_fit_docker_runtime() -> None:
    package_json = _load_json(LIVE_DIR / "package.json")
    package_lock = _load_json(LIVE_DIR / "package-lock.json")
    docker_majors = _docker_node_majors(LIVE_DIR / "Dockerfile")

    assert docker_majors, "aragora/live/Dockerfile must declare Node base images"
    docker_floor = min(docker_majors)

    package_floor = _node_engine_floor(package_json["engines"]["node"])
    assert docker_floor >= package_floor

    packages = package_lock["packages"]
    for package_name in SUPABASE_PACKAGES:
        engine = packages[package_name]["engines"]["node"]
        assert docker_floor >= _node_engine_floor(engine), (
            f"{package_name} requires Node {engine}, but Dockerfile uses Node {docker_floor}"
        )
