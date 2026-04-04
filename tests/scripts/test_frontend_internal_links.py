from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "aragora" / "live" / "src" / "app"
SCAN_ROOTS = [
    APP_ROOT,
    REPO_ROOT / "aragora" / "live" / "src" / "components",
]

LITERAL_HREF_RE = re.compile(r"""href=(?:"([^"]+)"|'([^']+)')""")
TEMPLATE_HREF_RE = re.compile(r"""href=\{`([^`]+)`\}""")


def _strip_route_groups(parts: list[str]) -> list[str]:
    return [part for part in parts if not (part.startswith("(") and part.endswith(")"))]


def _build_route_inventory() -> tuple[set[str], set[str]]:
    literal_routes: set[str] = set()
    dynamic_prefixes: set[str] = set()

    for page in APP_ROOT.rglob("page.tsx"):
        route_parts = _strip_route_groups(list(page.relative_to(APP_ROOT).parts[:-1]))
        url_parts: list[str] = []
        saw_non_optional_dynamic = False

        for part in route_parts:
            if part.startswith("[") and part.endswith("]"):
                dynamic_prefixes.add("/" + "/".join(url_parts + [""]))
                if part.startswith("[[..."):
                    continue
                saw_non_optional_dynamic = True
                continue
            url_parts.append(part)

        route = "/" + "/".join(url_parts)
        normalized = route.rstrip("/") or "/"
        if not saw_non_optional_dynamic:
            literal_routes.add(normalized)

    return literal_routes, dynamic_prefixes


def _normalize_path(value: str) -> str:
    path = value.split("#", 1)[0].split("?", 1)[0]
    return path.rstrip("/") or "/"


def _iter_href_issues() -> list[str]:
    literal_routes, dynamic_prefixes = _build_route_inventory()
    issues: list[str] = []

    for root in SCAN_ROOTS:
        for tsx_file in root.rglob("*.tsx"):
            text = tsx_file.read_text(encoding="utf-8", errors="ignore")

            for match in LITERAL_HREF_RE.finditer(text):
                href = match.group(1) or match.group(2) or ""
                if not href.startswith("/") or href.startswith("/api/"):
                    continue
                normalized = _normalize_path(href)
                if normalized not in literal_routes:
                    issues.append(
                        f"{tsx_file.relative_to(REPO_ROOT)} -> {href} does not map to an app route"
                    )

            for match in TEMPLATE_HREF_RE.finditer(text):
                href_template = match.group(1)
                if not href_template.startswith("/") or href_template.startswith("/api/"):
                    continue

                path_portion = href_template.split("#", 1)[0].split("?", 1)[0]
                if "${" not in path_portion:
                    normalized = _normalize_path(path_portion)
                    if normalized not in literal_routes:
                        issues.append(
                            f"{tsx_file.relative_to(REPO_ROOT)} -> {href_template} does not map to an app route"
                        )
                    continue

                prefix = path_portion.split("${", 1)[0]
                normalized_prefix = prefix if prefix.endswith("/") else _normalize_path(prefix)

                if prefix.endswith("/"):
                    if normalized_prefix not in dynamic_prefixes:
                        issues.append(
                            f"{tsx_file.relative_to(REPO_ROOT)} -> {href_template} targets a missing dynamic route prefix"
                        )
                elif normalized_prefix not in literal_routes:
                    issues.append(
                        f"{tsx_file.relative_to(REPO_ROOT)} -> {href_template} targets a missing app route"
                    )

    return sorted(set(issues))


def test_internal_frontend_links_resolve_to_known_routes() -> None:
    issues = _iter_href_issues()
    assert not issues, "Broken internal frontend links found:\n" + "\n".join(issues)
