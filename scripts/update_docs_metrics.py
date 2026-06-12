import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_adapter_count() -> int:
    """Return the product's canonical Knowledge Mound adapter count."""
    from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

    return len(ADAPTER_SPECS)


def _replace_adapter_count(content: str, adapter_count: int) -> str | None:
    pattern = r"(<!-- adpt-count -->)\d+(<!-- /adpt-count -->)"
    new_text = rf"\g<1>{adapter_count}\g<2>"
    if not re.search(pattern, content):
        return None
    return re.sub(pattern, new_text, content)


def update_adapter_count(
    *,
    project_root: Path = PROJECT_ROOT,
    adapter_count: int | None = None,
) -> None:
    """
    Counts canonical Knowledge Mound adapters and updates the README.md.
    """
    readme_path = project_root / "README.md"
    adapter_count = get_adapter_count() if adapter_count is None else adapter_count

    try:
        content = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: README.md not found at {readme_path}")
        return

    updated_content = _replace_adapter_count(content, adapter_count)
    if updated_content is None:
        print("Error: Could not find the adapter count placeholder in README.md")
        print("Please add '<!-- adpt-count -->...<!-- /adpt-count -->' to the README.")
        return

    if content != updated_content:
        try:
            readme_path.write_text(updated_content, encoding="utf-8")
            print(f"Successfully updated README.md with adapter count: {adapter_count}")
        except IOError as e:
            print(f"Error writing to README.md: {e}")
    else:
        print(
            f"Adapter count in README.md is already up to date ({adapter_count}). No changes made."
        )


if __name__ == "__main__":
    update_adapter_count()
