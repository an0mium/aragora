import os
import re


def update_adapter_count():
    """
    Counts the number of adapters in the knowledge mound and updates the README.md.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    adapters_dir = os.path.join(project_root, "aragora", "knowledge", "mound", "adapters")
    readme_path = os.path.join(project_root, "README.md")

    try:
        # Count the number of files in the adapters directory
        adapter_files = os.listdir(adapters_dir)
        # Filter out any hidden files like .DS_Store
        adapter_count = len([f for f in adapter_files if not f.startswith(".")])
    except FileNotFoundError:
        print(f"Error: Adapters directory not found at {adapters_dir}")
        return

    try:
        with open(readme_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: README.md not found at {readme_path}")
        return

    # Use a regex to find and replace the adapter count within the HTML comments
    pattern = r"(<!-- adpt-count -->)\d+(<!-- /adpt-count -->)"
    new_text = rf"\g<1>{adapter_count}\g<2>"

    if re.search(pattern, content):
        updated_content = re.sub(pattern, new_text, content)
    else:
        print("Error: Could not find the adapter count placeholder in README.md")
        print("Please add '<!-- adpt-count -->...<!-- /adpt-count -->' to the README.")
        return

    if content != updated_content:
        try:
            with open(readme_path, "w") as f:
                f.write(updated_content)
            print(f"Successfully updated README.md with adapter count: {adapter_count}")
        except IOError as e:
            print(f"Error writing to README.md: {e}")
    else:
        print(
            f"Adapter count in README.md is already up to date ({adapter_count}). No changes made."
        )


if __name__ == "__main__":
    update_adapter_count()
