"""
CLI badge command - generate badge markdown for README.

Extracted from main.py for modularity.
Generates shields.io badge markdown for Aragora verification.
"""

from __future__ import annotations

import argparse

# Badge URL configurations
BADGE_URLS = {
    "reviewed": {
        "flat": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=flat",
        "flat-square": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=flat-square",
        "for-the-badge": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=for-the-badge",
        "plastic": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=plastic",
    },
    "consensus": {
        "flat": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=flat",
        "flat-square": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=flat-square",
        "for-the-badge": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=for-the-badge",
        "plastic": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=plastic",
    },
    "gauntlet": {
        "flat": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=flat",
        "flat-square": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=flat-square",
        "for-the-badge": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=for-the-badge",
        "plastic": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=plastic",
    },
}

BADGE_STYLES = ["flat", "flat-square", "for-the-badge", "plastic"]
BADGE_TYPES = ["reviewed", "consensus", "gauntlet"]


def get_badge_url(badge_type: str, style: str) -> str:
    """Get badge URL for given type and style."""
    return BADGE_URLS.get(badge_type, BADGE_URLS["reviewed"]).get(
        style, BADGE_URLS[badge_type]["flat"]
    )


def generate_badge_markdown(
    badge_type: str = "reviewed", style: str = "flat", repo: str | None = None
) -> tuple[str, str]:
    """Generate badge markdown and HTML.

    Returns:
        Tuple of (markdown, html)
    """
    badge_url = get_badge_url(badge_type, style)
    link_url = f"https://github.com/{repo}" if repo else "https://github.com/an0mium/aragora"

    markdown = f"[![Aragora]({badge_url})]({link_url})"
    html = f'<a href="{link_url}"><img src="{badge_url}" alt="Aragora"></a>'

    return markdown, html


def main(args: argparse.Namespace) -> None:
    """Handle 'badge' command - generate badge markdown for README."""
    style = getattr(args, "style", "flat")
    repo = getattr(args, "repo", None)
    badge_type = getattr(args, "type", "reviewed")

    markdown, html = generate_badge_markdown(badge_type, style, repo)

    print("\nAragora Badge\n")
    print("Add this to your README.md:\n")
    print("```markdown")
    print(markdown)
    print("```\n")

    print("Or use HTML:\n")
    print("```html")
    print(html)
    print("```\n")

    print("Preview:")
    print(f"  {markdown}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Aragora badges")
    parser.add_argument("--type", "-t", choices=BADGE_TYPES, default="reviewed", help="Badge type")
    parser.add_argument("--style", "-s", choices=BADGE_STYLES, default="flat", help="Badge style")
    parser.add_argument("--repo", "-r", help="GitHub repo (owner/repo)")
    main(parser.parse_args())
