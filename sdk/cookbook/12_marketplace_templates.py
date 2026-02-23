#!/usr/bin/env python3
"""
12_marketplace_templates.py - Browse, install, and create marketplace templates.

The marketplace lets you discover and share reusable configurations for
agents, debates, and workflows. This example shows the full lifecycle.

Usage:
    python 12_marketplace_templates.py                    # Browse templates
    python 12_marketplace_templates.py --dry-run          # Mock mode
    python 12_marketplace_templates.py --publish           # Create and publish
"""

import argparse
import asyncio
from aragora_sdk import AragoraClient


async def browse_marketplace(dry_run: bool = False) -> dict:
    """Browse and search marketplace templates."""

    client = AragoraClient()

    if dry_run:
        print("[DRY RUN] Would browse marketplace")
        return {"status": "dry_run"}

    # List featured templates
    featured = await client.marketplace.featured(limit=5)
    print("Featured Templates:")
    for t in featured:
        stars = t.get("stars", 0)
        print(f"  [{t['category']}] {t['name']} ({stars} stars)")
        print(f"    {t.get('description', '')[:80]}")

    # Search by category
    coding_templates = await client.marketplace.search(category="coding", limit=5)
    print(f"\nCoding Templates ({len(coding_templates)}):")
    for t in coding_templates:
        print(f"  - {t['name']} by {t.get('author', 'unknown')}")

    # Get template details
    if featured:
        detail = await client.marketplace.get(featured[0]["id"])
        print(f"\nTemplate Detail: {detail['name']}")
        print(f"  Version: {detail.get('version', '?')}")
        print(f"  Downloads: {detail.get('downloads', 0)}")

    return {"featured": len(featured), "coding": len(coding_templates)}


async def install_template(template_id: str) -> dict:
    """Install a marketplace template locally."""

    client = AragoraClient()

    # Download the template
    template = await client.marketplace.download(template_id)
    print(f"Downloaded: {template['name']}")

    # Install it (makes it available for use in debates/workflows)
    result = await client.marketplace.install(template_id)
    print(f"Installed: {result.get('status', 'ok')}")

    return result


async def publish_template(dry_run: bool = False) -> dict:
    """Create and publish a custom template."""

    client = AragoraClient()

    # Define a custom debate template
    template = {
        "type": "DebateTemplate",
        "name": "Risk Assessment Debate",
        "description": "Multi-agent risk evaluation for business decisions",
        "category": "decision",
        "tags": ["risk", "business", "decision-making"],
        "task_template": "Evaluate risks and opportunities of: {decision}",
        "agent_roles": [
            {"role": "risk_analyst", "focus": "downside_risks"},
            {"role": "opportunity_scout", "focus": "upside_potential"},
            {"role": "devil_advocate", "focus": "assumptions"},
            {"role": "synthesizer", "aggregates": True},
        ],
        "protocol": {
            "rounds": 3,
            "consensus_mode": "synthesis",
            "require_evidence": True,
        },
        "evaluation_criteria": ["thoroughness", "specificity", "balance"],
    }

    if dry_run:
        print(f"[DRY RUN] Would publish: {template['name']}")
        print(f"  Roles: {[r['role'] for r in template['agent_roles']]}")
        return {"status": "dry_run"}

    result = await client.marketplace.publish(template)
    print(f"Published: {result.get('id', '?')}")
    print(f"URL: {result.get('url', 'N/A')}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Marketplace template operations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Create and publish a template")
    parser.add_argument("--install", type=str, help="Install a template by ID")
    args = parser.parse_args()

    if args.publish:
        asyncio.run(publish_template(args.dry_run))
    elif args.install:
        asyncio.run(install_template(args.install))
    else:
        asyncio.run(browse_marketplace(args.dry_run))


if __name__ == "__main__":
    main()
