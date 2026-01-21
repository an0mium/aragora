#!/usr/bin/env python3
"""
Seed a complete demo environment for Aragora.

This script creates a fully-populated demo environment including:
- Demo organization and users
- Sample debates with realistic content
- Agent rankings and history
- Consensus memories
- Workflow templates
- Sample gauntlet runs

Run: python scripts/seed_demo.py

Options:
    --clean     Remove existing demo data before seeding
    --minimal   Create minimal demo (faster, less data)
    --org-name  Custom organization name (default: "Demo Corp")
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Demo content templates
DEMO_DEBATES = [
    {
        "task": "Should our company adopt a 4-day work week?",
        "domain": "hr_policy",
        "agents": ["claude-opus", "gpt-4", "gemini-pro"],
        "consensus": True,
        "verdict": "Recommend pilot program with select teams before full rollout",
    },
    {
        "task": "What's the best approach for migrating our monolith to microservices?",
        "domain": "architecture",
        "agents": ["claude-sonnet", "codex", "deepseek-coder"],
        "consensus": True,
        "verdict": "Strangler fig pattern with domain-driven service boundaries",
    },
    {
        "task": "Should we build or buy a customer analytics platform?",
        "domain": "strategy",
        "agents": ["claude-opus", "gpt-4", "mistral-large"],
        "consensus": False,
        "verdict": None,
    },
    {
        "task": "How should we handle the security vulnerability in our auth system?",
        "domain": "security",
        "agents": ["claude-opus", "gpt-4", "gemini-pro"],
        "consensus": True,
        "verdict": "Immediate patch with rolling deployment, followed by auth system refactor",
    },
    {
        "task": "What pricing model should we use for our enterprise tier?",
        "domain": "pricing",
        "agents": ["claude-sonnet", "gpt-4", "mistral-large"],
        "consensus": True,
        "verdict": "Seat-based pricing with usage-based add-ons for compute-heavy features",
    },
    {
        "task": "Should we expand into the European market this quarter?",
        "domain": "expansion",
        "agents": ["claude-opus", "gpt-4", "gemini-pro"],
        "consensus": False,
        "verdict": None,
    },
    {
        "task": "What's the optimal cache invalidation strategy for our API?",
        "domain": "engineering",
        "agents": ["claude-sonnet", "codex", "deepseek-coder"],
        "consensus": True,
        "verdict": "Event-driven invalidation with TTL fallback and cache-aside pattern",
    },
    {
        "task": "How should we structure our data team: centralized or embedded?",
        "domain": "org_design",
        "agents": ["claude-opus", "gpt-4", "mistral-large"],
        "consensus": True,
        "verdict": "Hub-and-spoke model with central platform team and embedded analysts",
    },
]

DEMO_USERS = [
    {"name": "Alice Chen", "email": "alice@demo.aragora.ai", "role": "owner"},
    {"name": "Bob Smith", "email": "bob@demo.aragora.ai", "role": "admin"},
    {"name": "Carol Johnson", "email": "carol@demo.aragora.ai", "role": "member"},
    {"name": "David Lee", "email": "david@demo.aragora.ai", "role": "member"},
]

WORKFLOW_TEMPLATES = [
    {
        "name": "Security Review",
        "description": "Multi-agent security assessment for code changes",
        "category": "security",
        "steps": ["threat_modeling", "code_review", "penetration_test", "report"],
    },
    {
        "name": "Architecture Decision",
        "description": "Structured debate for technical architecture choices",
        "category": "engineering",
        "steps": ["requirements", "options", "debate", "decision", "document"],
    },
    {
        "name": "Policy Review",
        "description": "HR and compliance policy evaluation",
        "category": "compliance",
        "steps": ["draft_review", "legal_check", "stakeholder_input", "approval"],
    },
]


def generate_debate_id() -> str:
    """Generate a realistic debate ID."""
    return f"debate_{uuid4().hex[:12]}"


def generate_message_content(agent: str, round_num: int, task: str) -> str:
    """Generate realistic debate message content."""
    perspectives = {
        "claude-opus": [
            "From a comprehensive analysis perspective",
            "Considering the long-term implications",
            "Taking a nuanced view",
        ],
        "claude-sonnet": [
            "Looking at this systematically",
            "From an implementation standpoint",
            "Analyzing the practical aspects",
        ],
        "gpt-4": [
            "Examining multiple factors",
            "Based on industry best practices",
            "Considering various stakeholders",
        ],
        "gemini-pro": [
            "Taking a data-driven approach",
            "Looking at comparable scenarios",
            "From a research perspective",
        ],
        "mistral-large": [
            "Evaluating the trade-offs",
            "Considering efficiency",
            "From an optimization standpoint",
        ],
        "codex": [
            "From a technical implementation view",
            "Considering code maintainability",
            "Looking at developer experience",
        ],
        "deepseek-coder": [
            "Analyzing the technical constraints",
            "From an engineering perspective",
            "Considering system design",
        ],
    }

    intro = random.choice(perspectives.get(agent, ["Considering the options"]))

    if round_num == 1:
        return f"{intro}, I believe we should carefully evaluate this decision. {task} requires balancing multiple factors including cost, timeline, and risk."
    elif round_num == 2:
        return f"Building on the previous points, {intro.lower()}. I'd like to add that we should also consider the organizational impact and change management requirements."
    else:
        return f"After considering all perspectives, {intro.lower()}. My recommendation takes into account both short-term feasibility and long-term sustainability."


async def seed_organization(org_name: str, clean: bool = False) -> dict[str, Any]:
    """Seed demo organization and users."""
    logger.info(f"Creating demo organization: {org_name}")

    try:
        from aragora.storage.user_store import UserStore

        store = UserStore()

        # Generate org ID
        org_id = f"org_{uuid4().hex[:12]}"
        org_slug = org_name.lower().replace(" ", "-")

        # Check if demo org already exists
        existing = store.get_organization_by_slug(org_slug)
        if existing:
            if clean:
                logger.info(f"Removing existing organization: {org_slug}")
                # Would delete here if we had that method
            else:
                logger.info(f"Organization {org_slug} already exists, using existing")
                return {"org_id": existing.id, "users": []}

        # Create organization
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            id=org_id,
            name=org_name,
            slug=org_slug,
            tier=SubscriptionTier.PROFESSIONAL,
            owner_id="",  # Will be set after creating owner
        )

        created_org = store.create_organization(org)
        logger.info(f"  Created organization: {created_org.name} ({created_org.id})")

        # Create users
        users = []
        for user_data in DEMO_USERS:
            user_id = f"user_{uuid4().hex[:12]}"

            # Hash a demo password
            password = "demo123"  # Demo password
            salt = secrets.token_hex(16)
            password_hash = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            ).hex()

            from aragora.billing.models import User

            user = User(
                id=user_id,
                email=user_data["email"],
                name=user_data["name"],
                password_hash=password_hash,
                password_salt=salt,
                org_id=org_id,
                role=user_data["role"],
                is_active=True,
                email_verified=True,
            )

            created_user = store.create_user(user)
            users.append(created_user)
            logger.info(f"  Created user: {created_user.name} ({created_user.role})")

            # Set owner
            if user_data["role"] == "owner":
                store.update_organization(org_id, {"owner_id": user_id})

        return {"org_id": org_id, "users": users}

    except ImportError as e:
        logger.warning(f"UserStore not available: {e}")
        return {"org_id": f"org_{uuid4().hex[:12]}", "users": []}


async def seed_debates(org_id: str, minimal: bool = False) -> list[dict]:
    """Seed sample debates."""
    logger.info("Creating sample debates...")

    debates_to_create = DEMO_DEBATES[:3] if minimal else DEMO_DEBATES
    created_debates = []

    try:
        from aragora.persistence.db_config import get_nomic_dir

        nomic_dir = get_nomic_dir()
    except ImportError:
        nomic_dir = Path(".nomic")
        nomic_dir.mkdir(exist_ok=True)

    # Create debates directory
    debates_dir = nomic_dir / "debates"
    debates_dir.mkdir(exist_ok=True)

    for debate_data in debates_to_create:
        debate_id = generate_debate_id()
        created_at = datetime.now(timezone.utc) - timedelta(
            days=random.randint(1, 30),
            hours=random.randint(0, 23),
        )

        # Generate messages
        messages = []
        for round_num in range(1, 4):
            for agent in debate_data["agents"]:
                messages.append(
                    {
                        "agent": agent,
                        "round": round_num,
                        "content": generate_message_content(agent, round_num, debate_data["task"]),
                        "timestamp": (
                            created_at + timedelta(minutes=round_num * 5 + random.randint(0, 4))
                        ).isoformat(),
                    }
                )

        debate = {
            "id": debate_id,
            "task": debate_data["task"],
            "domain": debate_data["domain"],
            "agents": debate_data["agents"],
            "rounds": 3,
            "status": "completed",
            "consensus_reached": debate_data["consensus"],
            "verdict": debate_data["verdict"],
            "messages": messages,
            "org_id": org_id,
            "created_at": created_at.isoformat(),
            "completed_at": (created_at + timedelta(minutes=20)).isoformat(),
            "metadata": {
                "demo": True,
                "seeded_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Save to file
        debate_file = debates_dir / f"{debate_id}.json"
        with open(debate_file, "w") as f:
            json.dump(debate, f, indent=2)

        created_debates.append(debate)
        logger.info(f"  Created debate: {debate_data['task'][:50]}...")

    return created_debates


async def seed_agents() -> None:
    """Seed agent rankings using existing script."""
    logger.info("Seeding agent rankings...")

    try:
        # Import and run the existing seed script
        from scripts.seed_agents import main as seed_agents_main

        seed_agents_main()
    except ImportError:
        logger.warning("seed_agents script not available, skipping")
    except Exception as e:
        logger.warning(f"Error seeding agents: {e}")


async def seed_consensus_memory(debates: list[dict]) -> None:
    """Seed consensus memory from debates."""
    logger.info("Seeding consensus memory...")

    try:
        from aragora.memory.consensus import ConsensusMemory

        memory = ConsensusMemory()

        for debate in debates:
            if debate.get("consensus_reached") and debate.get("verdict"):
                memory.record(
                    question=debate["task"],
                    answer=debate["verdict"],
                    confidence=random.uniform(0.75, 0.95),
                    evidence=[f"Debate {debate['id']} consensus"],
                    domain=debate.get("domain"),
                )
                logger.info(f"  Recorded consensus: {debate['task'][:40]}...")

    except ImportError as e:
        logger.warning(f"ConsensusMemory not available: {e}")


async def seed_workflow_templates(org_id: str) -> None:
    """Seed workflow templates."""
    logger.info("Seeding workflow templates...")

    try:
        from aragora.storage.marketplace_store import get_marketplace_store

        store = get_marketplace_store()

        for template_data in WORKFLOW_TEMPLATES:
            template_id = f"tmpl_{uuid4().hex[:12]}"

            template = {
                "id": template_id,
                "name": template_data["name"],
                "description": template_data["description"],
                "category": template_data["category"],
                "author_id": org_id,
                "config": {"steps": template_data["steps"]},
                "is_public": True,
                "download_count": random.randint(10, 100),
                "rating_sum": random.randint(35, 50),
                "rating_count": random.randint(10, 15),
            }

            store.create_template(template)
            logger.info(f"  Created template: {template_data['name']}")

    except ImportError as e:
        logger.warning(f"MarketplaceStore not available: {e}")
    except Exception as e:
        logger.warning(f"Error seeding templates: {e}")


async def create_demo_api_key(org_id: str, user_id: str) -> str | None:
    """Create a demo API key for testing."""
    logger.info("Creating demo API key...")

    try:
        from aragora.storage.user_store import UserStore

        store = UserStore()

        # Generate API key
        api_key = f"ak_demo_{secrets.token_hex(16)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        store.set_api_key(user_id, api_key_hash, prefix=api_key[:10])

        logger.info(f"  Demo API key created: {api_key[:15]}...")
        return api_key

    except ImportError as e:
        logger.warning(f"Could not create API key: {e}")
        return None


def print_summary(org_data: dict, debates: list, api_key: str | None) -> None:
    """Print summary of seeded data."""
    print("\n" + "=" * 60)
    print("DEMO ENVIRONMENT READY")
    print("=" * 60)
    print(f"\nOrganization ID: {org_data['org_id']}")
    print(f"Users created: {len(org_data.get('users', []))}")
    print(f"Debates created: {len(debates)}")

    if api_key:
        print(f"\nDemo API Key: {api_key}")
        print("(Use this for API testing)")

    print("\nDemo credentials:")
    print("  Email: alice@demo.aragora.ai")
    print("  Password: demo123")

    print("\nAccess the demo at:")
    print("  http://localhost:8080")
    print("  http://localhost:8080/api/docs (API documentation)")

    print("\n" + "=" * 60)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed Aragora demo environment")
    parser.add_argument(
        "--clean", action="store_true", help="Remove existing demo data before seeding"
    )
    parser.add_argument("--minimal", action="store_true", help="Create minimal demo (faster)")
    parser.add_argument(
        "--org-name", default="Demo Corp", help="Organization name (default: Demo Corp)"
    )
    args = parser.parse_args()

    logger.info("Starting demo environment seeding...")
    logger.info(f"Options: clean={args.clean}, minimal={args.minimal}")

    # Seed in order
    org_data = await seed_organization(args.org_name, clean=args.clean)

    await seed_agents()

    debates = await seed_debates(org_data["org_id"], minimal=args.minimal)

    await seed_consensus_memory(debates)

    await seed_workflow_templates(org_data["org_id"])

    # Create API key for first user
    api_key = None
    if org_data.get("users"):
        owner = next((u for u in org_data["users"] if u.role == "owner"), None)
        if owner:
            api_key = await create_demo_api_key(org_data["org_id"], owner.id)

    print_summary(org_data, debates, api_key)

    logger.info("Demo environment seeding complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
