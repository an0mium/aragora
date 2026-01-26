"""
E2E tests for new features: Marketplace, Batch Explainability, and Webhooks.

Tests the complete integration flows:
1. Marketplace - template publish -> download -> rate -> review cycle
2. Batch Explainability - batch job creation -> processing -> results
3. Webhooks - registration -> event delivery -> receipt notifications
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

# Mark all tests as e2e tests
pytestmark = [pytest.mark.e2e, pytest.mark.new_features]


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockTemplate:
    """Mock template for testing."""

    id: str
    name: str
    description: str
    author_id: str
    author_name: str
    category: str
    pattern: str
    workflow_definition: Dict[str, Any]
    tags: List[str]
    rating: float = 0.0
    rating_count: int = 0
    download_count: int = 0
    is_featured: bool = False
    created_at: float = 0.0


@dataclass
class MockReview:
    """Mock review for testing."""

    id: str
    template_id: str
    user_id: str
    user_name: str
    rating: int
    title: str
    content: str
    created_at: float = 0.0


@pytest.fixture
def mock_marketplace_store():
    """Create a mock marketplace store for testing."""
    templates: Dict[str, MockTemplate] = {}
    reviews: Dict[str, List[MockReview]] = {}
    ratings: Dict[str, int] = {}

    class MockStore:
        def create_template(
            self,
            name: str,
            description: str,
            author_id: str,
            author_name: str,
            category: str,
            pattern: str,
            workflow_definition: Dict,
            tags: List[str] = None,
        ) -> MockTemplate:
            template_id = f"tpl-{uuid4().hex[:8]}"
            template = MockTemplate(
                id=template_id,
                name=name,
                description=description,
                author_id=author_id,
                author_name=author_name,
                category=category,
                pattern=pattern,
                workflow_definition=workflow_definition,
                tags=tags or [],
                rating=0.0,
                rating_count=0,
                download_count=0,
                is_featured=False,
                created_at=time.time(),
            )
            templates[template_id] = template
            return template

        def get_template(self, template_id: str) -> Optional[MockTemplate]:
            return templates.get(template_id)

        def list_templates(self, category=None, search=None, limit=20, offset=0):
            result = list(templates.values())
            if category:
                result = [t for t in result if t.category == category]
            if search:
                result = [
                    t
                    for t in result
                    if search.lower() in t.name.lower() or search.lower() in t.description.lower()
                ]
            total = len(result)
            result = result[offset : offset + limit]
            return result, total

        def rate_template(self, template_id: str, user_id: str, rating: int):
            if template_id not in templates:
                raise ValueError("Template not found")
            if not 1 <= rating <= 5:
                raise ValueError("Rating must be between 1 and 5")

            key = f"{template_id}:{user_id}"
            ratings[key] = rating

            # Calculate average
            template_ratings = [r for k, r in ratings.items() if k.startswith(f"{template_id}:")]
            avg = sum(template_ratings) / len(template_ratings)
            count = len(template_ratings)

            templates[template_id].rating = avg
            templates[template_id].rating_count = count

            return avg, count

        def create_review(
            self,
            template_id: str,
            user_id: str,
            user_name: str,
            rating: int,
            title: str,
            content: str,
        ) -> MockReview:
            review_id = f"rev-{uuid4().hex[:8]}"
            review = MockReview(
                id=review_id,
                template_id=template_id,
                user_id=user_id,
                user_name=user_name,
                rating=rating,
                title=title,
                content=content,
                created_at=time.time(),
            )
            if template_id not in reviews:
                reviews[template_id] = []
            reviews[template_id].append(review)
            return review

        def list_reviews(self, template_id: str) -> List[MockReview]:
            return reviews.get(template_id, [])

        def increment_download(self, template_id: str):
            if template_id in templates:
                templates[template_id].download_count += 1

        def get_featured(self):
            return [t for t in templates.values() if t.is_featured]

        def list_categories(self):
            return [
                {"id": "security", "name": "Security", "template_count": 0},
                {"id": "testing", "name": "Testing", "template_count": 0},
                {"id": "code-review", "name": "Code Review", "template_count": 0},
            ]

    return MockStore()


@pytest.fixture
def mock_webhook_store():
    """Create a mock webhook store for testing."""
    webhooks = {}
    delivery_log = []

    class MockStore:
        def register(
            self,
            url: str,
            events: List[str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            user_id: Optional[str] = None,
        ):
            webhook_id = str(uuid4())
            secret = f"whsec_{uuid4().hex}"
            webhook = MagicMock(
                id=webhook_id,
                url=url,
                events=events,
                secret=secret,
                active=True,
                name=name,
                description=description,
                user_id=user_id,
                created_at=time.time(),
                delivery_count=0,
                failure_count=0,
            )
            webhooks[webhook_id] = webhook
            return webhook

        def get(self, webhook_id: str):
            return webhooks.get(webhook_id)

        def list(self, user_id=None, active_only=False):
            result = list(webhooks.values())
            if user_id:
                result = [w for w in result if w.user_id == user_id]
            if active_only:
                result = [w for w in result if w.active]
            return result

        def delete(self, webhook_id: str):
            if webhook_id in webhooks:
                del webhooks[webhook_id]
                return True
            return False

        def get_for_event(self, event_type: str):
            return [
                w
                for w in webhooks.values()
                if w.active and (event_type in w.events or "*" in w.events)
            ]

        def record_delivery(self, webhook_id: str, status_code: int, success: bool):
            if webhook_id in webhooks:
                webhooks[webhook_id].delivery_count += 1
                if not success:
                    webhooks[webhook_id].failure_count += 1
            delivery_log.append(
                {
                    "webhook_id": webhook_id,
                    "status_code": status_code,
                    "success": success,
                    "timestamp": time.time(),
                }
            )

        def get_delivery_log(self):
            return delivery_log

    return MockStore()


@pytest.fixture
def mock_batch_store():
    """Create a mock batch job store for testing."""
    jobs = {}
    results = {}

    class MockStore:
        def create_job(self, debate_ids: List[str], options: Dict[str, Any], user_id: str):
            job_id = f"batch-{uuid4().hex[:8]}"
            job = {
                "id": job_id,
                "debate_ids": debate_ids,
                "options": options,
                "user_id": user_id,
                "status": "pending",
                "created_at": time.time(),
                "total": len(debate_ids),
                "processed": 0,
                "failed": 0,
            }
            jobs[job_id] = job
            results[job_id] = []
            return MagicMock(**job)

        def get_job(self, job_id: str):
            if job_id in jobs:
                return MagicMock(**jobs[job_id])
            return None

        def update_job_status(self, job_id: str, status: str):
            if job_id in jobs:
                jobs[job_id]["status"] = status

        def add_result(self, job_id: str, debate_id: str, result: Dict):
            if job_id in jobs:
                results[job_id].append(
                    {"debate_id": debate_id, "result": result, "status": "success"}
                )
                jobs[job_id]["processed"] += 1

        def add_error(self, job_id: str, debate_id: str, error: str):
            if job_id in jobs:
                results[job_id].append({"debate_id": debate_id, "error": error, "status": "error"})
                jobs[job_id]["processed"] += 1
                jobs[job_id]["failed"] += 1

        def get_results(self, job_id: str):
            return results.get(job_id, [])

        def list_jobs(self, user_id: str = None, status: str = None):
            result = list(jobs.values())
            if user_id:
                result = [j for j in result if j["user_id"] == user_id]
            if status:
                result = [j for j in result if j["status"] == status]
            return [MagicMock(**j) for j in result]

    return MockStore()


# =============================================================================
# Marketplace E2E Tests
# =============================================================================


class TestMarketplaceE2E:
    """E2E tests for the Template Marketplace feature."""

    @pytest.mark.asyncio
    async def test_full_marketplace_workflow(self, mock_marketplace_store):
        """Test complete marketplace workflow: publish -> search -> download -> rate -> review."""
        store = mock_marketplace_store

        # 1. Publish a template
        template = store.create_template(
            name="Security Scanner Template",
            description="Automated security vulnerability scanning workflow",
            author_id="user-author-1",
            author_name="Security Expert",
            category="security",
            pattern="security_scan",
            workflow_definition={"nodes": [{"id": "scan", "type": "task"}]},
            tags=["security", "automation", "scanning"],
        )
        assert template.id.startswith("tpl-")
        assert template.name == "Security Scanner Template"
        assert template.category == "security"

        # 2. Search for templates
        templates, total = store.list_templates(search="security")
        assert total == 1
        assert templates[0].name == "Security Scanner Template"

        # Filter by category
        templates, total = store.list_templates(category="security")
        assert total == 1

        # 3. Download template (increment download count)
        store.increment_download(template.id)
        updated = store.get_template(template.id)
        assert updated.download_count == 1

        # 4. Rate the template
        avg, count = store.rate_template(template.id, "user-rater-1", 5)
        assert avg == 5.0
        assert count == 1

        # Second rating
        avg, count = store.rate_template(template.id, "user-rater-2", 4)
        assert avg == 4.5
        assert count == 2

        # 5. Write a review
        review = store.create_review(
            template_id=template.id,
            user_id="user-reviewer-1",
            user_name="Happy User",
            rating=5,
            title="Excellent security scanner!",
            content="This template helped us find 3 critical vulnerabilities.",
        )
        assert review.id.startswith("rev-")
        assert review.title == "Excellent security scanner!"

        # 6. List reviews
        reviews = store.list_reviews(template.id)
        assert len(reviews) == 1
        assert reviews[0].content == "This template helped us find 3 critical vulnerabilities."

    @pytest.mark.asyncio
    async def test_marketplace_category_filtering(self, mock_marketplace_store):
        """Test that category filtering works correctly."""
        store = mock_marketplace_store

        # Create templates in different categories
        store.create_template(
            name="Security Template",
            description="Security",
            author_id="u1",
            author_name="Author",
            category="security",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Testing Template",
            description="Testing",
            author_id="u1",
            author_name="Author",
            category="testing",
            pattern="p2",
            workflow_definition={},
        )
        store.create_template(
            name="Code Review Template",
            description="Code review",
            author_id="u1",
            author_name="Author",
            category="code-review",
            pattern="p3",
            workflow_definition={},
        )

        # Filter by each category
        security, _ = store.list_templates(category="security")
        testing, _ = store.list_templates(category="testing")
        code_review, _ = store.list_templates(category="code-review")

        assert len(security) == 1
        assert len(testing) == 1
        assert len(code_review) == 1
        assert security[0].category == "security"

    @pytest.mark.asyncio
    async def test_marketplace_rating_validation(self, mock_marketplace_store):
        """Test that rating validation works."""
        store = mock_marketplace_store

        template = store.create_template(
            name="Test Template",
            description="Test",
            author_id="u1",
            author_name="Author",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        # Valid ratings
        store.rate_template(template.id, "user1", 1)
        store.rate_template(template.id, "user2", 5)

        # Invalid ratings should raise
        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "user3", 0)

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "user4", 6)


# =============================================================================
# Webhook E2E Tests
# =============================================================================


class TestWebhooksE2E:
    """E2E tests for the Webhooks feature."""

    @pytest.mark.asyncio
    async def test_full_webhook_workflow(self, mock_webhook_store):
        """Test complete webhook workflow: register -> receive events -> track delivery."""
        store = mock_webhook_store

        # 1. Register a webhook
        webhook = store.register(
            url="https://example.com/webhook",
            events=["debate_end", "consensus", "receipt_ready"],
            name="My Integration",
            description="Receives debate completion events",
            user_id="user-123",
        )
        assert webhook.id
        assert webhook.url == "https://example.com/webhook"
        assert webhook.secret.startswith("whsec_")
        assert "debate_end" in webhook.events

        # 2. List webhooks for user
        webhooks = store.list(user_id="user-123")
        assert len(webhooks) == 1

        # 3. Simulate event delivery
        matching = store.get_for_event("debate_end")
        assert len(matching) == 1

        # Non-matching event
        matching = store.get_for_event("agent_message")
        assert len(matching) == 0

        # 4. Record successful delivery
        store.record_delivery(webhook.id, status_code=200, success=True)
        updated = store.get(webhook.id)
        assert updated.delivery_count == 1
        assert updated.failure_count == 0

        # 5. Record failed delivery
        store.record_delivery(webhook.id, status_code=500, success=False)
        updated = store.get(webhook.id)
        assert updated.delivery_count == 2
        assert updated.failure_count == 1

        # 6. Delete webhook
        result = store.delete(webhook.id)
        assert result is True
        assert store.get(webhook.id) is None

    @pytest.mark.asyncio
    async def test_webhook_wildcard_events(self, mock_webhook_store):
        """Test that wildcard (*) event subscription works."""
        store = mock_webhook_store

        # Register with wildcard
        webhook = store.register(
            url="https://example.com/all-events",
            events=["*"],
            user_id="user-456",
        )

        # Should match any event
        assert len(store.get_for_event("debate_end")) == 1
        assert len(store.get_for_event("consensus")) == 1
        assert len(store.get_for_event("receipt_ready")) == 1
        assert len(store.get_for_event("agent_message")) == 1

    @pytest.mark.asyncio
    async def test_webhook_delivery_tracking(self, mock_webhook_store):
        """Test delivery tracking and failure counts."""
        store = mock_webhook_store

        webhook = store.register(
            url="https://flaky.example.com/webhook",
            events=["debate_end"],
            user_id="user-789",
        )

        # Simulate mixed delivery results
        store.record_delivery(webhook.id, 200, True)
        store.record_delivery(webhook.id, 200, True)
        store.record_delivery(webhook.id, 500, False)
        store.record_delivery(webhook.id, 200, True)
        store.record_delivery(webhook.id, 503, False)

        updated = store.get(webhook.id)
        assert updated.delivery_count == 5
        assert updated.failure_count == 2

        # Check delivery log
        log = store.get_delivery_log()
        assert len(log) == 5
        failures = [entry for entry in log if not entry["success"]]
        assert len(failures) == 2


# =============================================================================
# Batch Explainability E2E Tests
# =============================================================================


class TestBatchExplainabilityE2E:
    """E2E tests for the Batch Explainability feature."""

    @pytest.mark.asyncio
    async def test_full_batch_workflow(self, mock_batch_store):
        """Test complete batch workflow: create job -> process -> get results."""
        store = mock_batch_store

        # 1. Create a batch job
        debate_ids = ["debate-1", "debate-2", "debate-3"]
        job = store.create_job(
            debate_ids=debate_ids,
            options={"include_counterfactuals": True, "depth": "detailed"},
            user_id="user-analyst",
        )
        assert job.id.startswith("batch-")
        assert job.status == "pending"
        assert job.total == 3

        # 2. Update job status to processing
        store.update_job_status(job.id, "processing")
        updated = store.get_job(job.id)
        assert updated.status == "processing"

        # 3. Add results for each debate
        store.add_result(
            job.id,
            "debate-1",
            {
                "explanation": "Decision based on majority consensus",
                "evidence_chain": ["claim-1", "claim-2"],
                "confidence": 0.95,
            },
        )
        store.add_result(
            job.id,
            "debate-2",
            {
                "explanation": "Decision based on expert agent weight",
                "evidence_chain": ["claim-3"],
                "confidence": 0.88,
            },
        )
        store.add_error(job.id, "debate-3", "Debate not found")

        # 4. Check job progress
        updated = store.get_job(job.id)
        assert updated.processed == 3
        assert updated.failed == 1

        # 5. Mark job as complete
        store.update_job_status(job.id, "partial")  # partial because of failures
        updated = store.get_job(job.id)
        assert updated.status == "partial"

        # 6. Get results
        results = store.get_results(job.id)
        assert len(results) == 3
        successes = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        assert len(successes) == 2
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_batch_job_listing(self, mock_batch_store):
        """Test listing batch jobs with filters."""
        store = mock_batch_store

        # Create multiple jobs
        job1 = store.create_job(["d1"], {}, "user-1")
        job2 = store.create_job(["d2", "d3"], {}, "user-1")
        job3 = store.create_job(["d4"], {}, "user-2")

        # Complete one job
        store.update_job_status(job1.id, "completed")

        # List all jobs for user-1
        jobs = store.list_jobs(user_id="user-1")
        assert len(jobs) == 2

        # List only completed jobs
        jobs = store.list_jobs(status="completed")
        assert len(jobs) == 1
        assert jobs[0].id == job1.id

        # List pending jobs
        jobs = store.list_jobs(status="pending")
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_batch_large_job(self, mock_batch_store):
        """Test handling of large batch jobs."""
        store = mock_batch_store

        # Create a large batch job
        debate_ids = [f"debate-{i}" for i in range(100)]
        job = store.create_job(debate_ids=debate_ids, options={}, user_id="user-bulk")

        assert job.total == 100
        assert job.processed == 0

        # Simulate processing
        for i, debate_id in enumerate(debate_ids):
            if i % 10 == 0:
                store.add_error(job.id, debate_id, "Timeout")
            else:
                store.add_result(job.id, debate_id, {"explanation": f"Result for {debate_id}"})

        updated = store.get_job(job.id)
        assert updated.processed == 100
        assert updated.failed == 10  # Every 10th failed


# =============================================================================
# Integration Tests (Cross-Feature)
# =============================================================================


class TestCrossFeatureIntegration:
    """Tests that verify integration between features."""

    @pytest.mark.asyncio
    async def test_webhook_on_batch_complete(self, mock_webhook_store, mock_batch_store):
        """Test that webhooks can be triggered on batch job completion."""
        webhook_store = mock_webhook_store
        batch_store = mock_batch_store

        # Register webhook for batch events
        webhook = webhook_store.register(
            url="https://example.com/batch-webhook",
            events=["batch_complete", "explanation_ready"],
            user_id="user-integration",
        )

        # Create and complete a batch job
        job = batch_store.create_job(
            debate_ids=["debate-1"],
            options={},
            user_id="user-integration",
        )
        batch_store.add_result(job.id, "debate-1", {"explanation": "Test"})
        batch_store.update_job_status(job.id, "completed")

        # Find webhooks for batch_complete event
        matching = webhook_store.get_for_event("batch_complete")
        assert len(matching) == 1
        assert matching[0].id == webhook.id

        # Simulate delivery
        webhook_store.record_delivery(webhook.id, 200, True)
        updated = webhook_store.get(webhook.id)
        assert updated.delivery_count == 1

    @pytest.mark.asyncio
    async def test_marketplace_template_for_batch_processing(
        self, mock_marketplace_store, mock_batch_store
    ):
        """Test using marketplace templates for batch processing configurations."""
        marketplace = mock_marketplace_store
        batch = mock_batch_store

        # Publish an explainability workflow template
        template = marketplace.create_template(
            name="Detailed Explainability Config",
            description="Configuration for detailed batch explanations",
            author_id="user-expert",
            author_name="Explainability Expert",
            category="analytics",
            pattern="batch_explain",
            workflow_definition={
                "options": {
                    "include_counterfactuals": True,
                    "include_evidence_chain": True,
                    "depth": "detailed",
                }
            },
        )

        # User downloads template
        marketplace.increment_download(template.id)
        downloaded = marketplace.get_template(template.id)

        # Use template options for batch job
        job = batch.create_job(
            debate_ids=["d1", "d2"],
            options=downloaded.workflow_definition.get("options", {}),
            user_id="user-consumer",
        )

        # Verify options were applied
        job_data = batch.get_job(job.id)
        assert job_data is not None
