"""
Tests for the Essay Synthesis Pipeline.

Verifies end-to-end workflow:
1. Conversation loading
2. Claim extraction
3. Topic clustering
4. Outline generation
5. Export for synthesis
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aragora.pipelines.essay_synthesis import (
    EssaySynthesisPipeline,
    SynthesisConfig,
    TopicCluster,
    AttributedClaim,
    EssayOutline,
    create_essay_pipeline,
)

# Test fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "conversations"
CHATGPT_EXPORT = FIXTURES_DIR / "sample_chatgpt_export.json"
CLAUDE_EXPORT = FIXTURES_DIR / "sample_claude_export.json"


def test_pipeline_creation():
    """Test pipeline creation with default config."""
    pipeline = create_essay_pipeline()
    assert pipeline is not None
    assert pipeline.config.target_word_count == 50000
    print("✓ Pipeline creation passed")


def test_pipeline_custom_config():
    """Test pipeline creation with custom config."""
    config = SynthesisConfig(
        target_word_count=100000,
        min_claim_length=30,
        max_clusters=10,
    )
    pipeline = EssaySynthesisPipeline(config=config)
    assert pipeline.config.target_word_count == 100000
    assert pipeline.config.min_claim_length == 30
    print("✓ Custom config passed")


def test_conversation_loading():
    """Test loading conversations through pipeline."""
    pipeline = create_essay_pipeline()

    # Load both exports
    exports = pipeline.load_conversations(FIXTURES_DIR)
    assert len(exports) == 2, f"Expected 2 exports, got {len(exports)}"

    stats = pipeline.get_statistics()
    assert stats["loaded"] == True
    assert stats["total_conversations"] == 6

    print(f"✓ Conversation loading passed ({stats['total_conversations']} conversations)")


def test_claim_extraction():
    """Test claim extraction from loaded conversations."""
    pipeline = create_essay_pipeline()
    pipeline.load_conversations(FIXTURES_DIR)

    claims = pipeline.extract_all_claims()
    assert len(claims) > 0, "Should extract claims"

    # Check claim types
    claim_types = set(c.claim_type for c in claims)
    print(f"  Claim types found: {claim_types}")

    # Check deduplication worked
    claim_texts = [c.claim for c in claims]
    unique_texts = set(claim_texts)
    assert len(claim_texts) == len(unique_texts), "Should deduplicate claims"

    stats = pipeline.get_statistics()
    assert stats["claims_extracted"] == len(claims)

    print(f"✓ Claim extraction passed ({len(claims)} unique claims)")


def test_topic_clustering():
    """Test clustering claims by topic."""
    pipeline = create_essay_pipeline()
    pipeline.load_conversations(FIXTURES_DIR)
    pipeline.extract_all_claims()

    clusters = pipeline.cluster_claims()
    assert len(clusters) > 0, "Should create clusters"

    # Check cluster properties
    for cluster in clusters:
        assert isinstance(cluster, TopicCluster)
        assert cluster.id.startswith("cluster_")
        assert len(cluster.claims) >= 1
        assert len(cluster.keywords) > 0
        if cluster.claims:
            assert cluster.representative_claim is not None

    # Check cluster relationships
    has_relationships = any(len(c.related_clusters) > 0 for c in clusters)

    print(f"  Clusters created:")
    for cluster in clusters[:5]:
        print(f"    - {cluster.name}: {cluster.claim_count} claims, coherence={cluster.coherence_score:.2f}")

    stats = pipeline.get_statistics()
    assert stats["clusters_created"] == len(clusters)

    print(f"✓ Topic clustering passed ({len(clusters)} clusters)")


def test_outline_generation():
    """Test generating essay outline from clusters."""
    pipeline = create_essay_pipeline()
    pipeline.load_conversations(FIXTURES_DIR)
    pipeline.extract_all_claims()
    pipeline.cluster_claims()

    async def run_outline():
        outline = await pipeline.generate_outline(
            title="On Systems, Narratives, and Intelligence",
            thesis="This essay explores the interconnection between systems thinking, narrative interpretation, and artificial intelligence.",
        )
        return outline

    outline = asyncio.run(run_outline())
    assert isinstance(outline, EssayOutline)
    assert outline.title == "On Systems, Narratives, and Intelligence"
    assert len(outline.sections) > 0
    assert outline.target_word_count == 50000

    print(f"  Outline structure:")
    print(f"    Title: {outline.title}")
    print(f"    Thesis: {outline.thesis[:80]}...")
    print(f"    Sections: {outline.section_count}")
    print(f"    Target words: {outline.target_word_count:,}")

    for section in outline.sections[:3]:
        print(f"      - {section.title} ({len(section.subsections)} subsections)")

    print(f"✓ Outline generation passed ({outline.section_count} sections)")


def test_export_for_synthesis():
    """Test exporting data for synthesis."""
    pipeline = create_essay_pipeline()
    pipeline.load_conversations(FIXTURES_DIR)
    pipeline.extract_all_claims()
    pipeline.cluster_claims()

    async def run_export():
        outline = await pipeline.generate_outline(
            title="Test Essay",
        )
        return pipeline.export_for_synthesis(outline)

    export_data = asyncio.run(run_export())

    assert "outline" in export_data
    assert "claims" in export_data
    assert "clusters" in export_data
    assert "statistics" in export_data
    assert "config" in export_data

    # Verify data integrity
    assert len(export_data["claims"]) > 0
    assert len(export_data["clusters"]) > 0
    assert export_data["statistics"]["claims_extracted"] > 0

    # Test JSON serialization
    json_str = json.dumps(export_data, indent=2, default=str)
    assert len(json_str) > 1000

    print(f"  Export data:")
    print(f"    Claims: {len(export_data['claims'])}")
    print(f"    Clusters: {len(export_data['clusters'])}")
    print(f"    Export size: {len(json_str):,} bytes")

    print(f"✓ Export for synthesis passed")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    config = SynthesisConfig(
        target_word_count=50000,
        min_cluster_size=2,  # Lower for test data
        include_methodology=True,
        include_counterarguments=True,
    )
    pipeline = EssaySynthesisPipeline(config=config)

    # 1. Load conversations
    exports = pipeline.load_conversations(FIXTURES_DIR)

    # 2. Extract claims
    claims = pipeline.extract_all_claims()

    # 3. Cluster claims
    clusters = pipeline.cluster_claims()

    # 4. Generate outline
    async def run_synthesis():
        outline = await pipeline.generate_outline(
            title="Systems, Intelligence, and Equilibrium: A Synthesis",
        )
        export_data = pipeline.export_for_synthesis(outline)
        return outline, export_data

    outline, export_data = asyncio.run(run_synthesis())

    # Verify complete workflow
    assert len(exports) > 0
    assert len(claims) > 0
    assert len(clusters) > 0
    assert outline.section_count > 0

    # Print summary
    print(f"\n  End-to-end workflow summary:")
    print(f"    Conversations loaded: {len(list(pipeline.ingestor.iter_conversations()))}")
    print(f"    Claims extracted: {len(claims)}")
    print(f"    Clusters created: {len(clusters)}")
    print(f"    Essay sections: {outline.section_count}")
    print(f"    Bibliography entries: {len(outline.bibliography)}")

    print(f"✓ End-to-end workflow passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ESSAY SYNTHESIS PIPELINE TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Pipeline Creation", test_pipeline_creation),
        ("Custom Config", test_pipeline_custom_config),
        ("Conversation Loading", test_conversation_loading),
        ("Claim Extraction", test_claim_extraction),
        ("Topic Clustering", test_topic_clustering),
        ("Outline Generation", test_outline_generation),
        ("Export for Synthesis", test_export_for_synthesis),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
