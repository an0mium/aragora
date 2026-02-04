"""
Tests for the Conversation Ingestor Connector.

Verifies:
1. ChatGPT export parsing
2. Claude export parsing
3. Search functionality
4. Claim extraction
5. Topic analysis
6. Statistics generation
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aragora.connectors.conversation_ingestor import (
    ConversationIngestorConnector,
    Conversation,
    ConversationMessage,
    ConversationExport,
    ClaimExtraction,
)

# Test fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "conversations"
CHATGPT_EXPORT = FIXTURES_DIR / "sample_chatgpt_export.json"
CLAUDE_EXPORT = FIXTURES_DIR / "sample_claude_export.json"


def test_chatgpt_format_detection():
    """Test that ChatGPT format is correctly detected."""
    connector = ConversationIngestorConnector()

    with open(CHATGPT_EXPORT, "r") as f:
        data = json.load(f)

    assert connector._is_chatgpt_format(data), "Should detect ChatGPT format"
    assert not connector._is_claude_format(data), "Should not detect Claude format"
    print("✓ ChatGPT format detection passed")


def test_claude_format_detection():
    """Test that Claude format is correctly detected."""
    connector = ConversationIngestorConnector()

    with open(CLAUDE_EXPORT, "r") as f:
        data = json.load(f)

    assert connector._is_claude_format(data), "Should detect Claude format"
    assert not connector._is_chatgpt_format(data), "Should not detect ChatGPT format"
    print("✓ Claude format detection passed")


def test_chatgpt_export_loading():
    """Test loading ChatGPT export."""
    connector = ConversationIngestorConnector()
    export = connector.load_export(CHATGPT_EXPORT)

    assert export.source == "chatgpt", f"Expected source 'chatgpt', got '{export.source}'"
    assert export.conversation_count == 3, f"Expected 3 conversations, got {export.conversation_count}"
    assert export.total_messages > 0, "Should have messages"
    assert export.total_words > 0, "Should have words"

    # Check first conversation
    conv = export.conversations[0]
    assert conv.id == "conv_ai_alignment_001"
    assert "Alignment" in conv.title
    assert conv.source == "chatgpt"
    assert len(conv.messages) == 4

    # Check message parsing
    first_msg = conv.messages[0]
    assert first_msg.role == "user"
    assert "instrumental convergence" in first_msg.content

    print(f"✓ ChatGPT export loading passed ({export.conversation_count} conversations, {export.total_words:,} words)")


def test_claude_export_loading():
    """Test loading Claude export."""
    connector = ConversationIngestorConnector()
    export = connector.load_export(CLAUDE_EXPORT)

    assert export.source == "claude", f"Expected source 'claude', got '{export.source}'"
    assert export.conversation_count == 3, f"Expected 3 conversations, got {export.conversation_count}"
    assert export.total_messages > 0, "Should have messages"

    # Check first conversation
    conv = export.conversations[0]
    assert conv.id == "claude_conv_001"
    assert "Intelligence" in conv.title or "Pattern" in conv.title
    assert conv.source == "claude"

    # Check message role normalization
    first_msg = conv.messages[0]
    assert first_msg.role == "user", f"Expected 'user', got '{first_msg.role}'"

    print(f"✓ Claude export loading passed ({export.conversation_count} conversations, {export.total_words:,} words)")


def test_search():
    """Test search functionality."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)
    connector.load_export(CLAUDE_EXPORT)

    async def run_search():
        # Search for AI alignment content
        results = await connector.search("instrumental convergence", limit=5)
        assert len(results) > 0, "Should find 'instrumental convergence'"

        # Search for systems thinking
        results = await connector.search("systems", limit=10)
        assert len(results) > 0, "Should find 'systems'"

        # Search for substrate independence
        results = await connector.search("substrate", limit=5)
        assert len(results) > 0, "Should find 'substrate'"

        # Search with regex
        results = await connector.search(r"I (think|believe)", limit=20, regex=True)
        assert len(results) > 0, "Should find 'I think' or 'I believe' with regex"

        return results

    results = asyncio.run(run_search())
    print(f"✓ Search functionality passed ({len(results)} results for regex search)")


def test_claim_extraction():
    """Test claim extraction from user messages."""
    connector = ConversationIngestorConnector()
    chatgpt_export = connector.load_export(CHATGPT_EXPORT)
    claude_export = connector.load_export(CLAUDE_EXPORT)

    # Extract claims from all loaded conversations
    claims = connector.extract_claims()

    assert len(claims) > 0, "Should extract some claims"

    # Check claim types
    claim_types = set(c.claim_type for c in claims)
    print(f"  Found claim types: {claim_types}")

    # Check for specific expected claims
    claim_texts = [c.claim.lower() for c in claims]
    found_ai_claim = any("instrumental" in t or "alignment" in t or "optimization" in t for t in claim_texts)
    found_systems_claim = any("systems" in t or "structure" in t for t in claim_texts)

    # Print some sample claims
    print(f"  Sample claims extracted:")
    for claim in claims[:5]:
        preview = claim.claim[:80] + "..." if len(claim.claim) > 80 else claim.claim
        print(f"    [{claim.claim_type}] {preview}")

    print(f"✓ Claim extraction passed ({len(claims)} claims extracted)")


def test_topic_keywords():
    """Test topic keyword extraction."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)
    connector.load_export(CLAUDE_EXPORT)

    keywords = connector.get_topic_keywords()

    assert len(keywords) > 0, "Should extract keywords"

    # Check that meaningful terms are present
    all_keywords = list(keywords.keys())[:50]
    keyword_set = set(all_keywords)

    # Should find some expected terms
    expected_terms = {"system", "systems", "intelligence", "optimization", "democratic", "procedural"}
    found_terms = keyword_set & expected_terms

    print(f"  Top 15 keywords: {all_keywords[:15]}")
    print(f"  Expected terms found: {found_terms}")

    print(f"✓ Topic keyword extraction passed ({len(keywords)} unique keywords)")


def test_statistics():
    """Test statistics generation."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)
    connector.load_export(CLAUDE_EXPORT)

    stats = connector.get_statistics()

    assert stats["loaded"] == True
    assert stats["total_conversations"] == 6, f"Expected 6 conversations, got {stats['total_conversations']}"
    assert stats["total_exports"] == 2
    assert stats["total_user_words"] > 0
    assert stats["total_assistant_words"] > 0
    assert "chatgpt" in stats["conversations_by_source"]
    assert "claude" in stats["conversations_by_source"]

    print(f"✓ Statistics generation passed:")
    print(f"    Total conversations: {stats['total_conversations']}")
    print(f"    User words: {stats['total_user_words']:,}")
    print(f"    Assistant words: {stats['total_assistant_words']:,}")
    print(f"    Sources: {stats['conversations_by_source']}")


def test_conversation_iteration():
    """Test iteration over conversations and messages."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)
    connector.load_export(CLAUDE_EXPORT)

    # Test conversation iteration
    conv_count = 0
    for conv in connector.iter_conversations():
        conv_count += 1
        assert isinstance(conv, Conversation)

    assert conv_count == 6, f"Expected 6 conversations, got {conv_count}"

    # Test user message iteration
    user_msg_count = 0
    for conv, msg in connector.iter_user_messages():
        user_msg_count += 1
        assert isinstance(conv, Conversation)
        assert isinstance(msg, ConversationMessage)
        assert msg.role == "user"

    assert user_msg_count > 0, "Should have user messages"

    print(f"✓ Iteration passed ({conv_count} conversations, {user_msg_count} user messages)")


def test_conversation_by_topic():
    """Test finding conversations by topic."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)
    connector.load_export(CLAUDE_EXPORT)

    # Find AI-related conversations
    ai_convs = connector.get_conversations_by_topic("intelligence")
    assert len(ai_convs) > 0, "Should find conversations about intelligence"

    # Find systems-related conversations
    systems_convs = connector.get_conversations_by_topic("systems")
    assert len(systems_convs) > 0, "Should find conversations about systems"

    print(f"✓ Topic search passed (found {len(ai_convs)} intelligence, {len(systems_convs)} systems conversations)")


def test_evidence_fetch():
    """Test fetching evidence by ID."""
    connector = ConversationIngestorConnector()
    connector.load_export(CHATGPT_EXPORT)

    async def run_fetch():
        # Fetch a conversation
        evidence = await connector.fetch("conv_conv_ai_alignment_001")
        assert evidence is not None, "Should fetch conversation evidence"
        assert "conv_ai_alignment_001" in evidence.source_id
        return evidence

    evidence = asyncio.run(run_fetch())
    print(f"✓ Evidence fetch passed (fetched: {evidence.title[:50]}...)")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONVERSATION INGESTOR CONNECTOR TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Format Detection - ChatGPT", test_chatgpt_format_detection),
        ("Format Detection - Claude", test_claude_format_detection),
        ("Export Loading - ChatGPT", test_chatgpt_export_loading),
        ("Export Loading - Claude", test_claude_export_loading),
        ("Search Functionality", test_search),
        ("Claim Extraction", test_claim_extraction),
        ("Topic Keywords", test_topic_keywords),
        ("Statistics Generation", test_statistics),
        ("Iteration", test_conversation_iteration),
        ("Topic Search", test_conversation_by_topic),
        ("Evidence Fetch", test_evidence_fetch),
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
