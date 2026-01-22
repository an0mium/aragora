---
title: Document Analysis
description: Document Analysis
---

# Document Analysis

Natural language querying and analysis capabilities for uploaded documents.

## Overview

The Analysis module enables asking natural language questions about documents with AI-powered answer synthesis and source citations. It combines hybrid semantic search with LLM-powered answer generation.

**Key Features:**
- Natural language question answering
- Multiple query modes (factual, analytical, comparative, summary)
- Source citations with relevance scores
- Multi-turn conversation context
- Streaming response support
- Document comparison and summarization

## Quick Start

```python
from aragora.analysis import query_documents, summarize_document

# Quick query
result = await query_documents(
    question="What are the key terms in this contract?",
    document_ids=["doc_123"]
)
print(result.answer)
for citation in result.citations:
    print(f"  - {citation.document_name}: {citation.snippet}")

# Summarize a document
summary = await summarize_document(
    document_id="doc_123",
    focus="payment terms"
)
print(summary.answer)
```

## Full Engine Usage

```python
from aragora.analysis import DocumentQueryEngine, QueryConfig

# Create engine with custom config
config = QueryConfig(
    max_chunks=15,
    min_relevance=0.4,
    model="claude-3.5-sonnet",
    include_quotes=True,
)

engine = await DocumentQueryEngine.create(config=config)

# Simple query
result = await engine.query(
    question="What contracts mention exclusivity clauses?",
    workspace_id="ws_123",
    document_ids=["doc1", "doc2"]
)

# Multi-turn conversation
result1 = await engine.query(
    question="What is the notice period?",
    conversation_id="conv_456"
)
result2 = await engine.query(
    question="Are there any exceptions to that?",  # Context preserved
    conversation_id="conv_456"
)

# Compare documents
comparison = await engine.compare_documents(
    document_ids=["contract_v1", "contract_v2"],
    aspects=["payment terms", "liability clauses"]
)

# Extract structured information
fields = await engine.extract_information(
    document_ids=["contract_123"],
    extraction_template={
        "parties": "Who are the parties to this agreement?",
        "effective_date": "What is the effective date?",
        "term": "What is the term or duration?",
        "payment": "What are the payment terms?",
    }
)
```

---

## Query Modes

The engine automatically detects query intent or you can specify the mode explicitly.

| Mode | Description | Triggers |
|------|-------------|----------|
| `FACTUAL` | Direct fact extraction | Default mode |
| `ANALYTICAL` | Analysis and reasoning | "why", "analyze", "explain" |
| `COMPARATIVE` | Compare across documents | "compare", "difference", "vs" |
| `SUMMARY` | Summarize content | "summarize", "overview" |
| `EXTRACTIVE` | Extract specific information | "list", "extract", "find all" |

---

## Configuration

### QueryConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_chunks` | `10` | Maximum chunks to retrieve per query |
| `min_relevance` | `0.3` | Minimum relevance score threshold |
| `vector_weight` | `0.7` | Weight for semantic vs keyword search |
| `max_answer_length` | `500` | Maximum answer length in words |
| `include_quotes` | `True` | Include direct quotes from sources |
| `require_citations` | `True` | Always cite sources in answers |
| `model` | `claude-3.5-sonnet` | Primary model for answer generation |
| `fallback_model` | `gemini-1.5-flash` | Fallback if primary fails |
| `expand_query` | `True` | Generate query variations for better retrieval |
| `detect_intent` | `True` | Auto-detect question type/intent |
| `enable_context` | `True` | Use conversation history |
| `max_context_turns` | `3` | Maximum conversation turns to include |

---

## Response Types

### QueryResult

```python
@dataclass
class QueryResult:
    query_id: str           # Unique query identifier
    question: str           # Original question
    answer: str             # Generated answer
    confidence: AnswerConfidence  # HIGH, MEDIUM, LOW, NONE
    citations: list[Citation]     # Source citations
    query_mode: QueryMode         # Detected query mode
    chunks_searched: int          # Total chunks searched
    chunks_relevant: int          # Chunks meeting relevance threshold
    processing_time_ms: int       # Processing time
    model_used: str              # Model that generated answer
```

### Citation

```python
@dataclass
class Citation:
    document_id: str        # Source document ID
    document_name: str      # Document name
    chunk_id: str           # Specific chunk ID
    snippet: str            # Relevant excerpt (200 chars)
    page: int | None        # Page number if available
    relevance_score: float  # Relevance score (0-1)
    heading_context: str    # Section heading context
```

### AnswerConfidence

| Level | Description |
|-------|-------------|
| `HIGH` | Strong evidence, clear answer (max relevance > 0.8, avg > 0.5) |
| `MEDIUM` | Moderate evidence, likely answer (max relevance > 0.5) |
| `LOW` | Weak evidence, uncertain |
| `NONE` | No relevant information found |

---

## Streaming Responses

For real-time UI updates, use streaming:

```python
async for chunk in engine.query_stream(
    question="Summarize the agreement",
    document_ids=["doc_123"]
):
    if chunk.is_final:
        # Final chunk includes citations
        for citation in chunk.citations:
            print(f"Source: {citation.document_name}")
    else:
        # Partial answer text
        print(chunk.text, end="", flush=True)
```

---

## API Endpoints

The Analysis module is exposed via the Knowledge API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/query` | Natural language query |
| POST | `/api/knowledge/mound/query` | Semantic query against knowledge mound |

### Example Request

```bash
curl -X POST http://localhost:8080/api/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the payment terms?",
    "workspace_id": "enterprise",
    "document_ids": ["contract_001"],
    "options": {
      "max_chunks": 10,
      "min_relevance": 0.4,
      "include_quotes": true
    }
  }'
```

### Example Response

```json
{
  "query_id": "query_abc123def456",
  "question": "What are the payment terms?",
  "answer": "According to the contract [Source 1], payment is due within 30 days of invoice...",
  "confidence": "high",
  "citations": [
    {
      "document_id": "contract_001",
      "document_name": "Service Agreement.pdf",
      "chunk_id": "chunk_789",
      "snippet": "Payment Terms: All invoices are due and payable within thirty (30) days...",
      "page": 4,
      "relevance_score": 0.92,
      "heading_context": "Section 5: Payment"
    }
  ],
  "query_mode": "factual",
  "chunks_searched": 42,
  "chunks_relevant": 8,
  "processing_time_ms": 1250,
  "model_used": "claude-3.5-sonnet"
}
```

---

## Integration with Knowledge Mound

The Analysis module integrates with the Knowledge Mound for broader knowledge queries:

```python
from aragora.knowledge.mound import KnowledgeMound
from aragora.analysis import DocumentQueryEngine

# Initialize both systems
mound = KnowledgeMound(workspace_id="enterprise")
await mound.initialize()

engine = await DocumentQueryEngine.create()

# Query documents
doc_result = await engine.query("What are our SLA requirements?")

# Query knowledge mound for related facts
mound_result = await mound.query("SLA requirements", limit=5)

# Combine insights from both
combined_answer = f"""
Document Analysis:
{doc_result.answer}

Related Knowledge:
{chr(10).join(item.content for item in mound_result.items)}
"""
```

---

## Best Practices

1. **Scope Queries When Possible** - Use `document_ids` to limit search scope for faster, more relevant results

2. **Use Conversation IDs for Follow-ups** - Enable multi-turn context with `conversation_id` for related questions

3. **Adjust Relevance Thresholds** - Lower `min_relevance` for broader searches, raise for precision

4. **Monitor Confidence Levels** - Check `result.confidence` before displaying answers to users

5. **Handle NONE Confidence** - When confidence is NONE, suggest users refine their question or upload more relevant documents

6. **Use Streaming for Long Answers** - For better UX, use `query_stream` to show answers as they're generated

---

## Error Handling

```python
result = await engine.query("What is the penalty clause?")

if result.confidence == AnswerConfidence.NONE:
    print("No relevant information found in the documents.")
elif result.confidence == AnswerConfidence.LOW:
    print("Warning: Low confidence answer - please verify manually.")
    print(f"Answer: {result.answer}")
else:
    print(result.answer)
    for citation in result.citations:
        print(f"  Source: {citation.document_name}, p.{citation.page}")
```

---

## See Also

- [Knowledge Mound](../KNOWLEDGE_MOUND.md) - Unified knowledge storage
- [Document Processing](../DOCUMENTS.md) - Document upload and chunking
- [API Reference](../API_REFERENCE.md) - Full API documentation
