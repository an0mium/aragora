"""
Dataset Query Engine - Natural language query interface for the Knowledge Base.

Answers questions about document datasets using:
1. Semantic search via embeddings
2. Fact extraction via agents
3. Multi-agent debate for conflicting interpretations
4. Consensus verification for high-confidence answers
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Union

from aragora.knowledge.embeddings import (
    ChunkMatch,
    InMemoryEmbeddingService,
    WeaviateEmbeddingService,
)
from aragora.knowledge.fact_store import FactStore, InMemoryFactStore
from aragora.knowledge.types import (
    Fact,
    FactFilters,
    QueryResult,
    ValidationStatus,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agents that can answer questions."""

    name: str

    async def generate(self, prompt: str, context: list[dict[str, str]]) -> str:
        """Generate a response to a prompt with context."""
        ...


@dataclass
class QueryOptions:
    """Options for query execution."""

    # Search parameters
    max_chunks: int = 10
    search_alpha: float = 0.5  # 0=vector, 1=keyword
    min_chunk_score: float = 0.0

    # Agent parameters
    use_agents: bool = True
    extract_facts: bool = True
    verify_answer: bool = False

    # Multi-agent debate parameters
    use_debate: bool = False  # Enable multi-agent debate for complex queries
    debate_rounds: int = 2  # Number of debate rounds
    require_consensus: float = 0.66  # Agreement threshold for consensus
    parallel_agents: bool = True  # Run agents in parallel

    # Answer parameters
    max_answer_tokens: int = 1024
    include_citations: bool = True

    # Fact store parameters
    save_extracted_facts: bool = True
    min_fact_confidence: float = 0.5


@dataclass
class QueryContext:
    """Context for query execution."""

    query: str
    workspace_id: str
    options: QueryOptions
    chunks: list[ChunkMatch] = field(default_factory=list)
    extracted_facts: list[Fact] = field(default_factory=list)
    agent_responses: dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


class DatasetQueryEngine:
    """Natural language query interface for document datasets.

    Combines semantic search with agent-based analysis to answer
    questions about uploaded documents.

    Usage:
        engine = DatasetQueryEngine(
            fact_store=FactStore(),
            embedding_service=WeaviateEmbeddingService(),
        )

        result = await engine.query(
            question="What are the payment terms in the contract?",
            workspace_id="ws_123",
        )

        print(result.answer)
        print(result.facts)
        print(result.confidence)
    """

    def __init__(
        self,
        fact_store: Optional[Union[FactStore, InMemoryFactStore]] = None,
        embedding_service: Optional[
            Union[WeaviateEmbeddingService, InMemoryEmbeddingService]
        ] = None,
        agents: Optional[list[AgentProtocol]] = None,
        default_agent: Optional[AgentProtocol] = None,
    ):
        """Initialize the query engine.

        Args:
            fact_store: Fact storage backend
            embedding_service: Embedding/search backend
            agents: Optional list of agents for analysis
            default_agent: Default agent for answering questions
        """
        self._fact_store = fact_store or InMemoryFactStore()
        self._embedding_service = embedding_service or InMemoryEmbeddingService()
        self._agents = agents or []
        self._default_agent = default_agent

        # Progress callback
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set a callback for progress updates.

        Args:
            callback: Function(stage: str, progress: float)
        """
        self._progress_callback = callback

    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(stage, progress)

    async def query(
        self,
        question: str,
        workspace_id: str,
        options: Optional[QueryOptions] = None,
    ) -> QueryResult:
        """Answer a question about the dataset using multi-agent analysis.

        Args:
            question: Natural language question
            workspace_id: Workspace containing documents
            options: Query options

        Returns:
            QueryResult with answer, facts, evidence, and confidence

        Example:
            result = await engine.query(
                "What are the key contract terms?",
                "ws_123",
            )
        """
        options = options or QueryOptions()
        ctx = QueryContext(
            query=question,
            workspace_id=workspace_id,
            options=options,
        )

        try:
            # Stage 1: Search for relevant chunks
            self._report_progress("searching", 0.1)
            ctx.chunks = await self._search_chunks(ctx)

            if not ctx.chunks:
                return self._empty_result(ctx, "No relevant content found")

            # Stage 2: Check existing facts
            self._report_progress("checking_facts", 0.3)
            existing_facts = await self._get_existing_facts(ctx)

            # Stage 3: Generate answer from chunks
            self._report_progress("generating_answer", 0.5)
            answer = await self._generate_answer(ctx)

            # Stage 4: Extract new facts if enabled
            if options.extract_facts:
                self._report_progress("extracting_facts", 0.7)
                ctx.extracted_facts = await self._extract_facts(ctx, answer)

            # Stage 5: Calculate confidence
            self._report_progress("calculating_confidence", 0.9)
            confidence = self._calculate_confidence(ctx, existing_facts)

            # Combine facts
            all_facts = existing_facts + ctx.extracted_facts

            # Build result
            self._report_progress("complete", 1.0)
            processing_time = int((time.time() - ctx.start_time) * 1000)

            return QueryResult(
                answer=answer,
                facts=all_facts,
                evidence_ids=[c.chunk_id for c in ctx.chunks],
                confidence=confidence,
                query=question,
                workspace_id=workspace_id,
                processing_time_ms=processing_time,
                agent_contributions=ctx.agent_responses,
                metadata={
                    "chunks_searched": len(ctx.chunks),
                    "existing_facts": len(existing_facts),
                    "extracted_facts": len(ctx.extracted_facts),
                },
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return self._error_result(ctx, str(e))

    async def _search_chunks(self, ctx: QueryContext) -> list[ChunkMatch]:
        """Search for relevant chunks."""
        try:
            return await self._embedding_service.hybrid_search(
                query=ctx.query,
                workspace_id=ctx.workspace_id,
                limit=ctx.options.max_chunks,
                alpha=ctx.options.search_alpha,
                min_score=ctx.options.min_chunk_score,
            )
        except Exception as e:
            logger.warning(f"Embedding search failed, trying keyword: {e}")
            return await self._embedding_service.keyword_search(
                query=ctx.query,
                workspace_id=ctx.workspace_id,
                limit=ctx.options.max_chunks,
            )

    async def _get_existing_facts(self, ctx: QueryContext) -> list[Fact]:
        """Get existing facts relevant to the query."""
        filters = FactFilters(
            workspace_id=ctx.workspace_id,
            min_confidence=ctx.options.min_fact_confidence,
            limit=20,
        )
        return self._fact_store.query_facts(ctx.query, filters)

    async def _generate_answer(self, ctx: QueryContext) -> str:
        """Generate answer from chunks using agent(s)."""
        if not ctx.options.use_agents or not (self._default_agent or self._agents):
            return self._synthesize_from_chunks(ctx)

        # Use multi-agent debate if enabled and multiple agents available
        if ctx.options.use_debate and len(self._agents) >= 2:
            return await self._generate_with_debate(ctx)

        # Single agent answer
        return await self._generate_single_agent_answer(ctx)

    async def _generate_single_agent_answer(self, ctx: QueryContext) -> str:
        """Generate answer using a single agent."""
        agent = self._default_agent or (self._agents[0] if self._agents else None)
        if not agent:
            return self._synthesize_from_chunks(ctx)

        # Build context from chunks
        chunk_texts = []
        for i, chunk in enumerate(ctx.chunks, 1):
            chunk_texts.append(f"[{i}] {chunk.content[:500]}...")

        chunks_context = "\n\n".join(chunk_texts)

        prompt = f"""Based on the following document excerpts, answer this question:

Question: {ctx.query}

Document Excerpts:
{chunks_context}

Provide a clear, factual answer based only on the information in the excerpts.
If the excerpts don't contain enough information, say so.
Include references to excerpt numbers [N] when citing information."""

        try:
            response = await agent.generate(prompt, [])
            ctx.agent_responses[agent.name] = response
            return response
        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return self._synthesize_from_chunks(ctx)

    async def _generate_with_debate(self, ctx: QueryContext) -> str:
        """Generate answer using multi-agent debate.

        Multiple agents analyze the content and debate to reach consensus.
        This produces more reliable answers for complex queries.
        """
        # Build context from chunks
        chunk_texts = []
        for i, chunk in enumerate(ctx.chunks, 1):
            chunk_texts.append(f"[{i}] {chunk.content[:500]}...")
        chunks_context = "\n\n".join(chunk_texts)

        # Phase 1: Get initial answers from all agents
        initial_prompt = f"""Based on the following document excerpts, answer this question:

Question: {ctx.query}

Document Excerpts:
{chunks_context}

Provide a clear, factual answer based only on the information in the excerpts.
Include references to excerpt numbers [N] when citing information.
If you're uncertain about any claims, indicate your confidence level."""

        responses: dict[str, str] = {}

        if ctx.options.parallel_agents:
            # Run agents in parallel
            tasks = []
            for agent in self._agents:
                tasks.append(self._safe_generate(agent, initial_prompt))

            results = await asyncio.gather(*tasks)
            for agent, response in zip(self._agents, results):
                if response:
                    responses[agent.name] = response
                    ctx.agent_responses[agent.name] = response
        else:
            # Run agents sequentially
            for agent in self._agents:
                response = await self._safe_generate(agent, initial_prompt)
                if response:
                    responses[agent.name] = response
                    ctx.agent_responses[agent.name] = response

        if not responses:
            return self._synthesize_from_chunks(ctx)

        if len(responses) == 1:
            return list(responses.values())[0]

        # Phase 2: Debate rounds for refinement
        debate_context = responses.copy()

        for round_num in range(ctx.options.debate_rounds):
            critique_prompt = f"""Review the following answers to this question and identify any disagreements or areas that need clarification:

Question: {ctx.query}

Previous Answers:
{self._format_responses(debate_context)}

If there are significant disagreements:
1. Identify the key points of disagreement
2. Analyze which answer is better supported by the document excerpts
3. Provide your refined answer that addresses the disagreements

If the answers largely agree:
1. Synthesize the best elements from each answer
2. Provide your refined, consensus answer

Document Excerpts (for reference):
{chunks_context[:2000]}..."""

            # Get critique from one agent per round (rotating)
            critique_agent = self._agents[round_num % len(self._agents)]
            critique = await self._safe_generate(critique_agent, critique_prompt)

            if critique:
                debate_context[f"{critique_agent.name}_round{round_num + 1}"] = critique
                ctx.agent_responses[f"{critique_agent.name}_round{round_num + 1}"] = critique

        # Phase 3: Synthesize final answer
        synthesis_prompt = f"""Based on the following debate about this question, synthesize a final consensus answer:

Question: {ctx.query}

Debate Summary:
{self._format_responses(debate_context)}

Provide a single, definitive answer that:
1. Represents the consensus view where agents agreed
2. Acknowledges remaining uncertainties or disagreements if any
3. Is supported by the document excerpts
4. Uses citations [N] to reference specific excerpts"""

        synthesis_agent = self._default_agent or self._agents[0]
        final_answer = await self._safe_generate(synthesis_agent, synthesis_prompt)

        if final_answer:
            ctx.agent_responses["consensus"] = final_answer
            return final_answer

        # Fallback to first response if synthesis fails
        return list(responses.values())[0]

    async def _safe_generate(self, agent: AgentProtocol, prompt: str) -> Optional[str]:
        """Safely generate a response, returning None on failure."""
        try:
            return await agent.generate(prompt, [])
        except Exception as e:
            logger.warning(f"Agent {agent.name} failed: {e}")
            return None

    def _format_responses(self, responses: dict[str, str]) -> str:
        """Format multiple agent responses for context."""
        formatted = []
        for agent_name, response in responses.items():
            # Truncate long responses
            truncated = response[:1500] + "..." if len(response) > 1500 else response
            formatted.append(f"[{agent_name}]:\n{truncated}")
        return "\n\n".join(formatted)

    def _synthesize_from_chunks(self, ctx: QueryContext) -> str:
        """Synthesize answer directly from chunks without agent."""
        if not ctx.chunks:
            return "No relevant information found."

        # Simple extraction of relevant sentences
        query_terms = set(ctx.query.lower().split())
        relevant_sentences = []

        for chunk in ctx.chunks[:5]:  # Top 5 chunks
            sentences = chunk.content.split(".")
            for sentence in sentences:
                sentence_terms = set(sentence.lower().split())
                overlap = len(query_terms & sentence_terms)
                if overlap >= 2 and len(sentence) > 20:
                    relevant_sentences.append(sentence.strip() + ".")
                    if len(relevant_sentences) >= 5:
                        break
            if len(relevant_sentences) >= 5:
                break

        if relevant_sentences:
            return " ".join(relevant_sentences)
        else:
            return f"Found {len(ctx.chunks)} relevant sections but could not extract a direct answer. Please review the source documents."

    async def _extract_facts(self, ctx: QueryContext, answer: str) -> list[Fact]:
        """Extract facts from the answer and chunks."""
        facts = []

        if not ctx.options.use_agents or not self._default_agent:
            return facts

        # Build extraction prompt
        prompt = f"""Based on the following answer and source excerpts, extract specific factual statements.

Question: {ctx.query}

Answer: {answer}

Extract 3-5 specific, verifiable facts from this answer.
Format each fact on its own line starting with "FACT: "
Only include facts that are directly supported by the source material."""

        try:
            response = await self._default_agent.generate(prompt, [])

            # Parse facts from response
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("FACT:"):
                    statement = line[5:].strip()
                    if len(statement) > 10:
                        fact = self._fact_store.add_fact(
                            statement=statement,
                            workspace_id=ctx.workspace_id,
                            evidence_ids=[c.chunk_id for c in ctx.chunks[:3]],
                            source_documents=[c.document_id for c in ctx.chunks[:3]],
                            confidence=0.6,  # Initial confidence
                            validation_status=ValidationStatus.UNVERIFIED,
                        )
                        facts.append(fact)

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

        return facts

    def _calculate_confidence(self, ctx: QueryContext, existing_facts: list[Fact]) -> float:
        """Calculate confidence score for the answer."""
        factors = []

        # Factor 1: Chunk relevance scores
        if ctx.chunks:
            avg_chunk_score = sum(c.score for c in ctx.chunks) / len(ctx.chunks)
            factors.append(avg_chunk_score)

        # Factor 2: Existing fact support
        if existing_facts:
            verified_count = sum(1 for f in existing_facts if f.is_verified)
            fact_confidence = verified_count / len(existing_facts)
            factors.append(fact_confidence)

        # Factor 3: Agent availability
        if self._default_agent and ctx.agent_responses:
            factors.append(0.8)  # Higher confidence with agent
        else:
            factors.append(0.5)  # Lower without agent

        # Factor 4: Number of supporting chunks
        chunk_factor = min(len(ctx.chunks) / 5.0, 1.0)  # 5+ chunks = max
        factors.append(chunk_factor)

        if factors:
            return sum(factors) / len(factors)
        return 0.3  # Default low confidence

    def _empty_result(self, ctx: QueryContext, message: str) -> QueryResult:
        """Create empty result."""
        return QueryResult(
            answer=message,
            facts=[],
            evidence_ids=[],
            confidence=0.0,
            query=ctx.query,
            workspace_id=ctx.workspace_id,
            processing_time_ms=int((time.time() - ctx.start_time) * 1000),
            metadata={"error": "no_content"},
        )

    def _error_result(self, ctx: QueryContext, error: str) -> QueryResult:
        """Create error result."""
        return QueryResult(
            answer=f"Query failed: {error}",
            facts=[],
            evidence_ids=[],
            confidence=0.0,
            query=ctx.query,
            workspace_id=ctx.workspace_id,
            processing_time_ms=int((time.time() - ctx.start_time) * 1000),
            metadata={"error": error},
        )

    async def get_facts_for_query(
        self,
        question: str,
        workspace_id: str,
        min_confidence: float = 0.5,
        limit: int = 20,
    ) -> list[Fact]:
        """Get facts relevant to a query without generating new answer.

        Useful for checking existing knowledge before running full query.

        Args:
            question: Search query
            workspace_id: Workspace to search
            min_confidence: Minimum confidence filter
            limit: Maximum facts to return

        Returns:
            List of relevant facts
        """
        filters = FactFilters(
            workspace_id=workspace_id,
            min_confidence=min_confidence,
            limit=limit,
        )
        return self._fact_store.query_facts(question, filters)

    async def verify_fact(
        self,
        fact_id: str,
        agents: Optional[list[AgentProtocol]] = None,
    ) -> Fact:
        """Verify a fact using multiple agents.

        Args:
            fact_id: Fact to verify
            agents: Agents to use for verification

        Returns:
            Updated fact with verification results
        """
        fact = self._fact_store.get_fact(fact_id)
        if not fact:
            raise ValueError(f"Fact not found: {fact_id}")

        agents = agents or self._agents
        if not agents:
            logger.warning("No agents available for verification")
            return fact

        # Ask each agent to verify
        votes: dict[str, bool] = {}

        for agent in agents:
            prompt = f"""Verify whether the following statement is factually accurate:

Statement: {fact.statement}

Respond with:
- TRUE if the statement is accurate
- FALSE if the statement is inaccurate or misleading
- UNCERTAIN if you cannot determine accuracy

Then briefly explain your reasoning."""

            try:
                response = await agent.generate(prompt, [])
                response_upper = response.upper()

                if "TRUE" in response_upper[:50]:
                    votes[agent.name] = True
                elif "FALSE" in response_upper[:50]:
                    votes[agent.name] = False
                # UNCERTAIN doesn't vote

            except Exception as e:
                logger.warning(f"Agent {agent.name} verification failed: {e}")

        # Calculate new status based on votes
        if votes:
            agree_count = sum(1 for v in votes.values() if v)
            total = len(votes)
            agree_ratio = agree_count / total

            if agree_ratio >= 0.66:
                new_status = ValidationStatus.MAJORITY_AGREED
                new_confidence = agree_ratio
            elif agree_ratio <= 0.33:
                new_status = ValidationStatus.CONTESTED
                new_confidence = 0.3
            else:
                new_status = ValidationStatus.UNVERIFIED
                new_confidence = fact.confidence

            self._fact_store.update_fact(
                fact_id,
                validation_status=new_status,
                confidence=new_confidence,
            )

            return self._fact_store.get_fact(fact_id) or fact

        return fact

    def close(self) -> None:
        """Close resources."""
        try:
            self._embedding_service.close()
        except Exception as e:
            # Ignore cleanup errors but log for debugging
            logger.debug(f"Error closing embedding service: {e}")


class SimpleQueryEngine:
    """Simplified query engine without agent dependencies.

    For basic search and retrieval when agents aren't available.
    """

    def __init__(
        self,
        fact_store: Optional[Union[FactStore, InMemoryFactStore]] = None,
        embedding_service: Optional[
            Union[WeaviateEmbeddingService, InMemoryEmbeddingService]
        ] = None,
    ):
        """Initialize simple engine."""
        self._fact_store = fact_store or InMemoryFactStore()
        self._embedding_service = embedding_service or InMemoryEmbeddingService()

    async def query(
        self,
        question: str,
        workspace_id: str,
        options: Optional[QueryOptions] = None,
    ) -> QueryResult:
        """Simple query that returns search results without agent analysis.

        For basic search when agents aren't available.
        """
        options = options or QueryOptions()
        start_time = time.time()

        # Search for relevant chunks
        chunks = await self.search(question, workspace_id, options.max_chunks)

        # Get relevant facts
        facts = await self.get_facts(question, workspace_id, limit=10)

        # Build simple answer from chunks
        if chunks:
            answer = f"Found {len(chunks)} relevant sections. "
            if facts:
                answer += f"Extracted {len(facts)} relevant facts from the documents."
            else:
                answer += "Review the source documents for details."
        else:
            answer = "No relevant content found for your query."

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResult(
            answer=answer,
            facts=facts,
            evidence_ids=[c.chunk_id for c in chunks],
            confidence=0.3 if chunks else 0.0,  # Low confidence without agent
            query=question,
            workspace_id=workspace_id,
            processing_time_ms=processing_time,
            metadata={"mode": "simple", "chunks_found": len(chunks)},
        )

    async def search(
        self,
        query: str,
        workspace_id: str,
        limit: int = 10,
    ) -> list[ChunkMatch]:
        """Search for relevant chunks."""
        return await self._embedding_service.hybrid_search(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
        )

    async def get_facts(
        self,
        query: str,
        workspace_id: str,
        limit: int = 20,
    ) -> list[Fact]:
        """Get relevant facts."""
        filters = FactFilters(workspace_id=workspace_id, limit=limit)
        return self._fact_store.query_facts(query, filters)

    def add_fact(
        self,
        statement: str,
        workspace_id: str,
        evidence_ids: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
    ) -> Fact:
        """Add a fact directly."""
        return self._fact_store.add_fact(
            statement=statement,
            workspace_id=workspace_id,
            evidence_ids=evidence_ids,
            source_documents=source_documents,
        )

    def close(self) -> None:
        """Close resources."""
        try:
            self._embedding_service.close()
        except Exception as e:
            # Ignore cleanup errors but log for debugging
            logger.debug(f"Error closing embedding service: {e}")
