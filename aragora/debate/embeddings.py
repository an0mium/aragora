"""
Debate Embeddings Database for semantic search across historical debates.

Allows agents to retrieve and learn from past debates, enhancing continuity
and intelligence in the nomic loop.
"""

from typing import List, Tuple

from aragora.config import resolve_db_path
from aragora.memory.embeddings import SemanticRetriever
from aragora.persistence.models import DebateArtifact


class DebateEmbeddingsDatabase:
    """
    Semantic database for debate transcripts.

    Indexes full debate transcripts and allows similarity search
    to find relevant historical debates.
    """

    def __init__(self, db_path: str = "debate_embeddings.db"):
        self.retriever = SemanticRetriever(resolve_db_path(db_path))

    async def index_debate(self, debate: DebateArtifact) -> None:
        """
        Index a debate by embedding its full transcript.

        Args:
            debate: DebateArtifact to index
        """
        debate_id = f"{debate.loop_id}_{debate.cycle_number}_{debate.phase}"

        # Concatenate transcript into searchable text
        transcript_text = self._transcript_to_text(debate.transcript)

        # Add context about the debate
        full_text = f"""
Debate Context:
- Loop: {debate.loop_id}
- Cycle: {debate.cycle_number}
- Phase: {debate.phase}
- Task: {debate.task}
- Agents: {', '.join(debate.agents)}
- Consensus: {'Yes' if debate.consensus_reached else 'No'}
- Confidence: {debate.confidence}
- Winning Proposal: {debate.winning_proposal or 'None'}

Transcript:
{transcript_text}
        """.strip()

        await self.retriever.embed_and_store(debate_id, full_text)

    async def find_similar_debates(
        self, query: str, limit: int = 5, min_similarity: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Find debates similar to the query.

        Args:
            query: Search query (can be natural language)
            limit: Max results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (debate_id, excerpt, similarity) tuples
        """
        return await self.retriever.find_similar(query, limit, min_similarity)

    def _transcript_to_text(self, transcript: List[dict]) -> str:
        """Convert transcript list to readable text."""
        lines = []
        for msg in transcript:
            agent = msg.get("agent", "Unknown")
            content = msg.get("content", "")
            msg_type = msg.get("type", "message")
            lines.append(f"{agent} ({msg_type}): {content}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get database statistics."""
        return self.retriever.get_stats()
