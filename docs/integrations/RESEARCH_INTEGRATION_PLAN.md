# Aragora Research Integration Plan

## Executive Summary

This document provides a detailed implementation plan for integrating cutting-edge AI research into Aragora. Each recommendation builds on existing systems and includes specific code locations, APIs, and integration points.

**Guiding Principles:**
1. Build on existing RLM/REPL architecture (already sophisticated)
2. Enhance rather than replace existing calibration/consensus systems
3. Resolve conflicts by selecting the more synergistic approach
4. Phase implementations to minimize risk and maximize incremental value

---

## Pairwise Conflict Analysis & Resolutions

### Conflict 0: Adaptive Stability Detection vs. Fixed Rounds / Existing Consensus Estimator

**Existing:** `aragora/debate/ml_integration.py` already provides `ConsensusEstimator` + early termination hooks  
**Proposed:** Beta‑Binomial stability detection (MAD/Judge) for adaptive stopping

**Resolution:** ✅ **UPGRADE EXISTING** - Extend `ConsensusEstimator`
- The early termination path already exists (Arena/Orchestrator config + helpers)
- Add stability detection to **replace** rigid round counts, not compete with them
- Keep existing threshold as a **safety cap**, but stop earlier when stable

**Codebase fit check:** This is a low‑risk change because the early‑termination wiring already exists in:
`aragora/debate/ml_integration.py`, `arena_helpers.py`, `arena_initializer.py`, and `orchestrator_state.py`.

### Conflict 0b: LaRA Routing vs. Fixed Retrieval Path

**Existing:** `QueryOperationsMixin.query()` + `query_semantic()` + `query_graph()` + RLM mixin  
**Proposed:** LaRA‑style router (RAG vs Long‑Context vs RLM) based on query traits

**Resolution:** ✅ **ENHANCEMENT** - Add routing layer without removing existing behavior
- Keep `query()` as entry point; insert router decision before selecting retrieval mode
- Allow explicit override to keep deterministic behavior for audits

**Codebase fit check:** `aragora/knowledge/mound/api/query.py` + `api/rlm.py` already provide the hooks.

### Conflict 1: MUSE Multi-LLM Uncertainty vs. Existing Brier Calibration

**Existing:** `aragora/ranking/calibration_engine.py` uses Brier score for per-agent calibration
**Proposed:** MUSE uses Jensen-Shannon Divergence across model subsets

**Resolution:** ✅ **COMPLEMENTARY** - Integrate both
- Existing Brier calibration operates at **agent level** (individual model accuracy)
- MUSE operates at **ensemble level** (cross-model agreement)
- Use Brier for agent selection, MUSE for consensus confidence scoring

### Conflict 2: ThinkPRM vs. Existing Consensus Verification

**Existing:** `consensus_verification.py` verifies final claims with Z3/Lean4
**Proposed:** ThinkPRM provides step-by-step process reward scoring

**Resolution:** ✅ **COMPLEMENTARY** - Layer them
- ThinkPRM scores each debate round (process supervision)
- Existing formal verification validates final consensus (outcome verification)
- Combined: PRM catches reasoning errors early, formal verification ensures correctness

### Conflict 3: A-HMAD Dynamic Roles vs. Static Domain Mapping

**Existing:** `DOMAIN_CAPABILITY_MAP` in team_selector.py is static
**Proposed:** A-HMAD dynamically assigns expertise roles based on debate context

**Resolution:** ⚡ **UPGRADE** - Replace static with dynamic
- A-HMAD subsumes static mapping with learned specialization
- Keep static mapping as fallback/initialization
- Dynamic roles adapt to specific debate topics

### Conflict 4: GraphRAG vs. Knowledge Mound Semantic Search

**Existing:** `QueryOperationsMixin` does vector-based semantic search
**Proposed:** GraphRAG combines graph traversal with vector retrieval

**Resolution:** ✅ **ENHANCEMENT** - Extend existing
- Knowledge Mound already has relationship edges (supports/contradicts)
- Add graph traversal layer on top of existing vector search
- Hybrid retrieval: vector similarity + relationship paths

### Conflict 5: ClaimCheck Stepwise vs. Evidence Grounding

**Existing:** `evidence_grounding.py` uses keyword matching (min 2 keywords)
**Proposed:** ClaimCheck decomposes claims into atomic sub-claims

**Resolution:** ⚡ **UPGRADE** - Replace keyword matching
- ClaimCheck's stepwise verification is strictly better
- Keep reliability scoring, upgrade matching algorithm
- Atomic decomposition enables finer-grained verification

### Conflict 6: ASCoT Late-Stage Focus vs. Uniform Debate Rounds

**Existing:** All debate rounds treated equally in weight calculation
**Proposed:** ASCoT identifies "Late-Stage Fragility" - later rounds more error-prone

**Resolution:** ⚡ **UPGRADE** - Add round-aware weighting
- Apply ASCoT's insight: weight verification effort toward later rounds
- Earlier rounds: lower scrutiny, faster iteration
- Later rounds: higher scrutiny, targeted correction

### Conflict 7: Gödel Agent vs. SICA Self-Improvement

**Both:** Self-modifying agents for Nomic Loop
**Gödel:** Modifies both policy AND learning algorithm via monkey patching
**SICA:** Focused on coding task self-improvement

**Resolution:** ⚡ **SELECT SICA** - More aligned with Aragora
- Gödel's runtime monkey patching is high-risk for production
- SICA's focused approach better fits TestFixer/Nomic Loop
- SICA achieved 17%→53% on SWE-Bench (proven gains)

### Conflict 8: Hilbert (Apple) vs. Safe Framework (Lean4)

**Both:** Formal verification of reasoning chains
**Hilbert:** Recursive informal→formal proof generation
**Safe:** Step-by-step Lean4 translation of each CoT step

**Resolution:** ⚡ **SELECT Hilbert approach** - Better fit
- Aragora already has Z3/Lean4 verification infrastructure
- Hilbert's recursive approach aligns with RLM's hierarchical context
- 99.2% miniF2F vs Safe's lower benchmarks

---

## Phase 1: Foundation Enhancements (Weeks 1-4)

### 1.0 Adaptive Stability Detection (MAD/Judge)

**Goal:** Stop debate rounds when consensus becomes statistically stable (Beta‑Binomial), lowering compute cost.

**Codebase integration points:**
- `aragora/debate/ml_integration.py` → extend `ConsensusEstimator` with stability scoring
- `aragora/debate/arena_helpers.py` → use stability score in early‑termination check
- `aragora/debate/arena_sub_configs.py` → add config flags for stability detection

**Why this fits:** The early‑termination mechanism already exists; we only need to enrich it.

### 1.1 MUSE Multi-Model Uncertainty Integration

**Goal:** Add ensemble-level uncertainty quantification using Jensen-Shannon Divergence

**Files to modify:**
- `aragora/ranking/calibration_engine.py` - Add MUSEEnsembleCalibration class
- `aragora/debate/phases/consensus_phase.py` - Integrate MUSE scores into vote weights
- `aragora/debate/phases/weight_calculator.py` - Add muse_weight parameter

**Implementation:**

```python
# aragora/ranking/muse_calibration.py (NEW FILE)
"""
MUSE: Multi-LLM Uncertainty via Subset Ensembles
Based on: https://pmc.ncbi.nlm.nih.gov/articles/PMC12702469/
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from scipy.spatial.distance import jensenshannon

@dataclass
class MUSEResult:
    """Result from MUSE ensemble uncertainty calculation."""
    consensus_confidence: float  # 0.0-1.0, higher = more agreement
    divergence_score: float      # JSD, lower = more agreement
    best_subset: Set[str]        # Agent IDs in best-calibrated subset
    subset_agreement: float      # Agreement within best subset

class MUSECalculator:
    """
    Calculates ensemble uncertainty using Jensen-Shannon Divergence.

    Key insight: Well-calibrated model subsets produce more reliable
    uncertainty estimates than full ensemble averaging.
    """

    def __init__(self, min_subset_size: int = 2, max_subset_size: int = 5):
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self._calibration_history: Dict[str, List[float]] = {}

    def calculate_ensemble_uncertainty(
        self,
        agent_responses: Dict[str, Dict],  # agent_id -> {answer, confidence, distribution}
        historical_calibration: Dict[str, float]  # agent_id -> Brier score
    ) -> MUSEResult:
        """
        Calculate MUSE uncertainty score for an ensemble of responses.

        Steps:
        1. Enumerate subsets of agents
        2. Score each subset by historical calibration
        3. Calculate JSD for best subset
        4. Return consensus confidence
        """
        if len(agent_responses) < self.min_subset_size:
            # Fallback to simple averaging
            avg_confidence = np.mean([r['confidence'] for r in agent_responses.values()])
            return MUSEResult(
                consensus_confidence=avg_confidence,
                divergence_score=0.0,
                best_subset=set(agent_responses.keys()),
                subset_agreement=avg_confidence
            )

        # Find best-calibrated subset
        best_subset, best_score = self._find_best_subset(
            agent_responses, historical_calibration
        )

        # Calculate JSD for best subset
        distributions = [
            agent_responses[agent_id].get('distribution', [agent_responses[agent_id]['confidence']])
            for agent_id in best_subset
        ]

        # Normalize distributions to same length
        max_len = max(len(d) for d in distributions)
        normalized = [self._normalize_distribution(d, max_len) for d in distributions]

        # Pairwise JSD
        jsd_scores = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                jsd = jensenshannon(normalized[i], normalized[j])
                jsd_scores.append(jsd)

        avg_jsd = np.mean(jsd_scores) if jsd_scores else 0.0

        # Convert JSD to confidence (lower divergence = higher confidence)
        consensus_confidence = 1.0 - min(avg_jsd, 1.0)

        # Calculate subset agreement
        subset_confidences = [agent_responses[a]['confidence'] for a in best_subset]
        subset_agreement = 1.0 - np.std(subset_confidences)

        return MUSEResult(
            consensus_confidence=consensus_confidence,
            divergence_score=avg_jsd,
            best_subset=best_subset,
            subset_agreement=subset_agreement
        )

    def _find_best_subset(
        self,
        responses: Dict[str, Dict],
        calibration: Dict[str, float]
    ) -> Tuple[Set[str], float]:
        """Find the best-calibrated subset of agents."""
        from itertools import combinations

        agents = list(responses.keys())
        best_subset = set(agents[:self.min_subset_size])
        best_score = float('inf')

        for size in range(self.min_subset_size, min(len(agents), self.max_subset_size) + 1):
            for subset in combinations(agents, size):
                # Score = average Brier score (lower is better)
                subset_scores = [calibration.get(a, 0.5) for a in subset]
                avg_score = np.mean(subset_scores)

                if avg_score < best_score:
                    best_score = avg_score
                    best_subset = set(subset)

        return best_subset, best_score

    def _normalize_distribution(self, dist: List[float], target_len: int) -> np.ndarray:
        """Normalize distribution to target length with valid probabilities."""
        if len(dist) == target_len:
            arr = np.array(dist)
        elif len(dist) == 1:
            # Single confidence value -> binary distribution
            conf = dist[0]
            arr = np.array([conf, 1 - conf] + [0] * (target_len - 2))
        else:
            # Interpolate/pad
            arr = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(dist)),
                dist
            )

        # Ensure valid probability distribution
        arr = np.clip(arr, 1e-10, 1.0)
        return arr / arr.sum()
```

**Integration into consensus_phase.py:**

```python
# In ConsensusPhase._calculate_final_weights()
from aragora.ranking.muse_calibration import MUSECalculator, MUSEResult

class ConsensusPhase:
    def __init__(self, ...):
        ...
        self.muse_calculator = MUSECalculator()
        self.muse_weight = config.get('muse_weight', 0.15)  # New weight factor

    async def _apply_muse_adjustment(
        self,
        votes: List[Vote],
        calibration_engine: CalibrationEngine
    ) -> Tuple[List[Vote], MUSEResult]:
        """Apply MUSE ensemble uncertainty to vote weights."""

        # Collect agent responses
        agent_responses = {
            vote.agent_id: {
                'answer': vote.choice,
                'confidence': vote.confidence,
                'distribution': vote.confidence_distribution or [vote.confidence]
            }
            for vote in votes
        }

        # Get historical calibration
        historical_cal = {
            agent_id: calibration_engine.get_agent_brier_score(agent_id)
            for agent_id in agent_responses.keys()
        }

        # Calculate MUSE
        muse_result = self.muse_calculator.calculate_ensemble_uncertainty(
            agent_responses, historical_cal
        )

        # Adjust weights for agents in best subset
        adjusted_votes = []
        for vote in votes:
            if vote.agent_id in muse_result.best_subset:
                # Boost weight for well-calibrated subset members
                vote.weight *= (1.0 + self.muse_weight * muse_result.subset_agreement)
            adjusted_votes.append(vote)

        return adjusted_votes, muse_result
```

**Tests to add:** `tests/ranking/test_muse_calibration.py`

---

### 1.2 ASCoT Late-Stage Fragility Detection

**Goal:** Apply higher scrutiny to later debate rounds based on ASCoT's finding that errors compound

**Files to modify:**
- `aragora/debate/orchestrator.py` - Add round-aware verification intensity
- `aragora/debate/phases/consensus_phase.py` - Weight later round critiques higher

**Implementation:**

```python
# aragora/debate/ascot_fragility.py (NEW FILE)
"""
ASCoT: Adaptive Self-Correction Chain-of-Thought
Based on: https://arxiv.org/pdf/2508.05282

Key insight: Late-stage errors in reasoning chains are significantly
more impactful than early-stage errors. Aragora debates follow similar
patterns - later rounds build on earlier conclusions.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class FragilityScore:
    """Fragility assessment for a debate round."""
    round_number: int
    base_fragility: float      # Position-based fragility (higher for later rounds)
    dependency_depth: int      # How many prior rounds this depends on
    error_risk: float          # Compound error probability
    recommended_scrutiny: str  # LOW, MEDIUM, HIGH, CRITICAL

class ASCoTFragilityAnalyzer:
    """
    Analyzes debate rounds for late-stage fragility.

    Applies exponential weighting to later rounds:
    fragility(r) = 1 - exp(-lambda * r / total_rounds)

    Where lambda controls the steepness of the fragility curve.
    """

    def __init__(
        self,
        lambda_factor: float = 2.0,  # Steepness of fragility curve
        critical_threshold: float = 0.8,
        high_threshold: float = 0.6,
        medium_threshold: float = 0.3
    ):
        self.lambda_factor = lambda_factor
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def calculate_round_fragility(
        self,
        round_number: int,
        total_rounds: int,
        dependencies: Optional[List[int]] = None
    ) -> FragilityScore:
        """
        Calculate fragility score for a specific debate round.

        Args:
            round_number: Current round (1-indexed)
            total_rounds: Expected total rounds
            dependencies: List of prior round numbers this round depends on

        Returns:
            FragilityScore with recommended scrutiny level
        """
        # Base fragility: exponential increase toward end
        normalized_position = round_number / total_rounds
        base_fragility = 1 - np.exp(-self.lambda_factor * normalized_position)

        # Dependency depth increases risk
        dependency_depth = len(dependencies) if dependencies else round_number - 1

        # Compound error probability
        # P(error) = 1 - (1 - base_error_rate)^dependency_depth
        base_error_rate = 0.05  # Assumed 5% per-step error rate
        error_risk = 1 - (1 - base_error_rate) ** dependency_depth

        # Combined fragility
        combined_fragility = 0.6 * base_fragility + 0.4 * error_risk

        # Determine scrutiny level
        if combined_fragility >= self.critical_threshold:
            scrutiny = "CRITICAL"
        elif combined_fragility >= self.high_threshold:
            scrutiny = "HIGH"
        elif combined_fragility >= self.medium_threshold:
            scrutiny = "MEDIUM"
        else:
            scrutiny = "LOW"

        return FragilityScore(
            round_number=round_number,
            base_fragility=base_fragility,
            dependency_depth=dependency_depth,
            error_risk=error_risk,
            recommended_scrutiny=scrutiny
        )

    def get_verification_intensity(self, fragility: FragilityScore) -> Dict:
        """
        Get verification parameters based on fragility.

        Returns config for consensus verification phase.
        """
        scrutiny_configs = {
            "LOW": {
                "formal_verification": False,
                "evidence_check": False,
                "critique_weight_boost": 1.0,
                "timeout_seconds": 30,
            },
            "MEDIUM": {
                "formal_verification": False,
                "evidence_check": True,
                "critique_weight_boost": 1.2,
                "timeout_seconds": 60,
            },
            "HIGH": {
                "formal_verification": True,
                "evidence_check": True,
                "critique_weight_boost": 1.5,
                "timeout_seconds": 120,
            },
            "CRITICAL": {
                "formal_verification": True,
                "evidence_check": True,
                "critique_weight_boost": 2.0,
                "timeout_seconds": 180,
                "require_multi_agent_agreement": True,
            },
        }
        return scrutiny_configs[fragility.recommended_scrutiny]
```

**Integration into orchestrator.py:**

```python
# In Arena.run_debate_round()
from aragora.debate.ascot_fragility import ASCoTFragilityAnalyzer

class Arena:
    def __init__(self, ...):
        ...
        self.fragility_analyzer = ASCoTFragilityAnalyzer()

    async def run_debate_round(self, round_number: int, ...):
        """Run a single debate round with fragility-aware verification."""

        # Calculate fragility for this round
        fragility = self.fragility_analyzer.calculate_round_fragility(
            round_number=round_number,
            total_rounds=self.config.max_rounds,
            dependencies=self._get_round_dependencies(round_number)
        )

        # Log fragility assessment
        self.event_emitter.emit('round_fragility_assessed', {
            'round': round_number,
            'fragility': fragility.base_fragility,
            'scrutiny': fragility.recommended_scrutiny
        })

        # Get verification intensity
        verification_config = self.fragility_analyzer.get_verification_intensity(fragility)

        # Apply to consensus phase
        self.consensus_phase.set_verification_config(verification_config)

        # ... rest of round execution
```

---

### 1.3 RLM Enhancement: Prime Intellect Integration

**Goal:** Extend existing RLM with insights from Prime Intellect paper for better long-context handling

**Files to modify:**
- `aragora/rlm/repl.py` - Add iterative refinement primitives
- `aragora/rlm/adapter.py` - Enhance context budgeting

**The existing RLM is already well-designed.** Key enhancements:

```python
# aragora/rlm/repl.py - ADD to existing primitives

# New primitives for iterative refinement (Prime Intellect style)
ITERATIVE_PRIMITIVES = {
    'CHECKPOINT': """
def CHECKPOINT(state: dict, label: str = 'default') -> bool:
    '''Save computation state for potential rollback.'''
    _checkpoints[label] = {
        'state': state.copy(),
        'timestamp': time.time(),
        'iteration': _iteration_count
    }
    return True
""",

    'ROLLBACK': """
def ROLLBACK(label: str = 'default') -> dict:
    '''Restore to checkpointed state.'''
    if label not in _checkpoints:
        raise ValueError(f"No checkpoint '{label}' found")
    return _checkpoints[label]['state'].copy()
""",

    'VERIFY_STEP': """
def VERIFY_STEP(claim: str, evidence: list) -> dict:
    '''Request step-level verification before proceeding.'''
    return {
        'claim': claim,
        'evidence': evidence,
        'verified': None,  # To be filled by verifier
        'confidence': None
    }
""",

    'BRANCH': """
def BRANCH(alternatives: list) -> list:
    '''Explore multiple reasoning paths in parallel.'''
    return [{'path': alt, 'result': None} for alt in alternatives]
""",

    'MERGE': """
def MERGE(branches: list, strategy: str = 'best') -> dict:
    '''Merge branch results using specified strategy.'''
    if strategy == 'best':
        return max(branches, key=lambda b: b.get('confidence', 0))
    elif strategy == 'consensus':
        # Voting-based merge
        from collections import Counter
        votes = Counter(b['result'] for b in branches if b['result'])
        return {'result': votes.most_common(1)[0][0], 'agreement': votes.most_common(1)[0][1] / len(branches)}
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
"""
}

# Add to RLMEnvironment.SAFE_BUILTINS
class RLMEnvironment:
    def __init__(self, ...):
        ...
        # Add iterative refinement state
        self._checkpoints: Dict[str, dict] = {}
        self._iteration_count: int = 0
        self._branches: List[dict] = []
```

---

## Phase 2: Process Verification (Weeks 5-8)

### 2.1 ThinkPRM Step-by-Step Verification

**Goal:** Add process reward model for step-by-step debate verification

**Files to create:**
- `aragora/verification/think_prm.py` - Process reward model integration
- `aragora/verification/step_verifier.py` - Step-level verification

**Implementation:**

```python
# aragora/verification/think_prm.py (NEW FILE)
"""
ThinkPRM: Process Reward Models That Think
Based on: https://arxiv.org/abs/2504.16828

Provides verbalized step-wise verification with minimal labels.
Integrates with Aragora's debate rounds as "steps" to verify.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import asyncio

class StepVerdict(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NEEDS_REVISION = "needs_revision"

@dataclass
class StepVerification:
    """Verification result for a single debate step."""
    step_id: str
    round_number: int
    agent_id: str
    content_summary: str
    verdict: StepVerdict
    confidence: float
    reasoning: str
    suggested_fix: Optional[str] = None
    dependencies_verified: bool = True

@dataclass
class ProcessVerificationResult:
    """Full process verification across all debate rounds."""
    debate_id: str
    total_steps: int
    correct_steps: int
    incorrect_steps: int
    uncertain_steps: int
    overall_score: float  # 0.0-1.0
    critical_errors: List[StepVerification] = field(default_factory=list)
    step_results: List[StepVerification] = field(default_factory=list)

class ThinkPRMVerifier:
    """
    Process Reward Model for debate step verification.

    Uses verbalized reasoning to verify each debate round:
    1. Extract claim from round
    2. Check logical consistency with prior rounds
    3. Verify evidence citations
    4. Assess reasoning validity
    """

    VERIFICATION_PROMPT = '''You are a rigorous debate step verifier. Analyze the following debate step and determine if the reasoning is valid.

PRIOR CONTEXT (what came before):
{prior_context}

CURRENT STEP TO VERIFY:
Round {round_number} by {agent_id}:
{step_content}

CLAIMED DEPENDENCIES:
{dependencies}

VERIFICATION TASKS:
1. Is the logical reasoning valid?
2. Are the claimed dependencies actually used correctly?
3. Are there any factual errors or unsupported claims?
4. Does this step build correctly on prior context?

Respond in this exact format:
VERDICT: [CORRECT|INCORRECT|UNCERTAIN|NEEDS_REVISION]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed analysis]
SUGGESTED_FIX: [If INCORRECT or NEEDS_REVISION, what should change]'''

    def __init__(self, verifier_agent_id: str = "claude"):
        self.verifier_agent_id = verifier_agent_id
        self._cache: Dict[str, StepVerification] = {}

    async def verify_step(
        self,
        step_content: str,
        round_number: int,
        agent_id: str,
        prior_context: str,
        dependencies: List[str],
        agent_pool: 'AgentPool'
    ) -> StepVerification:
        """Verify a single debate step using verbalized reasoning."""

        # Format verification prompt
        prompt = self.VERIFICATION_PROMPT.format(
            prior_context=prior_context[:2000],  # Truncate for context window
            round_number=round_number,
            agent_id=agent_id,
            step_content=step_content,
            dependencies="\n".join(f"- {d}" for d in dependencies) or "None claimed"
        )

        # Get verification from verifier agent
        response = await agent_pool.query(
            agent_id=self.verifier_agent_id,
            prompt=prompt,
            max_tokens=1000
        )

        # Parse response
        return self._parse_verification_response(
            response=response,
            step_content=step_content,
            round_number=round_number,
            agent_id=agent_id
        )

    async def verify_debate_process(
        self,
        debate_rounds: List[Dict],
        agent_pool: 'AgentPool'
    ) -> ProcessVerificationResult:
        """
        Verify entire debate as a sequence of steps.

        Applies fragility-aware verification: later rounds get more scrutiny.
        """
        step_results = []
        prior_context = ""

        for i, round_data in enumerate(debate_rounds):
            round_number = i + 1

            # Verify each agent's contribution in this round
            for contribution in round_data.get('contributions', []):
                verification = await self.verify_step(
                    step_content=contribution['content'],
                    round_number=round_number,
                    agent_id=contribution['agent_id'],
                    prior_context=prior_context,
                    dependencies=contribution.get('dependencies', []),
                    agent_pool=agent_pool
                )
                step_results.append(verification)

            # Update prior context
            prior_context += f"\n\nRound {round_number}:\n"
            prior_context += "\n".join(
                c['content'][:500] for c in round_data.get('contributions', [])
            )

        # Calculate overall score
        correct = sum(1 for s in step_results if s.verdict == StepVerdict.CORRECT)
        incorrect = sum(1 for s in step_results if s.verdict == StepVerdict.INCORRECT)
        uncertain = sum(1 for s in step_results if s.verdict == StepVerdict.UNCERTAIN)
        total = len(step_results)

        overall_score = correct / total if total > 0 else 0.0

        # Identify critical errors (late-stage incorrect steps)
        critical_threshold = len(debate_rounds) * 0.7  # Last 30% of debate
        critical_errors = [
            s for s in step_results
            if s.verdict == StepVerdict.INCORRECT and s.round_number >= critical_threshold
        ]

        return ProcessVerificationResult(
            debate_id=debate_rounds[0].get('debate_id', 'unknown'),
            total_steps=total,
            correct_steps=correct,
            incorrect_steps=incorrect,
            uncertain_steps=uncertain,
            overall_score=overall_score,
            critical_errors=critical_errors,
            step_results=step_results
        )

    def _parse_verification_response(
        self,
        response: str,
        step_content: str,
        round_number: int,
        agent_id: str
    ) -> StepVerification:
        """Parse LLM verification response into structured result."""
        lines = response.strip().split('\n')

        verdict = StepVerdict.UNCERTAIN
        confidence = 0.5
        reasoning = ""
        suggested_fix = None

        for line in lines:
            if line.startswith('VERDICT:'):
                verdict_str = line.split(':', 1)[1].strip().upper()
                verdict = StepVerdict(verdict_str.lower()) if verdict_str.lower() in [v.value for v in StepVerdict] else StepVerdict.UNCERTAIN
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
            elif line.startswith('SUGGESTED_FIX:'):
                suggested_fix = line.split(':', 1)[1].strip()
                if suggested_fix.lower() == 'none':
                    suggested_fix = None

        return StepVerification(
            step_id=f"step_{round_number}_{agent_id}",
            round_number=round_number,
            agent_id=agent_id,
            content_summary=step_content[:200],
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            suggested_fix=suggested_fix
        )
```

**Integration into debate orchestrator:**

```python
# In Arena.run_debate() - add process verification
from aragora.verification.think_prm import ThinkPRMVerifier

class Arena:
    def __init__(self, ...):
        ...
        self.prm_verifier = ThinkPRMVerifier()
        self.enable_process_verification = config.get('process_verification', True)

    async def run_debate(self, ...):
        # ... existing debate logic ...

        # After all rounds complete, run process verification
        if self.enable_process_verification:
            prm_result = await self.prm_verifier.verify_debate_process(
                debate_rounds=self.debate_history,
                agent_pool=self.agent_pool
            )

            # Log process verification results
            self.event_emitter.emit('process_verification_complete', {
                'debate_id': self.debate_id,
                'overall_score': prm_result.overall_score,
                'critical_errors': len(prm_result.critical_errors)
            })

            # If critical errors found, trigger revision phase
            if prm_result.critical_errors:
                await self._trigger_targeted_revision(prm_result.critical_errors)
```

---

### 2.2 Hilbert-Style Recursive Formal Proofs

**Goal:** Extend existing Lean4/Z3 verification with recursive informal→formal proof generation

**Files to modify:**
- `aragora/verification/formal_prover.py` - Add recursive proof generation
- `aragora/debate/phases/consensus_verification.py` - Integrate Hilbert approach

**Implementation:**

```python
# aragora/verification/hilbert_prover.py (NEW FILE)
"""
Hilbert: Recursive Formal Proofs with Informal Reasoning
Based on: https://machinelearning.apple.com/research/hilbert

Key insight: Decompose complex claims into sub-claims, prove each formally,
then compose into complete proof. Aligns with RLM's hierarchical approach.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import asyncio

class ProofStatus(Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    TIMEOUT = "timeout"
    DECOMPOSED = "decomposed"
    PENDING = "pending"

@dataclass
class ProofNode:
    """Node in the recursive proof tree."""
    claim_id: str
    claim_text: str
    formal_statement: Optional[str] = None
    proof_script: Optional[str] = None
    status: ProofStatus = ProofStatus.PENDING
    confidence: float = 0.0
    children: List['ProofNode'] = field(default_factory=list)
    parent_id: Optional[str] = None

@dataclass
class HilbertProofResult:
    """Result of Hilbert recursive proof attempt."""
    root_claim: str
    status: ProofStatus
    proof_tree: ProofNode
    total_subclaims: int
    proven_subclaims: int
    formal_proof: Optional[str] = None
    verification_time_ms: float = 0.0

class HilbertProver:
    """
    Recursive formal prover using Hilbert's decomposition approach.

    Process:
    1. Decompose complex claim into atomic sub-claims
    2. Translate each sub-claim to formal language (Lean4)
    3. Attempt proof for each sub-claim
    4. Compose proofs bottom-up
    5. If sub-proof fails, try alternative decomposition
    """

    DECOMPOSITION_PROMPT = '''You are a formal reasoning assistant. Decompose this claim into simpler sub-claims that, if all proven true, would prove the original claim.

CLAIM TO DECOMPOSE:
{claim}

CONTEXT:
{context}

Output a list of sub-claims, each on its own line, prefixed with "SUB: ".
If the claim is already atomic (cannot be decomposed), output "ATOMIC: {claim}".
Keep sub-claims precise and verifiable.'''

    FORMALIZATION_PROMPT = '''Translate this informal claim into a Lean4 theorem statement.

INFORMAL CLAIM:
{claim}

CONTEXT (definitions available):
{context}

Output ONLY the Lean4 theorem statement, starting with "theorem".
If the claim cannot be formalized, output "UNFORMALIZABLE: {reason}".'''

    def __init__(
        self,
        max_depth: int = 5,
        timeout_per_proof_ms: int = 30000,
        lean4_path: str = "/usr/bin/lean"
    ):
        self.max_depth = max_depth
        self.timeout_per_proof_ms = timeout_per_proof_ms
        self.lean4_path = lean4_path

    async def prove_claim(
        self,
        claim: str,
        context: str,
        agent_pool: 'AgentPool',
        depth: int = 0
    ) -> HilbertProofResult:
        """
        Recursively prove a claim using decomposition.
        """
        import time
        start_time = time.time()

        # Create root node
        root = ProofNode(
            claim_id=f"claim_{hash(claim) % 10000}",
            claim_text=claim
        )

        # Attempt recursive proof
        await self._prove_node(root, context, agent_pool, depth)

        # Count results
        total, proven = self._count_subclaims(root)

        # Compose final proof if all subclaims proven
        formal_proof = None
        if root.status == ProofStatus.PROVEN:
            formal_proof = self._compose_proof(root)

        return HilbertProofResult(
            root_claim=claim,
            status=root.status,
            proof_tree=root,
            total_subclaims=total,
            proven_subclaims=proven,
            formal_proof=formal_proof,
            verification_time_ms=(time.time() - start_time) * 1000
        )

    async def _prove_node(
        self,
        node: ProofNode,
        context: str,
        agent_pool: 'AgentPool',
        depth: int
    ) -> None:
        """Recursively prove a single node."""

        if depth >= self.max_depth:
            node.status = ProofStatus.TIMEOUT
            return

        # Try to formalize directly
        formal = await self._formalize_claim(node.claim_text, context, agent_pool)

        if formal and not formal.startswith("UNFORMALIZABLE"):
            node.formal_statement = formal

            # Attempt direct proof
            proven = await self._attempt_lean4_proof(formal)

            if proven:
                node.status = ProofStatus.PROVEN
                node.confidence = 1.0
                return

        # If direct proof fails, decompose
        subclaims = await self._decompose_claim(node.claim_text, context, agent_pool)

        if not subclaims or subclaims[0].startswith("ATOMIC"):
            # Cannot decompose further, mark as unproven
            node.status = ProofStatus.DISPROVEN
            node.confidence = 0.0
            return

        # Recursively prove subclaims
        node.status = ProofStatus.DECOMPOSED
        for i, subclaim in enumerate(subclaims):
            child = ProofNode(
                claim_id=f"{node.claim_id}_sub{i}",
                claim_text=subclaim,
                parent_id=node.claim_id
            )
            node.children.append(child)
            await self._prove_node(child, context, agent_pool, depth + 1)

        # Check if all children proven
        if all(c.status == ProofStatus.PROVEN for c in node.children):
            node.status = ProofStatus.PROVEN
            node.confidence = min(c.confidence for c in node.children)
        elif any(c.status == ProofStatus.DISPROVEN for c in node.children):
            node.status = ProofStatus.DISPROVEN
            node.confidence = 0.0
        else:
            node.confidence = sum(c.confidence for c in node.children) / len(node.children)

    async def _decompose_claim(
        self,
        claim: str,
        context: str,
        agent_pool: 'AgentPool'
    ) -> List[str]:
        """Decompose a claim into sub-claims."""
        prompt = self.DECOMPOSITION_PROMPT.format(claim=claim, context=context[:1000])

        response = await agent_pool.query(
            agent_id="claude",  # Use strong reasoner for decomposition
            prompt=prompt,
            max_tokens=500
        )

        # Parse sub-claims
        subclaims = []
        for line in response.strip().split('\n'):
            if line.startswith('SUB: '):
                subclaims.append(line[5:].strip())
            elif line.startswith('ATOMIC: '):
                return [line]  # Signal that claim is atomic

        return subclaims

    async def _formalize_claim(
        self,
        claim: str,
        context: str,
        agent_pool: 'AgentPool'
    ) -> Optional[str]:
        """Translate informal claim to Lean4."""
        prompt = self.FORMALIZATION_PROMPT.format(claim=claim, context=context[:500])

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=300
        )

        if response.strip().startswith("theorem") or response.strip().startswith("UNFORMALIZABLE"):
            return response.strip()
        return None

    async def _attempt_lean4_proof(self, formal_statement: str) -> bool:
        """Attempt to prove the formal statement with Lean4."""
        # Integration with existing Lean4 verification
        # This would call the existing lean4 subprocess
        # Simplified for illustration
        try:
            from aragora.verification.lean4_runner import Lean4Runner
            runner = Lean4Runner(timeout_ms=self.timeout_per_proof_ms)
            result = await runner.verify(formal_statement)
            return result.verified
        except Exception:
            return False

    def _count_subclaims(self, node: ProofNode) -> Tuple[int, int]:
        """Count total and proven subclaims."""
        total = 1
        proven = 1 if node.status == ProofStatus.PROVEN else 0

        for child in node.children:
            child_total, child_proven = self._count_subclaims(child)
            total += child_total
            proven += child_proven

        return total, proven

    def _compose_proof(self, node: ProofNode) -> str:
        """Compose formal proof from proven subclaims."""
        if not node.children:
            return node.proof_script or f"-- Direct proof of: {node.claim_text}"

        child_proofs = [self._compose_proof(c) for c in node.children]
        return f"""
-- Proof of: {node.claim_text}
-- Decomposed into {len(node.children)} subclaims

{chr(10).join(child_proofs)}

-- Composition: All subclaims proven, therefore main claim holds.
"""
```

---

## Phase 3: Knowledge & Evidence (Weeks 9-12)

### 3.1 GraphRAG Enhancement for Knowledge Mound

**Goal:** Add graph traversal to existing vector search for hybrid retrieval

**Files to modify:**
- `aragora/knowledge/mound/api/query.py` - Add graph traversal
- `aragora/knowledge/mound/ops/graph_rag.py` - New GraphRAG operations

**Implementation:**

```python
# aragora/knowledge/mound/ops/graph_rag.py (NEW FILE)
"""
GraphRAG: Hybrid Graph + Vector Retrieval
Based on: https://pub.towardsai.net/graphrag-explained

Enhances existing Knowledge Mound with:
1. Relationship-aware retrieval (follows supports/contradicts edges)
2. Community detection for topic clustering
3. Multi-hop reasoning paths
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import heapq

@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    node_type: str = "knowledge"
    confidence: float = 1.0

@dataclass
class GraphEdge:
    """Edge connecting two knowledge nodes."""
    source_id: str
    target_id: str
    relation_type: str  # SUPPORTS, CONTRADICTS, DERIVED_FROM, CLARIFIES
    weight: float = 1.0

@dataclass
class RetrievalPath:
    """A retrieval path through the knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    path_score: float
    reasoning: str

@dataclass
class GraphRAGResult:
    """Result of GraphRAG hybrid retrieval."""
    query: str
    direct_matches: List[GraphNode]      # Vector similarity matches
    graph_expanded: List[GraphNode]       # Nodes reached via graph traversal
    reasoning_paths: List[RetrievalPath]  # Multi-hop reasoning paths
    communities: List[Set[str]]           # Topic communities
    combined_score: float

class GraphRAGRetriever:
    """
    Hybrid retrieval combining vector similarity with graph traversal.

    Strategy:
    1. Vector search for initial seed nodes
    2. Expand via relationship edges (supports → include, contradicts → flag)
    3. Community detection for topic coherence
    4. Path scoring for reasoning chains
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        max_hops: int = 3,
        top_k_seeds: int = 10,
        top_k_expanded: int = 25
    ):
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.max_hops = max_hops
        self.top_k_seeds = top_k_seeds
        self.top_k_expanded = top_k_expanded

    async def retrieve(
        self,
        query: str,
        knowledge_mound: 'KnowledgeMound',
        include_contradictions: bool = True
    ) -> GraphRAGResult:
        """
        Perform hybrid GraphRAG retrieval.
        """
        # Step 1: Vector search for seed nodes
        vector_results = await knowledge_mound.semantic_search(
            query=query,
            top_k=self.top_k_seeds
        )

        seed_nodes = [
            GraphNode(
                id=r.id,
                content=r.content,
                embedding=r.embedding,
                confidence=r.confidence
            )
            for r in vector_results
        ]

        # Step 2: Graph expansion from seeds
        expanded_nodes, edges = await self._expand_graph(
            seeds=[n.id for n in seed_nodes],
            knowledge_mound=knowledge_mound,
            include_contradictions=include_contradictions
        )

        # Step 3: Find reasoning paths
        reasoning_paths = self._find_reasoning_paths(
            seed_ids=[n.id for n in seed_nodes],
            nodes={n.id: n for n in seed_nodes + expanded_nodes},
            edges=edges
        )

        # Step 4: Community detection
        communities = self._detect_communities(
            nodes=seed_nodes + expanded_nodes,
            edges=edges
        )

        # Step 5: Score and rank
        combined_score = self._calculate_combined_score(
            seed_nodes, expanded_nodes, reasoning_paths
        )

        return GraphRAGResult(
            query=query,
            direct_matches=seed_nodes,
            graph_expanded=expanded_nodes,
            reasoning_paths=reasoning_paths,
            communities=communities,
            combined_score=combined_score
        )

    async def _expand_graph(
        self,
        seeds: List[str],
        knowledge_mound: 'KnowledgeMound',
        include_contradictions: bool
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Expand from seed nodes via relationship edges."""
        visited: Set[str] = set(seeds)
        expanded: List[GraphNode] = []
        edges: List[GraphEdge] = []
        frontier = list(seeds)

        for hop in range(self.max_hops):
            if not frontier:
                break

            next_frontier = []

            for node_id in frontier:
                # Get relationships from Knowledge Mound
                relationships = await knowledge_mound.get_relationships(node_id)

                for rel in relationships:
                    target_id = rel['target_id']
                    relation_type = rel['relation_type']

                    # Skip contradictions if not requested
                    if relation_type == 'CONTRADICTS' and not include_contradictions:
                        continue

                    # Create edge
                    edge = GraphEdge(
                        source_id=node_id,
                        target_id=target_id,
                        relation_type=relation_type,
                        weight=rel.get('weight', 1.0)
                    )
                    edges.append(edge)

                    if target_id not in visited:
                        visited.add(target_id)
                        next_frontier.append(target_id)

                        # Fetch node content
                        node_data = await knowledge_mound.get(target_id)
                        if node_data:
                            expanded.append(GraphNode(
                                id=target_id,
                                content=node_data.content,
                                confidence=node_data.confidence
                            ))

            frontier = next_frontier[:self.top_k_expanded - len(expanded)]

        return expanded, edges

    def _find_reasoning_paths(
        self,
        seed_ids: List[str],
        nodes: Dict[str, GraphNode],
        edges: List[GraphEdge]
    ) -> List[RetrievalPath]:
        """Find multi-hop reasoning paths between seeds."""
        # Build adjacency list
        adj: Dict[str, List[Tuple[str, GraphEdge]]] = defaultdict(list)
        for edge in edges:
            adj[edge.source_id].append((edge.target_id, edge))

        paths = []

        # Find paths between pairs of seed nodes
        for i, start in enumerate(seed_ids):
            for end in seed_ids[i+1:]:
                path = self._bfs_path(start, end, adj, nodes)
                if path:
                    paths.append(path)

        # Sort by path score
        paths.sort(key=lambda p: p.path_score, reverse=True)
        return paths[:10]  # Top 10 paths

    def _bfs_path(
        self,
        start: str,
        end: str,
        adj: Dict[str, List[Tuple[str, GraphEdge]]],
        nodes: Dict[str, GraphNode]
    ) -> Optional[RetrievalPath]:
        """BFS to find path between two nodes."""
        if start == end:
            return None

        queue = [(start, [start], [], 0.0)]  # (current, path_nodes, path_edges, score)
        visited = {start}

        while queue:
            current, path_nodes, path_edges, score = queue.pop(0)

            for neighbor, edge in adj.get(current, []):
                if neighbor == end:
                    # Found path
                    final_nodes = [nodes[n] for n in path_nodes + [neighbor] if n in nodes]
                    final_edges = path_edges + [edge]

                    # Calculate path score
                    edge_score = sum(e.weight for e in final_edges) / len(final_edges)
                    node_score = sum(n.confidence for n in final_nodes) / len(final_nodes)
                    path_score = 0.5 * edge_score + 0.5 * node_score

                    # Generate reasoning
                    reasoning = self._generate_path_reasoning(final_nodes, final_edges)

                    return RetrievalPath(
                        nodes=final_nodes,
                        edges=final_edges,
                        path_score=path_score,
                        reasoning=reasoning
                    )

                if neighbor not in visited and len(path_nodes) < self.max_hops:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge],
                        score + edge.weight
                    ))

        return None

    def _generate_path_reasoning(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> str:
        """Generate human-readable reasoning for a path."""
        if len(nodes) < 2:
            return ""

        parts = [f"Starting from: {nodes[0].content[:100]}..."]

        for i, edge in enumerate(edges):
            relation = edge.relation_type.lower().replace('_', ' ')
            target = nodes[i + 1].content[:100] if i + 1 < len(nodes) else "..."
            parts.append(f"  → {relation} → {target}...")

        return "\n".join(parts)

    def _detect_communities(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[Set[str]]:
        """Simple community detection via connected components."""
        # Union-Find for connected components
        parent = {n.id: n.id for n in nodes}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for edge in edges:
            if edge.source_id in parent and edge.target_id in parent:
                union(edge.source_id, edge.target_id)

        # Group by root
        communities: Dict[str, Set[str]] = defaultdict(set)
        for node in nodes:
            root = find(node.id)
            communities[root].add(node.id)

        return list(communities.values())

    def _calculate_combined_score(
        self,
        seeds: List[GraphNode],
        expanded: List[GraphNode],
        paths: List[RetrievalPath]
    ) -> float:
        """Calculate overall retrieval quality score."""
        if not seeds:
            return 0.0

        seed_score = sum(n.confidence for n in seeds) / len(seeds)
        expanded_score = sum(n.confidence for n in expanded) / len(expanded) if expanded else 0.0
        path_score = sum(p.path_score for p in paths) / len(paths) if paths else 0.0

        return (
            self.vector_weight * seed_score +
            self.graph_weight * 0.5 * expanded_score +
            self.graph_weight * 0.5 * path_score
        )
```

**Integration into QueryOperationsMixin:**

```python
# In aragora/knowledge/mound/api/query.py
from aragora.knowledge.mound.ops.graph_rag import GraphRAGRetriever, GraphRAGResult

class QueryOperationsMixin:
    def __init__(self, ...):
        ...
        self.graph_rag = GraphRAGRetriever()

    async def hybrid_retrieve(
        self,
        query: str,
        include_contradictions: bool = True
    ) -> GraphRAGResult:
        """
        Hybrid GraphRAG retrieval combining vector + graph.

        Preferred over semantic_search() for complex queries
        that benefit from relationship traversal.
        """
        return await self.graph_rag.retrieve(
            query=query,
            knowledge_mound=self,
            include_contradictions=include_contradictions
        )
```

---

### 3.2 ClaimCheck Stepwise Fact Verification

**Goal:** Replace keyword-based evidence grounding with atomic claim decomposition

**Files to modify:**
- `aragora/evidence/evidence_grounding.py` - Upgrade matching algorithm
- `aragora/evidence/claim_decomposer.py` - New atomic decomposition

**Implementation:**

```python
# aragora/evidence/claim_decomposer.py (NEW FILE)
"""
ClaimCheck: Stepwise Claim Verification
Based on: https://arxiv.org/abs/2510.01226

Decomposes complex claims into atomic sub-claims for finer-grained
verification against evidence.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

class ClaimType(Enum):
    FACTUAL = "factual"           # Verifiable fact
    CAUSAL = "causal"             # X causes Y
    COMPARATIVE = "comparative"    # X > Y
    DEFINITIONAL = "definitional"  # X is defined as Y
    TEMPORAL = "temporal"          # X happened at time T
    QUANTITATIVE = "quantitative"  # X has value N
    COMPOSITE = "composite"        # Multiple sub-claims

@dataclass
class AtomicClaim:
    """An atomic, indivisible claim."""
    id: str
    text: str
    claim_type: ClaimType
    entities: List[str]
    relations: List[str]
    parent_claim_id: Optional[str] = None

@dataclass
class ClaimVerification:
    """Verification result for an atomic claim."""
    claim: AtomicClaim
    verified: bool
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_method: str  # "exact_match", "semantic", "inference"

@dataclass
class ClaimCheckResult:
    """Full ClaimCheck verification result."""
    original_claim: str
    atomic_claims: List[AtomicClaim]
    verifications: List[ClaimVerification]
    overall_verified: bool
    overall_confidence: float
    unverified_claims: List[AtomicClaim]

class ClaimDecomposer:
    """
    Decomposes complex claims into atomic sub-claims.

    Uses pattern matching and LLM assistance to identify:
    - Conjunctions (X and Y → two claims)
    - Causal chains (X causes Y which causes Z → two claims)
    - Comparisons (X > Y → one claim per entity)
    - Temporal sequences (first X, then Y → ordered claims)
    """

    DECOMPOSITION_PROMPT = '''Decompose this claim into atomic sub-claims that can each be independently verified.

CLAIM: {claim}

For each atomic claim, identify:
1. The specific factual assertion
2. Key entities involved
3. The type (factual, causal, comparative, temporal, quantitative)

Output format (one per line):
ATOMIC: [claim text] | ENTITIES: [entity1, entity2] | TYPE: [type] | RELATIONS: [relation1]

If the claim is already atomic, output:
ATOMIC: {claim} | ENTITIES: [...] | TYPE: [...] | RELATIONS: [...]'''

    def __init__(self):
        self._claim_counter = 0

    async def decompose(
        self,
        claim: str,
        agent_pool: Optional['AgentPool'] = None
    ) -> List[AtomicClaim]:
        """Decompose a claim into atomic sub-claims."""

        # Try rule-based decomposition first
        rule_based = self._rule_based_decompose(claim)
        if rule_based and len(rule_based) > 1:
            return rule_based

        # Fall back to LLM decomposition
        if agent_pool:
            return await self._llm_decompose(claim, agent_pool)

        # If no agent pool, return as single atomic claim
        return [self._create_atomic_claim(claim, ClaimType.FACTUAL)]

    def _rule_based_decompose(self, claim: str) -> List[AtomicClaim]:
        """Rule-based decomposition for common patterns."""
        atomic_claims = []

        # Conjunction pattern: "X and Y"
        if ' and ' in claim.lower() and ',' not in claim:
            parts = claim.lower().split(' and ')
            if len(parts) == 2 and len(parts[0]) > 10 and len(parts[1]) > 10:
                for part in parts:
                    atomic_claims.append(
                        self._create_atomic_claim(part.strip(), ClaimType.FACTUAL)
                    )
                return atomic_claims

        # Causal pattern: "X because Y" or "X causes Y"
        causal_markers = [' because ', ' causes ', ' leads to ', ' results in ']
        for marker in causal_markers:
            if marker in claim.lower():
                parts = claim.lower().split(marker)
                if len(parts) == 2:
                    atomic_claims.append(
                        self._create_atomic_claim(parts[0].strip(), ClaimType.FACTUAL)
                    )
                    atomic_claims.append(
                        self._create_atomic_claim(
                            f"{parts[0].strip()} {marker.strip()} {parts[1].strip()}",
                            ClaimType.CAUSAL
                        )
                    )
                    return atomic_claims

        # Comparative pattern: "X is better/larger/faster than Y"
        comparative_markers = [' better than ', ' larger than ', ' faster than ',
                             ' more than ', ' less than ', ' greater than ']
        for marker in comparative_markers:
            if marker in claim.lower():
                atomic_claims.append(
                    self._create_atomic_claim(claim, ClaimType.COMPARATIVE)
                )
                return atomic_claims

        return []

    async def _llm_decompose(
        self,
        claim: str,
        agent_pool: 'AgentPool'
    ) -> List[AtomicClaim]:
        """LLM-based decomposition for complex claims."""
        prompt = self.DECOMPOSITION_PROMPT.format(claim=claim)

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=500
        )

        return self._parse_decomposition_response(response, claim)

    def _parse_decomposition_response(
        self,
        response: str,
        original_claim: str
    ) -> List[AtomicClaim]:
        """Parse LLM decomposition response."""
        atomic_claims = []

        for line in response.strip().split('\n'):
            if not line.startswith('ATOMIC:'):
                continue

            try:
                # Parse: ATOMIC: [text] | ENTITIES: [...] | TYPE: [...] | RELATIONS: [...]
                parts = line.split('|')

                text = parts[0].replace('ATOMIC:', '').strip()

                entities = []
                relations = []
                claim_type = ClaimType.FACTUAL

                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith('ENTITIES:'):
                        entities_str = part.replace('ENTITIES:', '').strip()
                        entities = [e.strip() for e in entities_str.strip('[]').split(',')]
                    elif part.startswith('TYPE:'):
                        type_str = part.replace('TYPE:', '').strip().lower()
                        try:
                            claim_type = ClaimType(type_str)
                        except ValueError:
                            claim_type = ClaimType.FACTUAL
                    elif part.startswith('RELATIONS:'):
                        relations_str = part.replace('RELATIONS:', '').strip()
                        relations = [r.strip() for r in relations_str.strip('[]').split(',')]

                atomic_claims.append(AtomicClaim(
                    id=f"claim_{self._claim_counter}",
                    text=text,
                    claim_type=claim_type,
                    entities=entities,
                    relations=relations
                ))
                self._claim_counter += 1

            except Exception:
                continue

        # Fallback if parsing failed
        if not atomic_claims:
            atomic_claims.append(self._create_atomic_claim(original_claim, ClaimType.FACTUAL))

        return atomic_claims

    def _create_atomic_claim(self, text: str, claim_type: ClaimType) -> AtomicClaim:
        """Create an atomic claim with auto-generated ID."""
        self._claim_counter += 1
        return AtomicClaim(
            id=f"claim_{self._claim_counter}",
            text=text,
            claim_type=claim_type,
            entities=[],
            relations=[]
        )


class ClaimChecker:
    """
    Verifies atomic claims against evidence.

    Uses multiple verification strategies:
    1. Exact match: Direct text matching
    2. Semantic: Embedding similarity
    3. Inference: LLM-based entailment checking
    """

    def __init__(
        self,
        exact_match_threshold: float = 0.9,
        semantic_threshold: float = 0.75,
        inference_threshold: float = 0.7
    ):
        self.exact_match_threshold = exact_match_threshold
        self.semantic_threshold = semantic_threshold
        self.inference_threshold = inference_threshold
        self.decomposer = ClaimDecomposer()

    async def verify_claim(
        self,
        claim: str,
        evidence_store: 'EvidenceStore',
        agent_pool: Optional['AgentPool'] = None
    ) -> ClaimCheckResult:
        """
        Full ClaimCheck verification pipeline.
        """
        # Step 1: Decompose into atomic claims
        atomic_claims = await self.decomposer.decompose(claim, agent_pool)

        # Step 2: Verify each atomic claim
        verifications = []
        for atomic in atomic_claims:
            verification = await self._verify_atomic_claim(
                atomic, evidence_store, agent_pool
            )
            verifications.append(verification)

        # Step 3: Aggregate results
        verified_count = sum(1 for v in verifications if v.verified)
        overall_verified = verified_count == len(verifications)
        overall_confidence = sum(v.confidence for v in verifications) / len(verifications)

        unverified = [v.claim for v in verifications if not v.verified]

        return ClaimCheckResult(
            original_claim=claim,
            atomic_claims=atomic_claims,
            verifications=verifications,
            overall_verified=overall_verified,
            overall_confidence=overall_confidence,
            unverified_claims=unverified
        )

    async def _verify_atomic_claim(
        self,
        claim: AtomicClaim,
        evidence_store: 'EvidenceStore',
        agent_pool: Optional['AgentPool']
    ) -> ClaimVerification:
        """Verify a single atomic claim."""

        supporting = []
        contradicting = []

        # Strategy 1: Exact/fuzzy text match
        exact_matches = await evidence_store.search(
            query=claim.text,
            method="keyword",
            top_k=5
        )

        for match in exact_matches:
            if self._text_similarity(claim.text, match.snippet) >= self.exact_match_threshold:
                supporting.append(match.id)

        # Strategy 2: Semantic similarity
        semantic_matches = await evidence_store.search(
            query=claim.text,
            method="semantic",
            top_k=10
        )

        for match in semantic_matches:
            if match.combined_score >= self.semantic_threshold:
                if match.id not in supporting:
                    supporting.append(match.id)

        # Strategy 3: LLM inference (if agent pool available and still uncertain)
        if agent_pool and not supporting:
            inference_result = await self._llm_inference_check(
                claim, semantic_matches[:5], agent_pool
            )
            if inference_result['supports']:
                supporting.extend(inference_result['evidence_ids'])
            if inference_result['contradicts']:
                contradicting.extend(inference_result['evidence_ids'])

        # Determine verification result
        verified = len(supporting) > 0 and len(contradicting) == 0
        confidence = self._calculate_confidence(supporting, contradicting, semantic_matches)
        method = "exact_match" if exact_matches else ("semantic" if semantic_matches else "inference")

        return ClaimVerification(
            claim=claim,
            verified=verified,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            verification_method=method
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified Jaccard)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union

    async def _llm_inference_check(
        self,
        claim: AtomicClaim,
        evidence: List['EvidenceSnippet'],
        agent_pool: 'AgentPool'
    ) -> Dict:
        """Use LLM to check if evidence supports/contradicts claim."""
        evidence_text = "\n".join([
            f"[{i+1}] {e.snippet[:300]}..." for i, e in enumerate(evidence)
        ])

        prompt = f'''Does the following evidence support or contradict this claim?

CLAIM: {claim.text}

EVIDENCE:
{evidence_text}

Respond with:
VERDICT: SUPPORTS | CONTRADICTS | NEUTRAL
EVIDENCE_IDS: [list of evidence numbers that are relevant]
CONFIDENCE: [0.0-1.0]'''

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=200
        )

        # Parse response
        supports = 'SUPPORTS' in response.upper()
        contradicts = 'CONTRADICTS' in response.upper()

        # Extract evidence IDs
        evidence_ids = []
        if 'EVIDENCE_IDS:' in response:
            ids_part = response.split('EVIDENCE_IDS:')[1].split('\n')[0]
            for i, e in enumerate(evidence):
                if str(i+1) in ids_part:
                    evidence_ids.append(e.id)

        return {
            'supports': supports,
            'contradicts': contradicts,
            'evidence_ids': evidence_ids
        }

    def _calculate_confidence(
        self,
        supporting: List[str],
        contradicting: List[str],
        all_matches: List
    ) -> float:
        """Calculate verification confidence."""
        if not all_matches:
            return 0.0

        support_score = len(supporting) / max(len(all_matches), 1)
        contradict_penalty = len(contradicting) * 0.2

        return max(0.0, min(1.0, support_score - contradict_penalty))
```

---

## Phase 4: Team Selection & Self-Improvement (Weeks 13-16)

### 4.1 A-HMAD Dynamic Role Specialization

**Goal:** Replace static DOMAIN_CAPABILITY_MAP with learned dynamic specialization

**Files to modify:**
- `aragora/debate/team_selector.py` - Add dynamic role assignment
- `aragora/debate/role_specializer.py` - New dynamic specialization

```python
# aragora/debate/role_specializer.py (NEW FILE)
"""
A-HMAD: Adaptive Heterogeneous Multi-Agent Debate
Based on: https://link.springer.com/article/10.1007/s44443-025-00353-3

Dynamically assigns expertise roles based on debate topic analysis.
Achieves 4-6% accuracy gains over static assignment.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum

class ExpertiseRole(Enum):
    DOMAIN_EXPERT = "domain_expert"
    METHODOLOGIST = "methodologist"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    FACT_CHECKER = "fact_checker"
    DEVIL_ADVOCATE = "devil_advocate"
    MEDIATOR = "mediator"

@dataclass
class RoleAssignment:
    """Assignment of an agent to a role."""
    agent_id: str
    role: ExpertiseRole
    confidence: float
    reasoning: str

@dataclass
class TeamComposition:
    """Complete team composition for a debate."""
    assignments: List[RoleAssignment]
    diversity_score: float
    coverage_score: float
    topic_alignment: float

class AHMADRoleSpecializer:
    """
    Adaptive role specialization for heterogeneous agent teams.

    Key innovations from A-HMAD:
    1. Topic analysis determines required expertise
    2. Agent capabilities matched to role requirements
    3. Diversity enforced to avoid echo chambers
    4. Dynamic rebalancing during debate
    """

    # Role requirements (what capabilities each role needs)
    ROLE_REQUIREMENTS = {
        ExpertiseRole.DOMAIN_EXPERT: {
            'primary': ['domain_knowledge', 'accuracy'],
            'secondary': ['explanation', 'examples']
        },
        ExpertiseRole.METHODOLOGIST: {
            'primary': ['reasoning', 'structure'],
            'secondary': ['rigor', 'consistency']
        },
        ExpertiseRole.CRITIC: {
            'primary': ['analysis', 'skepticism'],
            'secondary': ['edge_cases', 'assumptions']
        },
        ExpertiseRole.SYNTHESIZER: {
            'primary': ['integration', 'summarization'],
            'secondary': ['clarity', 'coherence']
        },
        ExpertiseRole.FACT_CHECKER: {
            'primary': ['verification', 'sources'],
            'secondary': ['accuracy', 'citations']
        },
        ExpertiseRole.DEVIL_ADVOCATE: {
            'primary': ['contrarian', 'creativity'],
            'secondary': ['alternatives', 'assumptions']
        },
        ExpertiseRole.MEDIATOR: {
            'primary': ['balance', 'fairness'],
            'secondary': ['consensus', 'diplomacy']
        }
    }

    # Agent capability profiles (learned from historical performance)
    # This would be populated from ELO/calibration data
    DEFAULT_AGENT_CAPABILITIES = {
        'claude': {
            'reasoning': 0.95, 'accuracy': 0.9, 'creativity': 0.85,
            'explanation': 0.9, 'structure': 0.85, 'integration': 0.9
        },
        'gpt-4': {
            'reasoning': 0.9, 'accuracy': 0.85, 'creativity': 0.9,
            'explanation': 0.85, 'structure': 0.9, 'integration': 0.85
        },
        'gemini': {
            'reasoning': 0.85, 'accuracy': 0.85, 'creativity': 0.8,
            'explanation': 0.9, 'structure': 0.85, 'integration': 0.8
        },
        'deepseek-r1': {
            'reasoning': 0.95, 'accuracy': 0.9, 'creativity': 0.7,
            'rigor': 0.95, 'structure': 0.9, 'verification': 0.85
        },
        'llama': {
            'reasoning': 0.8, 'accuracy': 0.75, 'creativity': 0.85,
            'contrarian': 0.8, 'alternatives': 0.85
        }
    }

    TOPIC_ANALYSIS_PROMPT = '''Analyze this debate topic and identify what expertise roles are most important.

TOPIC: {topic}
CONTEXT: {context}

For each of these roles, rate importance (1-5) and explain why:
1. DOMAIN_EXPERT - Deep knowledge of the specific subject
2. METHODOLOGIST - Rigorous reasoning and structure
3. CRITIC - Finding flaws and edge cases
4. SYNTHESIZER - Combining perspectives
5. FACT_CHECKER - Verifying claims
6. DEVIL_ADVOCATE - Challenging assumptions
7. MEDIATOR - Building consensus

Output format:
ROLE: [role_name] | IMPORTANCE: [1-5] | REASON: [brief explanation]'''

    def __init__(
        self,
        min_diversity: float = 0.6,
        max_same_model: int = 2
    ):
        self.min_diversity = min_diversity
        self.max_same_model = max_same_model
        self._capability_cache: Dict[str, Dict[str, float]] = {}

    async def compose_team(
        self,
        topic: str,
        context: str,
        available_agents: List[str],
        team_size: int,
        agent_pool: 'AgentPool',
        elo_system: Optional['EloSystem'] = None
    ) -> TeamComposition:
        """
        Compose optimal team with dynamic role assignment.
        """
        # Step 1: Analyze topic to determine role importance
        role_importance = await self._analyze_topic(topic, context, agent_pool)

        # Step 2: Get agent capabilities (from ELO or defaults)
        agent_capabilities = self._get_agent_capabilities(
            available_agents, elo_system
        )

        # Step 3: Match agents to roles
        assignments = self._match_agents_to_roles(
            role_importance=role_importance,
            agent_capabilities=agent_capabilities,
            team_size=team_size
        )

        # Step 4: Enforce diversity
        assignments = self._enforce_diversity(assignments, available_agents)

        # Step 5: Calculate team scores
        diversity_score = self._calculate_diversity(assignments)
        coverage_score = self._calculate_coverage(assignments, role_importance)
        topic_alignment = self._calculate_alignment(assignments, role_importance)

        return TeamComposition(
            assignments=assignments,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            topic_alignment=topic_alignment
        )

    async def _analyze_topic(
        self,
        topic: str,
        context: str,
        agent_pool: 'AgentPool'
    ) -> Dict[ExpertiseRole, float]:
        """Analyze topic to determine role importance."""
        prompt = self.TOPIC_ANALYSIS_PROMPT.format(
            topic=topic,
            context=context[:500]
        )

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=500
        )

        # Parse response
        role_importance = {role: 3.0 for role in ExpertiseRole}  # Default medium

        for line in response.strip().split('\n'):
            if not line.startswith('ROLE:'):
                continue

            try:
                parts = line.split('|')
                role_name = parts[0].replace('ROLE:', '').strip().upper()
                importance = float(parts[1].replace('IMPORTANCE:', '').strip())

                for role in ExpertiseRole:
                    if role.name == role_name:
                        role_importance[role] = importance / 5.0  # Normalize to 0-1
                        break
            except (IndexError, ValueError):
                continue

        return role_importance

    def _get_agent_capabilities(
        self,
        agents: List[str],
        elo_system: Optional['EloSystem']
    ) -> Dict[str, Dict[str, float]]:
        """Get capability profiles for agents."""
        capabilities = {}

        for agent in agents:
            # Try to get from ELO system (learned capabilities)
            if elo_system:
                rating = elo_system.get_rating(agent)
                if rating:
                    # Convert ELO metrics to capabilities
                    capabilities[agent] = self._elo_to_capabilities(rating)
                    continue

            # Fall back to defaults
            base_agent = agent.split('-')[0].lower()
            if base_agent in self.DEFAULT_AGENT_CAPABILITIES:
                capabilities[agent] = self.DEFAULT_AGENT_CAPABILITIES[base_agent].copy()
            else:
                # Generic default
                capabilities[agent] = {cap: 0.7 for cap in
                    ['reasoning', 'accuracy', 'creativity', 'explanation', 'structure']}

        return capabilities

    def _elo_to_capabilities(self, rating: 'AgentRating') -> Dict[str, float]:
        """Convert ELO rating to capability profile."""
        # Normalize ELO (1000 = 0.5, 1400 = 0.9)
        elo_normalized = min(1.0, max(0.0, (rating.elo - 800) / 800))

        # Use calibration for accuracy
        accuracy = rating.calibration_accuracy if rating.calibration_total > 0 else 0.7

        return {
            'reasoning': elo_normalized,
            'accuracy': accuracy,
            'creativity': 0.7,  # Default, could be learned
            'explanation': elo_normalized * 0.9,
            'structure': elo_normalized * 0.85,
            'integration': elo_normalized * 0.9,
            'verification': accuracy,
            'rigor': elo_normalized * 0.95
        }

    def _match_agents_to_roles(
        self,
        role_importance: Dict[ExpertiseRole, float],
        agent_capabilities: Dict[str, Dict[str, float]],
        team_size: int
    ) -> List[RoleAssignment]:
        """Match agents to roles using Hungarian algorithm approximation."""
        assignments = []
        assigned_agents: Set[str] = set()

        # Sort roles by importance
        sorted_roles = sorted(
            role_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for role, importance in sorted_roles[:team_size]:
            if len(assignments) >= team_size:
                break

            # Find best unassigned agent for this role
            best_agent = None
            best_score = -1

            requirements = self.ROLE_REQUIREMENTS.get(role, {})
            primary_caps = requirements.get('primary', [])
            secondary_caps = requirements.get('secondary', [])

            for agent, caps in agent_capabilities.items():
                if agent in assigned_agents:
                    continue

                # Calculate fit score
                primary_score = sum(caps.get(c, 0.5) for c in primary_caps) / max(len(primary_caps), 1)
                secondary_score = sum(caps.get(c, 0.5) for c in secondary_caps) / max(len(secondary_caps), 1)

                fit_score = 0.7 * primary_score + 0.3 * secondary_score

                if fit_score > best_score:
                    best_score = fit_score
                    best_agent = agent

            if best_agent:
                assignments.append(RoleAssignment(
                    agent_id=best_agent,
                    role=role,
                    confidence=best_score,
                    reasoning=f"Best fit for {role.value} based on capabilities"
                ))
                assigned_agents.add(best_agent)

        return assignments

    def _enforce_diversity(
        self,
        assignments: List[RoleAssignment],
        available_agents: List[str]
    ) -> List[RoleAssignment]:
        """Enforce diversity constraints (avoid too many same-model agents)."""
        # Count model families
        model_counts: Dict[str, int] = {}
        for a in assignments:
            base = a.agent_id.split('-')[0].lower()
            model_counts[base] = model_counts.get(base, 0) + 1

        # Replace over-represented models
        for base, count in model_counts.items():
            if count > self.max_same_model:
                # Find agents to replace
                to_replace = [
                    a for a in assignments
                    if a.agent_id.split('-')[0].lower() == base
                ][self.max_same_model:]

                # Find replacement agents
                assigned_ids = {a.agent_id for a in assignments}
                replacements = [
                    agent for agent in available_agents
                    if agent not in assigned_ids and
                    agent.split('-')[0].lower() != base
                ]

                for i, old in enumerate(to_replace):
                    if i < len(replacements):
                        # Replace assignment
                        idx = assignments.index(old)
                        assignments[idx] = RoleAssignment(
                            agent_id=replacements[i],
                            role=old.role,
                            confidence=old.confidence * 0.9,  # Slight penalty
                            reasoning=f"Diversity replacement for {old.role.value}"
                        )

        return assignments

    def _calculate_diversity(self, assignments: List[RoleAssignment]) -> float:
        """Calculate team diversity score."""
        if not assignments:
            return 0.0

        # Count unique model families
        families = set(a.agent_id.split('-')[0].lower() for a in assignments)

        # Diversity = unique families / total assignments
        return len(families) / len(assignments)

    def _calculate_coverage(
        self,
        assignments: List[RoleAssignment],
        role_importance: Dict[ExpertiseRole, float]
    ) -> float:
        """Calculate how well the team covers important roles."""
        if not role_importance:
            return 1.0

        assigned_roles = {a.role for a in assignments}

        # Weight by importance
        covered_importance = sum(
            importance for role, importance in role_importance.items()
            if role in assigned_roles
        )
        total_importance = sum(role_importance.values())

        return covered_importance / total_importance if total_importance > 0 else 0.0

    def _calculate_alignment(
        self,
        assignments: List[RoleAssignment],
        role_importance: Dict[ExpertiseRole, float]
    ) -> float:
        """Calculate alignment between agent confidence and role importance."""
        if not assignments:
            return 0.0

        alignment_scores = []
        for a in assignments:
            importance = role_importance.get(a.role, 0.5)
            # High confidence on high-importance roles = good alignment
            alignment_scores.append(a.confidence * importance)

        return sum(alignment_scores) / len(alignment_scores)
```

**Integration into team_selector.py:**

```python
# In aragora/debate/team_selector.py
from aragora.debate.role_specializer import AHMADRoleSpecializer, TeamComposition

class TeamSelector:
    def __init__(self, ...):
        ...
        self.role_specializer = AHMADRoleSpecializer()
        self.use_dynamic_roles = config.get('dynamic_roles', True)

    async def select_team(
        self,
        topic: str,
        context: str,
        team_size: int,
        ...
    ) -> List[str]:
        """Select debate team with optional dynamic role specialization."""

        if self.use_dynamic_roles:
            # Use A-HMAD dynamic role assignment
            composition = await self.role_specializer.compose_team(
                topic=topic,
                context=context,
                available_agents=self.available_agents,
                team_size=team_size,
                agent_pool=self.agent_pool,
                elo_system=self.elo_system
            )

            # Log team composition
            self.event_emitter.emit('team_composed', {
                'assignments': [
                    {'agent': a.agent_id, 'role': a.role.value, 'confidence': a.confidence}
                    for a in composition.assignments
                ],
                'diversity': composition.diversity_score,
                'coverage': composition.coverage_score
            })

            return [a.agent_id for a in composition.assignments]

        else:
            # Fall back to existing static selection
            return await self._static_select_team(topic, context, team_size)
```

---

### 4.2 SICA Self-Improvement for Nomic Loop

**Goal:** Enhance TestFixer and Nomic Loop with SICA's self-improvement patterns

**Files to modify:**
- `aragora/nomic/testfixer/` - Enhance with SICA patterns
- `aragora/nomic/loop.py` - Add self-editing capabilities

```python
# aragora/nomic/sica_improver.py (NEW FILE)
"""
SICA: Self-Improving Coding Agent
Based on: https://arxiv.org/abs/2504.15228

Enables Aragora to improve its own codebase through:
1. Identifying improvement opportunities
2. Generating patches
3. Validating changes
4. Rolling back failures
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile
import shutil

@dataclass
class ImprovementOpportunity:
    """An identified opportunity for self-improvement."""
    id: str
    file_path: str
    description: str
    category: str  # "performance", "reliability", "readability", "test_coverage"
    priority: float  # 0.0-1.0
    estimated_effort: str  # "trivial", "small", "medium", "large"

@dataclass
class Patch:
    """A code patch for an improvement."""
    opportunity_id: str
    file_path: str
    original_content: str
    patched_content: str
    diff: str

@dataclass
class ValidationResult:
    """Result of validating a patch."""
    patch_id: str
    tests_passed: bool
    tests_run: int
    tests_failed: int
    type_check_passed: bool
    lint_passed: bool
    performance_impact: Optional[float] = None  # % change in relevant metrics

@dataclass
class SICAResult:
    """Result of a SICA self-improvement cycle."""
    opportunities_found: int
    patches_generated: int
    patches_validated: int
    patches_applied: int
    improvements: List[str]
    rollbacks: List[str]

class SICAImprover:
    """
    Self-Improving Coding Agent for Aragora.

    Integrates with existing TestFixer and Nomic Loop to enable
    autonomous codebase improvement.
    """

    OPPORTUNITY_PROMPT = '''Analyze this code file for improvement opportunities.

FILE: {file_path}
CONTENT:
{content}

RECENT ISSUES (from tests/logs):
{issues}

Identify concrete improvements in these categories:
1. PERFORMANCE - Algorithmic or I/O improvements
2. RELIABILITY - Error handling, edge cases
3. READABILITY - Code clarity, documentation
4. TEST_COVERAGE - Missing tests

For each opportunity, output:
OPPORTUNITY: [description]
CATEGORY: [category]
PRIORITY: [0.0-1.0]
EFFORT: [trivial|small|medium|large]
LOCATION: [line numbers or function names]'''

    PATCH_PROMPT = '''Generate a code patch for this improvement.

FILE: {file_path}
CURRENT CODE:
{content}

IMPROVEMENT TO MAKE:
{description}

Output the complete patched file content. Preserve all existing functionality.
Only make changes directly related to the improvement.
Add appropriate tests if applicable.'''

    def __init__(
        self,
        aragora_root: Path,
        test_command: str = "pytest",
        type_check_command: str = "mypy",
        max_patches_per_cycle: int = 5
    ):
        self.aragora_root = Path(aragora_root)
        self.test_command = test_command
        self.type_check_command = type_check_command
        self.max_patches_per_cycle = max_patches_per_cycle
        self._backup_dir: Optional[Path] = None

    async def run_improvement_cycle(
        self,
        target_files: List[str],
        agent_pool: 'AgentPool',
        recent_issues: Optional[List[str]] = None
    ) -> SICAResult:
        """
        Run a complete self-improvement cycle.

        Steps:
        1. Identify improvement opportunities
        2. Prioritize and select top opportunities
        3. Generate patches
        4. Validate each patch
        5. Apply validated patches
        6. Rollback failures
        """
        # Create backup
        self._create_backup()

        try:
            # Step 1: Find opportunities
            all_opportunities = []
            for file_path in target_files:
                opportunities = await self._find_opportunities(
                    file_path, agent_pool, recent_issues or []
                )
                all_opportunities.extend(opportunities)

            # Step 2: Prioritize
            sorted_opportunities = sorted(
                all_opportunities,
                key=lambda o: o.priority,
                reverse=True
            )[:self.max_patches_per_cycle]

            # Step 3: Generate patches
            patches = []
            for opp in sorted_opportunities:
                patch = await self._generate_patch(opp, agent_pool)
                if patch:
                    patches.append(patch)

            # Step 4 & 5: Validate and apply
            applied = []
            rolled_back = []

            for patch in patches:
                validation = await self._validate_patch(patch)

                if validation.tests_passed and validation.type_check_passed:
                    # Apply patch
                    self._apply_patch(patch)
                    applied.append(patch.opportunity_id)
                else:
                    # Don't apply, record as rolled back
                    rolled_back.append(patch.opportunity_id)

            return SICAResult(
                opportunities_found=len(all_opportunities),
                patches_generated=len(patches),
                patches_validated=len([p for p in patches if p]),
                patches_applied=len(applied),
                improvements=[
                    f"Applied: {opp_id}" for opp_id in applied
                ],
                rollbacks=[
                    f"Rolled back: {opp_id}" for opp_id in rolled_back
                ]
            )

        except Exception as e:
            # Rollback everything on error
            self._restore_backup()
            raise

        finally:
            self._cleanup_backup()

    async def _find_opportunities(
        self,
        file_path: str,
        agent_pool: 'AgentPool',
        recent_issues: List[str]
    ) -> List[ImprovementOpportunity]:
        """Find improvement opportunities in a file."""
        full_path = self.aragora_root / file_path

        if not full_path.exists():
            return []

        content = full_path.read_text()

        prompt = self.OPPORTUNITY_PROMPT.format(
            file_path=file_path,
            content=content[:5000],  # Truncate for context
            issues="\n".join(recent_issues[:10]) or "None"
        )

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=1000
        )

        return self._parse_opportunities(response, file_path)

    def _parse_opportunities(
        self,
        response: str,
        file_path: str
    ) -> List[ImprovementOpportunity]:
        """Parse opportunity response."""
        opportunities = []
        current = {}

        for line in response.strip().split('\n'):
            if line.startswith('OPPORTUNITY:'):
                if current:
                    opportunities.append(self._create_opportunity(current, file_path))
                current = {'description': line.replace('OPPORTUNITY:', '').strip()}
            elif line.startswith('CATEGORY:'):
                current['category'] = line.replace('CATEGORY:', '').strip().lower()
            elif line.startswith('PRIORITY:'):
                try:
                    current['priority'] = float(line.replace('PRIORITY:', '').strip())
                except ValueError:
                    current['priority'] = 0.5
            elif line.startswith('EFFORT:'):
                current['effort'] = line.replace('EFFORT:', '').strip().lower()

        if current:
            opportunities.append(self._create_opportunity(current, file_path))

        return opportunities

    def _create_opportunity(
        self,
        data: Dict,
        file_path: str
    ) -> ImprovementOpportunity:
        """Create opportunity from parsed data."""
        import hashlib
        opp_id = hashlib.md5(
            f"{file_path}:{data.get('description', '')}".encode()
        ).hexdigest()[:8]

        return ImprovementOpportunity(
            id=f"opp_{opp_id}",
            file_path=file_path,
            description=data.get('description', ''),
            category=data.get('category', 'readability'),
            priority=data.get('priority', 0.5),
            estimated_effort=data.get('effort', 'medium')
        )

    async def _generate_patch(
        self,
        opportunity: ImprovementOpportunity,
        agent_pool: 'AgentPool'
    ) -> Optional[Patch]:
        """Generate a patch for an opportunity."""
        full_path = self.aragora_root / opportunity.file_path
        original = full_path.read_text()

        prompt = self.PATCH_PROMPT.format(
            file_path=opportunity.file_path,
            content=original,
            description=opportunity.description
        )

        response = await agent_pool.query(
            agent_id="claude",
            prompt=prompt,
            max_tokens=4000
        )

        # Extract code from response
        patched = self._extract_code(response)

        if not patched or patched == original:
            return None

        # Generate diff
        import difflib
        diff = '\n'.join(difflib.unified_diff(
            original.splitlines(),
            patched.splitlines(),
            fromfile=f"a/{opportunity.file_path}",
            tofile=f"b/{opportunity.file_path}"
        ))

        return Patch(
            opportunity_id=opportunity.id,
            file_path=opportunity.file_path,
            original_content=original,
            patched_content=patched,
            diff=diff
        )

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Look for code blocks
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        if '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        return response.strip()

    async def _validate_patch(self, patch: Patch) -> ValidationResult:
        """Validate a patch by running tests and checks."""
        # Apply patch temporarily
        full_path = self.aragora_root / patch.file_path
        original = full_path.read_text()

        try:
            full_path.write_text(patch.patched_content)

            # Run tests
            test_result = subprocess.run(
                [self.test_command, "-x", "--tb=short"],
                cwd=self.aragora_root,
                capture_output=True,
                timeout=300
            )
            tests_passed = test_result.returncode == 0

            # Parse test counts from output
            output = test_result.stdout.decode()
            tests_run = 0
            tests_failed = 0
            # Simplified parsing
            if 'passed' in output:
                tests_passed = True

            # Type check
            type_result = subprocess.run(
                [self.type_check_command, patch.file_path],
                cwd=self.aragora_root,
                capture_output=True,
                timeout=60
            )
            type_check_passed = type_result.returncode == 0

            return ValidationResult(
                patch_id=patch.opportunity_id,
                tests_passed=tests_passed,
                tests_run=tests_run,
                tests_failed=tests_failed,
                type_check_passed=type_check_passed,
                lint_passed=True  # Could add lint check
            )

        finally:
            # Restore original
            full_path.write_text(original)

    def _apply_patch(self, patch: Patch) -> None:
        """Apply a validated patch."""
        full_path = self.aragora_root / patch.file_path
        full_path.write_text(patch.patched_content)

    def _create_backup(self) -> None:
        """Create backup of files that might be modified."""
        self._backup_dir = Path(tempfile.mkdtemp(prefix="aragora_backup_"))

    def _restore_backup(self) -> None:
        """Restore from backup."""
        if self._backup_dir and self._backup_dir.exists():
            # Implementation would restore backed up files
            pass

    def _cleanup_backup(self) -> None:
        """Clean up backup directory."""
        if self._backup_dir and self._backup_dir.exists():
            shutil.rmtree(self._backup_dir)
        self._backup_dir = None
```

---

## Phase 5: Integration & Optimization (Weeks 17-20)

### 5.1 Unified Pipeline Integration

Create a unified configuration that enables all integrated features:

```python
# aragora/config/research_integration.py (NEW FILE)
"""
Configuration for research integration features.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class IntegrationLevel(Enum):
    MINIMAL = "minimal"      # Core features only
    STANDARD = "standard"    # Recommended features
    FULL = "full"           # All research integrations
    CUSTOM = "custom"       # User-defined

@dataclass
class ResearchIntegrationConfig:
    """Master configuration for all research integrations."""

    level: IntegrationLevel = IntegrationLevel.STANDARD

    # Phase 1: Foundation
    muse_enabled: bool = True
    muse_weight: float = 0.15
    ascot_enabled: bool = True
    ascot_lambda: float = 2.0

    # Phase 2: Process Verification
    think_prm_enabled: bool = True
    hilbert_enabled: bool = False  # Requires Lean4
    hilbert_max_depth: int = 5

    # Phase 3: Knowledge & Evidence
    graph_rag_enabled: bool = True
    graph_rag_max_hops: int = 3
    claimcheck_enabled: bool = True

    # Phase 4: Team Selection & Self-Improvement
    ahmag_dynamic_roles: bool = True
    ahmag_min_diversity: float = 0.6
    sica_enabled: bool = False  # Opt-in for self-modification
    sica_max_patches: int = 5

    # Feature flags
    features: List[str] = field(default_factory=lambda: [
        "muse_calibration",
        "ascot_fragility",
        "think_prm",
        "graph_rag",
        "claimcheck",
        "ahmag_roles"
    ])

    @classmethod
    def from_level(cls, level: IntegrationLevel) -> 'ResearchIntegrationConfig':
        """Create config from integration level."""
        if level == IntegrationLevel.MINIMAL:
            return cls(
                level=level,
                muse_enabled=False,
                ascot_enabled=False,
                think_prm_enabled=False,
                graph_rag_enabled=False,
                claimcheck_enabled=False,
                ahmag_dynamic_roles=False,
                features=[]
            )
        elif level == IntegrationLevel.STANDARD:
            return cls(level=level)  # Defaults
        elif level == IntegrationLevel.FULL:
            return cls(
                level=level,
                hilbert_enabled=True,
                sica_enabled=True,
                features=[
                    "muse_calibration",
                    "ascot_fragility",
                    "think_prm",
                    "hilbert_proofs",
                    "graph_rag",
                    "claimcheck",
                    "ahmag_roles",
                    "sica_improvement"
                ]
            )
        return cls(level=IntegrationLevel.CUSTOM)
```

### 5.2 Performance Monitoring

```python
# aragora/monitoring/research_metrics.py (NEW FILE)
"""
Metrics tracking for research integration features.
"""
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class IntegrationMetrics:
    """Metrics for evaluating integration effectiveness."""

    # MUSE metrics
    muse_consensus_confidence_avg: float = 0.0
    muse_subset_size_avg: float = 0.0

    # ASCoT metrics
    ascot_late_stage_errors_caught: int = 0
    ascot_verification_time_saved_ms: float = 0.0

    # ThinkPRM metrics
    prm_steps_verified: int = 0
    prm_errors_detected: int = 0
    prm_accuracy: float = 0.0

    # GraphRAG metrics
    graphrag_path_coverage: float = 0.0
    graphrag_retrieval_quality: float = 0.0

    # ClaimCheck metrics
    claimcheck_claims_verified: int = 0
    claimcheck_atomic_decompositions: int = 0

    # A-HMAD metrics
    ahmag_diversity_score_avg: float = 0.0
    ahmag_role_alignment_avg: float = 0.0

    # Overall
    debate_quality_improvement: float = 0.0
    consensus_time_change: float = 0.0
    verification_confidence_improvement: float = 0.0

class MetricsCollector:
    """Collects and aggregates integration metrics."""

    def __init__(self):
        self._metrics: List[IntegrationMetrics] = []
        self._start_time = datetime.now()

    def record(self, metrics: IntegrationMetrics) -> None:
        """Record metrics from a debate."""
        self._metrics.append(metrics)

    def aggregate(self) -> Dict:
        """Aggregate all recorded metrics."""
        if not self._metrics:
            return {}

        return {
            'total_debates': len(self._metrics),
            'avg_muse_confidence': sum(m.muse_consensus_confidence_avg for m in self._metrics) / len(self._metrics),
            'total_prm_errors_caught': sum(m.prm_errors_detected for m in self._metrics),
            'avg_diversity': sum(m.ahmag_diversity_score_avg for m in self._metrics) / len(self._metrics),
            'avg_quality_improvement': sum(m.debate_quality_improvement for m in self._metrics) / len(self._metrics),
        }
```

---

## Implementation Timeline

| Phase | Weeks | Focus | Key Deliverables |
|-------|-------|-------|------------------|
| 1 | 1-4 | Foundation | Adaptive stopping, MUSE, ASCoT, RLM enhancements, LaRA routing |
| 2 | 5-8 | Verification | ThinkPRM, Hilbert proofs |
| 3 | 9-12 | Knowledge | GraphRAG, ClaimCheck |
| 4 | 13-16 | Team & Self | A-HMAD roles, SICA improvement |
| 5 | 17-20 | Integration | Unified config, metrics, optimization |

---

## Testing Strategy

Each integration should have:

1. **Unit tests** in `tests/research/test_{feature}.py`
2. **Integration tests** in `tests/research/integration/`
3. **Benchmark tests** comparing with/without feature
4. **Regression tests** ensuring no degradation

Example test structure:
```
tests/research/
├── test_muse_calibration.py
├── test_ascot_fragility.py
├── test_think_prm.py
├── test_hilbert_prover.py
├── test_graph_rag.py
├── test_claimcheck.py
├── test_ahmag_roles.py
├── test_sica_improver.py
└── integration/
    ├── test_full_debate_pipeline.py
    └── test_self_improvement_cycle.py
```

---

## Benchmarks & Telemetry Specifications

### Benchmark Suite

Each integration requires before/after benchmarks to validate impact. Run benchmarks on a standardized debate corpus before merging.

#### B1: Adaptive Stability Detection Benchmarks

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Avg rounds per debate | Current | -25% | `mean(rounds_executed)` |
| Compute cost per debate | Current | -20% | `sum(token_usage)` |
| Consensus quality | Current | ≥ Current | `mean(consensus_confidence)` |
| False early stops | 0% | < 5% | `count(early_stop AND later_disagreement) / count(early_stop)` |

**Benchmark script:** `scripts/benchmarks/bench_adaptive_stopping.py`

```python
# scripts/benchmarks/bench_adaptive_stopping.py
"""Benchmark adaptive stability detection."""
import asyncio
from dataclasses import dataclass
from typing import List
from aragora.debate import Arena, DebateConfig
from aragora.debate.stability_detector import BetaBinomialStabilityDetector

@dataclass
class StabilityBenchmarkResult:
    debate_id: str
    baseline_rounds: int
    adaptive_rounds: int
    rounds_saved: int
    baseline_tokens: int
    adaptive_tokens: int
    tokens_saved: int
    consensus_quality_baseline: float
    consensus_quality_adaptive: float
    quality_delta: float
    false_early_stop: bool

async def run_stability_benchmark(
    debate_corpus: List[dict],
    config: DebateConfig
) -> List[StabilityBenchmarkResult]:
    """Run benchmark comparing baseline vs adaptive stopping."""
    results = []

    for debate_spec in debate_corpus:
        # Run baseline (fixed rounds)
        baseline_config = config.copy()
        baseline_config.adaptive_stopping = False
        baseline_arena = Arena(baseline_config)
        baseline_result = await baseline_arena.run_debate(debate_spec['topic'])

        # Run adaptive
        adaptive_config = config.copy()
        adaptive_config.adaptive_stopping = True
        adaptive_arena = Arena(adaptive_config)
        adaptive_result = await adaptive_arena.run_debate(debate_spec['topic'])

        # Check for false early stop (run extra rounds to verify)
        false_early = False
        if adaptive_result.rounds < baseline_result.rounds:
            # Continue debate to see if consensus would have changed
            continuation = await adaptive_arena.continue_debate(
                rounds=baseline_result.rounds - adaptive_result.rounds
            )
            if continuation.consensus != adaptive_result.consensus:
                false_early = True

        results.append(StabilityBenchmarkResult(
            debate_id=debate_spec['id'],
            baseline_rounds=baseline_result.rounds,
            adaptive_rounds=adaptive_result.rounds,
            rounds_saved=baseline_result.rounds - adaptive_result.rounds,
            baseline_tokens=baseline_result.total_tokens,
            adaptive_tokens=adaptive_result.total_tokens,
            tokens_saved=baseline_result.total_tokens - adaptive_result.total_tokens,
            consensus_quality_baseline=baseline_result.consensus_confidence,
            consensus_quality_adaptive=adaptive_result.consensus_confidence,
            quality_delta=adaptive_result.consensus_confidence - baseline_result.consensus_confidence,
            false_early_stop=false_early
        ))

    return results

def report_stability_benchmark(results: List[StabilityBenchmarkResult]) -> dict:
    """Generate benchmark report."""
    import statistics

    return {
        'total_debates': len(results),
        'avg_rounds_saved': statistics.mean(r.rounds_saved for r in results),
        'avg_rounds_saved_pct': statistics.mean(
            r.rounds_saved / r.baseline_rounds for r in results
        ) * 100,
        'avg_tokens_saved': statistics.mean(r.tokens_saved for r in results),
        'avg_tokens_saved_pct': statistics.mean(
            r.tokens_saved / r.baseline_tokens for r in results if r.baseline_tokens > 0
        ) * 100,
        'avg_quality_delta': statistics.mean(r.quality_delta for r in results),
        'false_early_stop_rate': sum(1 for r in results if r.false_early_stop) / len(results) * 100,
        'quality_maintained': all(r.quality_delta >= -0.05 for r in results),
    }
```

#### B2: LaRA Routing Benchmarks

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Retrieval relevance | Current RAG | +15% | `mean(relevance_score)` |
| Retrieval latency | Current | ≤ +10% | `p95(latency_ms)` |
| Token efficiency | Current | +20% | `relevance / tokens_used` |
| Routing accuracy | N/A | > 80% | `correct_route / total_queries` |

**Benchmark script:** `scripts/benchmarks/bench_lara_routing.py`

```python
# scripts/benchmarks/bench_lara_routing.py
"""Benchmark LaRA routing decisions."""
from dataclasses import dataclass
from typing import List, Literal
from aragora.knowledge.mound.api.router import LaRARouter, RetrievalMode

@dataclass
class RoutingBenchmarkResult:
    query_id: str
    query_text: str
    routed_mode: RetrievalMode
    optimal_mode: RetrievalMode  # Human-labeled or heuristic
    routing_correct: bool
    relevance_routed: float
    relevance_baseline: float
    relevance_delta: float
    latency_routed_ms: float
    latency_baseline_ms: float
    tokens_routed: int
    tokens_baseline: int

async def run_routing_benchmark(
    query_corpus: List[dict],  # {query, optimal_mode, expected_relevance}
    knowledge_mound: 'KnowledgeMound'
) -> List[RoutingBenchmarkResult]:
    """Benchmark routing quality vs baseline (always RAG)."""
    results = []
    router = LaRARouter()

    for query_spec in query_corpus:
        query = query_spec['query']
        optimal = RetrievalMode(query_spec['optimal_mode'])

        # Get document features
        doc_features = await knowledge_mound._get_document_features()

        # Route
        decision = router.route(query, doc_features)

        # Execute routed query
        import time
        start = time.perf_counter()
        routed_result = await knowledge_mound.query(query, mode=decision.mode.value)
        routed_latency = (time.perf_counter() - start) * 1000

        # Execute baseline (RAG)
        start = time.perf_counter()
        baseline_result = await knowledge_mound.query(query, mode='rag')
        baseline_latency = (time.perf_counter() - start) * 1000

        results.append(RoutingBenchmarkResult(
            query_id=query_spec['id'],
            query_text=query,
            routed_mode=decision.mode,
            optimal_mode=optimal,
            routing_correct=decision.mode == optimal,
            relevance_routed=routed_result.relevance_score,
            relevance_baseline=baseline_result.relevance_score,
            relevance_delta=routed_result.relevance_score - baseline_result.relevance_score,
            latency_routed_ms=routed_latency,
            latency_baseline_ms=baseline_latency,
            tokens_routed=routed_result.tokens_used,
            tokens_baseline=baseline_result.tokens_used,
        ))

    return results
```

#### B3: MUSE Calibration Benchmarks

| Metric | Baseline (Brier only) | Target | Measurement |
|--------|----------------------|--------|-------------|
| Calibration error | Current ECE | -10% | `expected_calibration_error` |
| Confidence-accuracy correlation | Current | +15% | `pearson(confidence, accuracy)` |
| Subset selection accuracy | N/A | > 75% | `best_subset_wins / total` |

#### B4: ThinkPRM Benchmarks

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Error detection rate | N/A | > 70% | `detected_errors / actual_errors` |
| False positive rate | N/A | < 15% | `false_positives / flagged_steps` |
| Verification overhead | N/A | < 20% debate time | `verification_time / debate_time` |

#### B5: End-to-End Pipeline Benchmarks

| Metric | Baseline | Target | Notes |
|--------|----------|--------|-------|
| Debate quality score | Current | +10% | Composite of consensus confidence, evidence grounding |
| Total compute cost | Current | -15% | Adaptive stopping + routing savings |
| Time to consensus | Current | -20% | Faster convergence |
| Audit trail completeness | Current | +25% | More signals captured |

---

### Telemetry Specification

Production telemetry to track integration health and inform future improvements.

#### T1: Telemetry Events

```python
# aragora/telemetry/research_events.py
"""Telemetry events for research integrations."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TelemetryEventType(Enum):
    # Adaptive Stopping
    STABILITY_CHECK = "stability_check"
    EARLY_TERMINATION = "early_termination"
    STABILITY_GATE_TRIGGERED = "stability_gate_triggered"

    # LaRA Routing
    ROUTING_DECISION = "routing_decision"
    ROUTING_FALLBACK = "routing_fallback"
    ROUTING_OVERRIDE = "routing_override"

    # MUSE
    MUSE_CALCULATION = "muse_calculation"
    MUSE_SUBSET_SELECTED = "muse_subset_selected"

    # ThinkPRM
    PRM_STEP_VERIFIED = "prm_step_verified"
    PRM_ERROR_DETECTED = "prm_error_detected"
    PRM_REVISION_TRIGGERED = "prm_revision_triggered"

    # A-HMAD
    ROLE_ASSIGNMENT = "role_assignment"
    DIVERSITY_SCORE = "diversity_score"

    # General
    FEATURE_FLAG_CHECK = "feature_flag_check"
    INTEGRATION_ERROR = "integration_error"

@dataclass
class TelemetryEvent:
    """Base telemetry event."""
    event_type: TelemetryEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    debate_id: Optional[str] = None
    workspace_id: Optional[str] = None
    round_number: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

# Specific event constructors

def stability_check_event(
    debate_id: str,
    round_number: int,
    is_stable: bool,
    stability_score: float,
    ks_distance: float,
    muse_gated: bool,
    ascot_gated: bool,
    recommendation: str,
) -> TelemetryEvent:
    return TelemetryEvent(
        event_type=TelemetryEventType.STABILITY_CHECK,
        debate_id=debate_id,
        round_number=round_number,
        properties={
            'is_stable': is_stable,
            'stability_score': stability_score,
            'ks_distance': ks_distance,
            'muse_gated': muse_gated,
            'ascot_gated': ascot_gated,
            'recommendation': recommendation,
        }
    )

def routing_decision_event(
    workspace_id: str,
    query_hash: str,  # Anonymized
    selected_mode: str,
    confidence: float,
    doc_tokens: int,
    query_features: Dict[str, Any],
    fallback_mode: Optional[str],
    duration_ms: float,
) -> TelemetryEvent:
    return TelemetryEvent(
        event_type=TelemetryEventType.ROUTING_DECISION,
        workspace_id=workspace_id,
        properties={
            'query_hash': query_hash,
            'selected_mode': selected_mode,
            'confidence': confidence,
            'doc_tokens': doc_tokens,
            'is_factual': query_features.get('is_factual'),
            'is_analytical': query_features.get('is_analytical'),
            'query_length': query_features.get('length_tokens'),
            'fallback_mode': fallback_mode,
        },
        duration_ms=duration_ms,
    )

def muse_calculation_event(
    debate_id: str,
    round_number: int,
    consensus_confidence: float,
    divergence_score: float,
    subset_size: int,
    subset_agents: List[str],
    duration_ms: float,
) -> TelemetryEvent:
    return TelemetryEvent(
        event_type=TelemetryEventType.MUSE_CALCULATION,
        debate_id=debate_id,
        round_number=round_number,
        properties={
            'consensus_confidence': consensus_confidence,
            'divergence_score': divergence_score,
            'subset_size': subset_size,
            'subset_agents': subset_agents,
        },
        duration_ms=duration_ms,
    )

def prm_error_detected_event(
    debate_id: str,
    round_number: int,
    step_id: str,
    agent_id: str,
    verdict: str,
    confidence: float,
    is_late_stage: bool,
) -> TelemetryEvent:
    return TelemetryEvent(
        event_type=TelemetryEventType.PRM_ERROR_DETECTED,
        debate_id=debate_id,
        round_number=round_number,
        properties={
            'step_id': step_id,
            'agent_id': agent_id,
            'verdict': verdict,
            'confidence': confidence,
            'is_late_stage': is_late_stage,
        }
    )
```

#### T2: Telemetry Collector

```python
# aragora/telemetry/collector.py
"""Telemetry collection and aggregation."""
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

from aragora.telemetry.research_events import TelemetryEvent, TelemetryEventType

class TelemetryCollector:
    """Collects and aggregates telemetry events."""

    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval_seconds: int = 60,
        backend: Optional['TelemetryBackend'] = None,
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        self.backend = backend
        self._buffer: List[TelemetryEvent] = []
        self._aggregates: Dict[str, Any] = defaultdict(lambda: defaultdict(float))
        self._flush_task: Optional[asyncio.Task] = None

    async def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        self._buffer.append(event)
        self._update_aggregates(event)

        if len(self._buffer) >= self.buffer_size:
            await self.flush()

    def _update_aggregates(self, event: TelemetryEvent) -> None:
        """Update running aggregates."""
        key = event.event_type.value

        self._aggregates[key]['count'] += 1

        if event.duration_ms:
            self._aggregates[key]['total_duration_ms'] += event.duration_ms
            self._aggregates[key]['max_duration_ms'] = max(
                self._aggregates[key]['max_duration_ms'],
                event.duration_ms
            )

        # Event-specific aggregates
        if event.event_type == TelemetryEventType.STABILITY_CHECK:
            if event.properties.get('is_stable'):
                self._aggregates[key]['stable_count'] += 1
            if event.properties.get('muse_gated'):
                self._aggregates[key]['muse_gated_count'] += 1
            if event.properties.get('ascot_gated'):
                self._aggregates[key]['ascot_gated_count'] += 1

        elif event.event_type == TelemetryEventType.ROUTING_DECISION:
            mode = event.properties.get('selected_mode', 'unknown')
            self._aggregates[key][f'mode_{mode}_count'] += 1

        elif event.event_type == TelemetryEventType.PRM_ERROR_DETECTED:
            if event.properties.get('is_late_stage'):
                self._aggregates[key]['late_stage_errors'] += 1

    async def flush(self) -> None:
        """Flush buffer to backend."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        if self.backend:
            await self.backend.write(events)

    def get_aggregates(self) -> Dict[str, Any]:
        """Get current aggregates."""
        result = {}
        for event_type, aggs in self._aggregates.items():
            result[event_type] = dict(aggs)
            if aggs['count'] > 0 and aggs.get('total_duration_ms', 0) > 0:
                result[event_type]['avg_duration_ms'] = (
                    aggs['total_duration_ms'] / aggs['count']
                )
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary."""
        aggs = self.get_aggregates()

        summary = {
            'period_start': datetime.utcnow() - timedelta(seconds=self.flush_interval),
            'period_end': datetime.utcnow(),
        }

        # Stability summary
        if 'stability_check' in aggs:
            sc = aggs['stability_check']
            summary['adaptive_stopping'] = {
                'checks': sc['count'],
                'stable_rate': sc.get('stable_count', 0) / sc['count'] if sc['count'] > 0 else 0,
                'muse_gate_rate': sc.get('muse_gated_count', 0) / sc['count'] if sc['count'] > 0 else 0,
                'ascot_gate_rate': sc.get('ascot_gated_count', 0) / sc['count'] if sc['count'] > 0 else 0,
            }

        # Routing summary
        if 'routing_decision' in aggs:
            rd = aggs['routing_decision']
            total = rd['count']
            summary['lara_routing'] = {
                'total_queries': total,
                'rag_pct': rd.get('mode_rag_count', 0) / total * 100 if total > 0 else 0,
                'rlm_pct': rd.get('mode_rlm_count', 0) / total * 100 if total > 0 else 0,
                'graph_pct': rd.get('mode_graph_count', 0) / total * 100 if total > 0 else 0,
                'long_context_pct': rd.get('mode_long_context_count', 0) / total * 100 if total > 0 else 0,
                'avg_latency_ms': rd.get('avg_duration_ms', 0),
            }

        # PRM summary
        if 'prm_error_detected' in aggs:
            prm = aggs['prm_error_detected']
            summary['think_prm'] = {
                'errors_detected': prm['count'],
                'late_stage_errors': prm.get('late_stage_errors', 0),
                'late_stage_pct': prm.get('late_stage_errors', 0) / prm['count'] * 100 if prm['count'] > 0 else 0,
            }

        return summary
```

#### T3: Dashboards & Alerts

**Grafana Dashboard Panels:**

1. **Adaptive Stopping Effectiveness**
   - Rounds saved per debate (time series)
   - Gate trigger rates (MUSE vs ASCoT)
   - False early stop rate (should be < 5%)

2. **LaRA Routing Distribution**
   - Mode selection pie chart
   - Relevance by mode (box plot)
   - Latency by mode (percentiles)

3. **MUSE Ensemble Health**
   - Divergence score distribution
   - Subset size trends
   - Confidence-accuracy calibration curve

4. **ThinkPRM Error Detection**
   - Errors per debate (time series)
   - Late-stage error ratio
   - Revision trigger rate

**Alert Thresholds:**

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| False early stop rate | > 3% | > 5% | Increase stability threshold |
| Routing fallback rate | > 10% | > 20% | Review router logic |
| PRM false positive rate | > 10% | > 15% | Retune verifier |
| MUSE timeout rate | > 5% | > 10% | Reduce subset enumeration |
| Late-stage error spike | +50% | +100% | Investigate debate quality |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance degradation | Feature flags, lazy loading, caching |
| Compatibility issues | Phased rollout, extensive integration tests |
| Formal verification failures | Graceful degradation, timeout handling |
| Self-improvement instability | Sandboxing, validation gates, human approval |
| Increased complexity | Clear documentation, modular design |

---

## Success Metrics

1. **Calibration Improvement**: 10% reduction in Brier score
2. **Error Detection**: 20% more errors caught before consensus
3. **Retrieval Quality**: 15% improvement in evidence relevance
4. **Team Diversity**: Maintain >0.6 diversity score
5. **Self-Improvement**: 5% test coverage increase per cycle

---

## Conclusion

This implementation plan integrates 8 major research advances into Aragora:

1. **MUSE** - Ensemble uncertainty quantification
2. **ASCoT** - Late-stage fragility detection
3. **ThinkPRM** - Process reward verification
4. **Hilbert** - Recursive formal proofs
5. **GraphRAG** - Hybrid graph+vector retrieval
6. **ClaimCheck** - Atomic claim verification
7. **A-HMAD** - Dynamic role specialization
8. **SICA** - Self-improving coding agent

Each builds on existing Aragora infrastructure, resolves conflicts with alternatives, and provides specific integration points with existing code.

---

## Appendix A: Additional Phase 1 Integrations (Codex Additions)

### A.1 Adaptive Stability Detection - Full Implementation

**Goal:** Stop debate rounds when consensus becomes statistically stable (Beta-Binomial), lowering compute cost.

**Integration with MUSE:** Feed MUSE divergence scores into stability calculation for richer signal. Don't stop if MUSE shows high disagreement even if votes appear stable.

**Integration with ASCoT:** Don't trigger early stopping during high-fragility rounds (typically rounds > 70% of max). ASCoT fragility score should gate stability-based termination.

```python
# aragora/debate/stability_detector.py (NEW FILE)
"""
Adaptive Stability Detection via Beta-Binomial Mixture
Based on: https://arxiv.org/abs/2510.12697
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats

@dataclass
class StabilityResult:
    """Result of stability detection."""
    is_stable: bool
    stability_score: float      # 0.0-1.0, higher = more stable
    ks_distance: float          # KS-distance between vote distributions
    rounds_since_stable: int    # How many rounds stability has held
    recommendation: str         # "stop", "continue", "one_more_round"
    muse_gated: bool           # True if MUSE disagreement blocked stopping
    ascot_gated: bool          # True if ASCoT fragility blocked stopping

class BetaBinomialStabilityDetector:
    """
    Detects consensus stability using Beta-Binomial mixture model.

    Key insight from MAD/Judge paper: Use KS-distance between consecutive
    vote distributions to detect when consensus has stabilized.
    """

    def __init__(
        self,
        stability_threshold: float = 0.85,
        ks_threshold: float = 0.1,      # Max KS-distance for stability
        min_stable_rounds: int = 1,      # Rounds stability must hold
        muse_disagreement_gate: float = 0.4,  # Max MUSE divergence to allow stop
        ascot_fragility_gate: float = 0.7,    # Max fragility to allow stop
    ):
        self.stability_threshold = stability_threshold
        self.ks_threshold = ks_threshold
        self.min_stable_rounds = min_stable_rounds
        self.muse_disagreement_gate = muse_disagreement_gate
        self.ascot_fragility_gate = ascot_fragility_gate
        self._vote_history: List[Dict[str, float]] = []
        self._stable_since: Optional[int] = None

    def update(
        self,
        round_votes: Dict[str, float],  # agent_id -> vote weight for winner
        round_num: int,
        muse_divergence: Optional[float] = None,
        ascot_fragility: Optional[float] = None,
    ) -> StabilityResult:
        """
        Update with new round votes and check stability.

        Args:
            round_votes: Mapping of agent_id to vote weight
            round_num: Current round number
            muse_divergence: Optional MUSE JSD score (0-1, lower = more agreement)
            ascot_fragility: Optional ASCoT fragility score (0-1, higher = more fragile)
        """
        self._vote_history.append(round_votes)

        if len(self._vote_history) < 2:
            return StabilityResult(
                is_stable=False,
                stability_score=0.0,
                ks_distance=1.0,
                rounds_since_stable=0,
                recommendation="continue",
                muse_gated=False,
                ascot_gated=False
            )

        # Calculate KS-distance between last two rounds
        prev_dist = self._to_distribution(self._vote_history[-2])
        curr_dist = self._to_distribution(self._vote_history[-1])

        ks_stat, _ = stats.ks_2samp(prev_dist, curr_dist)

        # Calculate stability score (inverse of KS distance)
        stability_score = 1.0 - min(ks_stat, 1.0)

        # Check if stable
        is_stable = ks_stat < self.ks_threshold

        # Track stable rounds
        if is_stable:
            if self._stable_since is None:
                self._stable_since = round_num
            rounds_since_stable = round_num - self._stable_since + 1
        else:
            self._stable_since = None
            rounds_since_stable = 0

        # Check gates
        muse_gated = muse_divergence is not None and muse_divergence > self.muse_disagreement_gate
        ascot_gated = ascot_fragility is not None and ascot_fragility > self.ascot_fragility_gate

        # Determine recommendation
        if not is_stable:
            recommendation = "continue"
        elif muse_gated:
            recommendation = "continue"  # MUSE says disagreement too high
        elif ascot_gated:
            recommendation = "one_more_round"  # ASCoT says we're in fragile zone
        elif rounds_since_stable >= self.min_stable_rounds:
            recommendation = "stop"
        else:
            recommendation = "one_more_round"

        return StabilityResult(
            is_stable=is_stable,
            stability_score=stability_score,
            ks_distance=ks_stat,
            rounds_since_stable=rounds_since_stable,
            recommendation=recommendation,
            muse_gated=muse_gated,
            ascot_gated=ascot_gated
        )

    def _to_distribution(self, votes: Dict[str, float]) -> np.ndarray:
        """Convert vote dict to normalized distribution."""
        values = np.array(list(votes.values()))
        if values.sum() == 0:
            return np.ones_like(values) / len(values)
        return values / values.sum()

    def reset(self):
        """Reset for new debate."""
        self._vote_history.clear()
        self._stable_since = None
```

**Integration into ConsensusEstimator:**

```python
# In aragora/debate/ml_integration.py - extend ConsensusEstimator

from aragora.debate.stability_detector import BetaBinomialStabilityDetector, StabilityResult

class ConsensusEstimator:
    def __init__(self, ...):
        ...
        self.stability_detector = BetaBinomialStabilityDetector()

    def estimate_consensus_with_stability(
        self,
        responses: Sequence[tuple[str, str]],
        round_votes: Dict[str, float],
        round_num: int,
        muse_divergence: Optional[float] = None,
        ascot_fragility: Optional[float] = None,
        **kwargs
    ) -> dict[str, Any]:
        """Extended consensus estimation with stability detection."""

        # Get base estimation
        base_result = self.estimate_consensus(responses, **kwargs)

        # Add stability detection
        stability = self.stability_detector.update(
            round_votes=round_votes,
            round_num=round_num,
            muse_divergence=muse_divergence,
            ascot_fragility=ascot_fragility
        )

        # Combine recommendations
        if stability.recommendation == "stop" and base_result["recommendation"] != "intervene":
            combined_recommendation = "terminate"
        else:
            combined_recommendation = base_result["recommendation"]

        return {
            **base_result,
            "stability": {
                "is_stable": stability.is_stable,
                "score": stability.stability_score,
                "ks_distance": stability.ks_distance,
                "rounds_stable": stability.rounds_since_stable,
                "muse_gated": stability.muse_gated,
                "ascot_gated": stability.ascot_gated,
            },
            "recommendation": combined_recommendation,
        }
```

---

### A.2 LaRA Router (RAG vs RLM vs Long-Context)

**Goal:** Dynamically route retrieval to RAG, Graph, RLM, or long-context based on query characteristics.

**Based on:** [LaRA Benchmark](https://arxiv.org/abs/2502.09977) - shows optimal strategy varies by task type and document characteristics.

```python
# aragora/knowledge/mound/api/router.py (NEW FILE)
"""
LaRA-style Retrieval Router
Based on: https://arxiv.org/abs/2502.09977

Dynamically selects between RAG, RLM, Graph, and Long-Context
based on query and document characteristics.
"""
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List
from enum import Enum

class RetrievalMode(Enum):
    RAG = "rag"                    # Vector + BM25 retrieval
    RLM = "rlm"                    # Recursive Language Model
    GRAPH = "graph"               # Graph traversal
    LONG_CONTEXT = "long_context"  # Full document in context
    HYBRID = "hybrid"             # Combination

@dataclass
class RoutingDecision:
    """Decision from the LaRA router."""
    mode: RetrievalMode
    confidence: float           # 0.0-1.0
    reasoning: str
    fallback_mode: Optional[RetrievalMode] = None
    estimated_tokens: int = 0
    estimated_quality: float = 0.0

@dataclass
class QueryFeatures:
    """Features extracted from query for routing."""
    length_tokens: int
    is_factual: bool            # Seeks specific facts
    is_analytical: bool         # Requires synthesis/reasoning
    is_comparative: bool        # Compares multiple items
    requires_recency: bool      # Needs recent information
    specificity: float          # 0.0 (broad) to 1.0 (narrow)
    entities_count: int         # Named entities in query

@dataclass
class DocumentFeatures:
    """Features of the document corpus for routing."""
    total_tokens: int
    document_count: int
    avg_doc_length: int
    has_graph_relationships: bool
    relationship_density: float  # Edges per node
    freshness_days: float       # Average age of documents

class LaRARouter:
    """
    Routes queries to optimal retrieval strategy.

    Decision matrix based on LaRA findings:
    - Short factual queries → RAG (fast, precise)
    - Analytical queries on small corpus → Long-context
    - Analytical queries on large corpus → RLM
    - Queries with entity relationships → Graph + RAG hybrid
    - Complex multi-hop queries → RLM with graph
    """

    # Token thresholds
    LONG_CONTEXT_MAX = 100_000   # Max tokens for long-context
    RLM_MIN_BENEFIT = 50_000    # Min tokens where RLM helps
    RAG_SWEET_SPOT = 10_000     # RAG works best here

    def __init__(
        self,
        prefer_rlm: bool = True,          # Prefer RLM when available
        rlm_cost_multiplier: float = 1.5,  # RLM cost relative to RAG
        quality_threshold: float = 0.7,    # Min acceptable quality
    ):
        self.prefer_rlm = prefer_rlm
        self.rlm_cost_multiplier = rlm_cost_multiplier
        self.quality_threshold = quality_threshold

    def route(
        self,
        query: str,
        doc_features: DocumentFeatures,
        rlm_available: bool = True,
        graph_available: bool = True,
        budget_tokens: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Select optimal retrieval mode.

        Args:
            query: The user query
            doc_features: Characteristics of document corpus
            rlm_available: Whether RLM backend is available
            graph_available: Whether graph relationships exist
            budget_tokens: Optional token budget constraint
        """
        query_features = self._extract_query_features(query)

        # Calculate scores for each mode
        scores = self._calculate_mode_scores(
            query_features, doc_features, rlm_available, graph_available
        )

        # Apply constraints
        if budget_tokens:
            scores = self._apply_budget_constraints(scores, doc_features, budget_tokens)

        # Select best mode
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]

        # Determine fallback
        sorted_modes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fallback = sorted_modes[1][0] if len(sorted_modes) > 1 else None

        return RoutingDecision(
            mode=best_mode,
            confidence=best_score,
            reasoning=self._generate_reasoning(query_features, doc_features, best_mode),
            fallback_mode=fallback,
            estimated_tokens=self._estimate_tokens(best_mode, doc_features),
            estimated_quality=best_score
        )

    def _extract_query_features(self, query: str) -> QueryFeatures:
        """Extract features from query text."""
        words = query.split()
        query_lower = query.lower()

        # Simple heuristics (could be enhanced with ML)
        is_factual = any(w in query_lower for w in ['what', 'who', 'when', 'where', 'which'])
        is_analytical = any(w in query_lower for w in ['why', 'how', 'explain', 'analyze', 'compare'])
        is_comparative = any(w in query_lower for w in ['compare', 'versus', 'difference', 'better', 'vs'])
        requires_recency = any(w in query_lower for w in ['recent', 'latest', 'current', 'new', 'today'])

        # Specificity based on length and structure
        specificity = min(1.0, len(words) / 20)  # Longer = more specific

        # Count potential entities (capitalized words)
        entities = sum(1 for w in words if w[0].isupper() and len(w) > 1)

        return QueryFeatures(
            length_tokens=len(words),
            is_factual=is_factual,
            is_analytical=is_analytical,
            is_comparative=is_comparative,
            requires_recency=requires_recency,
            specificity=specificity,
            entities_count=entities
        )

    def _calculate_mode_scores(
        self,
        query: QueryFeatures,
        docs: DocumentFeatures,
        rlm_available: bool,
        graph_available: bool
    ) -> Dict[RetrievalMode, float]:
        """Calculate quality scores for each retrieval mode."""
        scores = {}

        # RAG: Good for factual, specific queries
        rag_score = 0.7
        if query.is_factual:
            rag_score += 0.15
        if query.specificity > 0.5:
            rag_score += 0.1
        if docs.total_tokens > self.RAG_SWEET_SPOT:
            rag_score -= 0.1  # RAG degrades on very large corpora
        scores[RetrievalMode.RAG] = min(1.0, rag_score)

        # Long-context: Good for small corpora, analytical queries
        if docs.total_tokens <= self.LONG_CONTEXT_MAX:
            lc_score = 0.8
            if query.is_analytical:
                lc_score += 0.15
            if docs.total_tokens < 20_000:
                lc_score += 0.1  # Sweet spot for long-context
            scores[RetrievalMode.LONG_CONTEXT] = min(1.0, lc_score)
        else:
            scores[RetrievalMode.LONG_CONTEXT] = 0.0  # Can't fit

        # RLM: Good for large corpora, complex queries
        if rlm_available:
            rlm_score = 0.6
            if docs.total_tokens > self.RLM_MIN_BENEFIT:
                rlm_score += 0.2  # RLM shines on large corpora
            if query.is_analytical:
                rlm_score += 0.15
            if self.prefer_rlm:
                rlm_score += 0.1
            scores[RetrievalMode.RLM] = min(1.0, rlm_score)
        else:
            scores[RetrievalMode.RLM] = 0.0

        # Graph: Good for relationship queries, comparative
        if graph_available and docs.has_graph_relationships:
            graph_score = 0.5
            if query.is_comparative:
                graph_score += 0.25
            if query.entities_count > 1:
                graph_score += 0.15  # Multiple entities → relationship query
            if docs.relationship_density > 0.5:
                graph_score += 0.1
            scores[RetrievalMode.GRAPH] = min(1.0, graph_score)
        else:
            scores[RetrievalMode.GRAPH] = 0.0

        # Hybrid: For complex multi-faceted queries
        if query.is_analytical and query.entities_count > 1 and graph_available:
            hybrid_score = 0.7
            if docs.has_graph_relationships:
                hybrid_score += 0.15
            scores[RetrievalMode.HYBRID] = min(1.0, hybrid_score)
        else:
            scores[RetrievalMode.HYBRID] = 0.0

        return scores

    def _apply_budget_constraints(
        self,
        scores: Dict[RetrievalMode, float],
        docs: DocumentFeatures,
        budget: int
    ) -> Dict[RetrievalMode, float]:
        """Penalize modes that exceed budget."""
        adjusted = scores.copy()

        # Long-context uses most tokens
        if docs.total_tokens > budget:
            adjusted[RetrievalMode.LONG_CONTEXT] = 0.0

        # RLM has higher cost
        if docs.total_tokens * self.rlm_cost_multiplier > budget * 2:
            adjusted[RetrievalMode.RLM] *= 0.7

        return adjusted

    def _estimate_tokens(self, mode: RetrievalMode, docs: DocumentFeatures) -> int:
        """Estimate tokens used by mode."""
        if mode == RetrievalMode.LONG_CONTEXT:
            return docs.total_tokens
        elif mode == RetrievalMode.RAG:
            return min(docs.total_tokens, 4000)  # Typical RAG context
        elif mode == RetrievalMode.RLM:
            return int(docs.total_tokens * 0.3)  # RLM compression
        elif mode == RetrievalMode.GRAPH:
            return min(docs.total_tokens, 8000)  # Graph traversal
        else:
            return min(docs.total_tokens, 10000)  # Hybrid

    def _generate_reasoning(
        self,
        query: QueryFeatures,
        docs: DocumentFeatures,
        mode: RetrievalMode
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []

        if mode == RetrievalMode.RAG:
            reasons.append("RAG selected for factual/specific query")
            if docs.total_tokens <= self.RAG_SWEET_SPOT:
                reasons.append("corpus size in RAG sweet spot")

        elif mode == RetrievalMode.LONG_CONTEXT:
            reasons.append("Long-context selected for small corpus")
            if query.is_analytical:
                reasons.append("analytical query benefits from full context")

        elif mode == RetrievalMode.RLM:
            reasons.append("RLM selected for large corpus")
            if query.is_analytical:
                reasons.append("complex query needs hierarchical access")

        elif mode == RetrievalMode.GRAPH:
            reasons.append("Graph selected for relationship query")
            if query.entities_count > 1:
                reasons.append(f"multiple entities ({query.entities_count}) suggest relationships")

        elif mode == RetrievalMode.HYBRID:
            reasons.append("Hybrid selected for multi-faceted query")

        return "; ".join(reasons)
```

**Integration into QueryOperationsMixin:**

```python
# In aragora/knowledge/mound/api/query.py

from aragora.knowledge.mound.api.router import LaRARouter, RoutingDecision, DocumentFeatures, RetrievalMode

class QueryOperationsMixin:
    def __init__(self, ...):
        ...
        self.lara_router = LaRARouter()
        self.auto_routing = config.get('auto_routing', True)

    async def query(
        self,
        query_text: str,
        mode: Optional[str] = None,  # Force specific mode
        **kwargs
    ) -> QueryResult:
        """
        Query with automatic LaRA routing.

        Args:
            query_text: The query
            mode: Optional forced mode ("rag", "rlm", "graph", "long_context")
            **kwargs: Additional query parameters
        """
        # If mode forced, use it
        if mode:
            return await self._query_with_mode(query_text, mode, **kwargs)

        # If auto-routing disabled, default to semantic
        if not self.auto_routing:
            return await self.query_semantic(query_text, **kwargs)

        # Get document features
        doc_features = await self._get_document_features()

        # Route
        decision = self.lara_router.route(
            query=query_text,
            doc_features=doc_features,
            rlm_available=self.is_rlm_available(),
            graph_available=self._has_graph_relationships(),
        )

        # Log routing decision
        self._log_routing_decision(decision)

        # Execute with selected mode
        try:
            return await self._query_with_mode(query_text, decision.mode.value, **kwargs)
        except Exception as e:
            # Fallback on error
            if decision.fallback_mode:
                return await self._query_with_mode(
                    query_text, decision.fallback_mode.value, **kwargs
                )
            raise

    async def _query_with_mode(
        self,
        query_text: str,
        mode: str,
        **kwargs
    ) -> QueryResult:
        """Execute query with specific mode."""
        if mode == "rag":
            return await self.query_semantic(query_text, **kwargs)
        elif mode == "rlm":
            return await self.query_with_rlm(query_text, **kwargs)
        elif mode == "graph":
            return await self.query_graph(query_text, **kwargs)
        elif mode == "long_context":
            return await self._query_full_context(query_text, **kwargs)
        elif mode == "hybrid":
            return await self.hybrid_retrieve(query_text, **kwargs)
        else:
            return await self.query_semantic(query_text, **kwargs)

    async def _get_document_features(self) -> DocumentFeatures:
        """Get document corpus features for routing."""
        # Implementation would query actual corpus stats
        stats = await self.get_corpus_stats()
        return DocumentFeatures(
            total_tokens=stats.get('total_tokens', 0),
            document_count=stats.get('document_count', 0),
            avg_doc_length=stats.get('avg_doc_length', 0),
            has_graph_relationships=stats.get('has_relationships', False),
            relationship_density=stats.get('relationship_density', 0.0),
            freshness_days=stats.get('avg_age_days', 30.0)
        )
```

**Config flags for safe rollout:**

```python
# In aragora/knowledge/mound/config.py

@dataclass
class KnowledgeMoundConfig:
    # ... existing fields ...

    # LaRA routing
    auto_routing: bool = True                    # Enable automatic routing
    routing_log_decisions: bool = True           # Log routing decisions
    routing_prefer_rlm: bool = True              # Prefer RLM when applicable
    routing_quality_threshold: float = 0.7       # Min quality to use mode
    routing_budget_tokens: Optional[int] = None  # Optional token budget
```

**Goal:** Dynamically route retrieval to RAG, Graph, or RLM depending on query type.

**Codebase integration points:**
- `aragora/knowledge/mound/api/query.py` → add router inside `query()`
- `aragora/knowledge/mound/api/rlm.py` → use `query_with_rlm()` when routed to RLM
- `aragora/knowledge/embeddings.py` → keep as RAG backend (vector + BM25)

**Routing signals:**
- Query length / specificity
- Workspace document volume
- Existing RLM availability (`is_rlm_available()`)
- Graph relationship density (if available)

**Why this fits:** `query_semantic`, `query_graph`, and `query_with_rlm` already exist.
