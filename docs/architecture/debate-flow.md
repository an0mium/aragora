# Debate Flow

A structured 9-round debate orchestrated by the Arena class. Agents propose,
critique, revise, and converge toward consensus under judge evaluation.

## Round Structure

```mermaid
flowchart TD
    Start([Debate Start]) --> R0

    subgraph R0["Round 0 -- Context Gathering"]
        Ctx["Gather environment context"]
        KnowledgeInject["Inject Knowledge Mound context"]
        PulseInject["Inject Pulse trending topics"]
        Ctx --> KnowledgeInject --> PulseInject
    end

    R0 --> R1

    subgraph R1R2["Rounds 1-2 -- Proposals and Critique"]
        R1["Round 1: Initial Proposals"]
        R2["Round 2: Cross-Agent Critique"]
        R1 --> R2
    end

    R1R2 --> R3

    subgraph R3R5["Rounds 3-5 -- Revision with Convergence"]
        R3["Round 3: First Revision"]
        CD1{Convergence detected?}
        R4["Round 4: Second Revision"]
        CD2{Convergence detected?}
        R5["Round 5: Third Revision"]
        R3 --> CD1
        CD1 -- No --> R4
        CD1 -- Yes --> EarlyConsensus
        R4 --> CD2
        CD2 -- No --> R5
        CD2 -- Yes --> EarlyConsensus
    end

    R3R5 --> R6

    subgraph R6R7["Rounds 6-7 -- Refinement and Voting"]
        R6["Round 6: Final Refinement"]
        R7["Round 7: Consensus Voting"]
        R6 --> R7
    end

    R6R7 --> R8

    subgraph R8["Round 8 -- Judge Evaluation"]
        Judge["Judge evaluates proposals"]
        EarlyStop{Early stop criteria met?}
        Judge --> EarlyStop
    end

    EarlyStop -- Yes --> Done
    EarlyStop -- No --> Extended["Extended rounds if needed"]
    Extended --> Done

    EarlyConsensus([Early Consensus]) --> Done([Debate End])
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant A as Arena
    participant TS as TeamSelector
    participant Ag as Agents (N)
    participant Con as Consensus
    participant CV as Convergence
    participant Mem as Memory
    participant KM as Knowledge Mound

    A->>KM: Fetch relevant knowledge
    A->>Mem: Load cross-debate memory
    A->>TS: Select agent team (ELO + calibration)
    TS-->>A: Selected agents

    rect rgb(230, 240, 255)
        Note over A,Ag: Round 0 -- Context
        A->>Ag: Distribute environment + knowledge context
    end

    rect rgb(240, 255, 240)
        Note over A,Ag: Rounds 1-2 -- Propose and Critique
        A->>Ag: Request initial proposals (concurrent)
        Ag-->>A: Proposals
        A->>Ag: Request critiques of other proposals
        Ag-->>A: Critiques
    end

    rect rgb(255, 245, 230)
        Note over A,CV: Rounds 3-5 -- Revise and Converge
        loop Each revision round
            A->>Ag: Request revisions incorporating critiques
            Ag-->>A: Revised proposals
            A->>CV: Check semantic similarity
            CV-->>A: Convergence score
            alt Score above threshold
                A->>Con: Trigger early consensus
            end
        end
    end

    rect rgb(245, 235, 255)
        Note over A,Con: Rounds 6-7 -- Refine and Vote
        A->>Ag: Final refinement pass
        Ag-->>A: Final proposals
        A->>Ag: Cast consensus votes
        Ag-->>A: Votes
        A->>Con: Evaluate votes
        Con-->>A: Consensus result
    end

    rect rgb(255, 235, 235)
        Note over A,Ag: Round 8 -- Judge
        A->>Ag: Judge evaluates all proposals
        Ag-->>A: Judgment + scores
        A->>Mem: Persist outcome
        A->>KM: Store decision + evidence
    end
```

## Phase Details

| Round | Phase | Concurrency | Key Mechanism |
|-------|-------|-------------|---------------|
| 0 | Context | Sequential | Knowledge injection, Pulse topics |
| 1 | Proposals | `MAX_CONCURRENT_PROPOSALS` | Parallel agent generation |
| 2 | Critique | `MAX_CONCURRENT_CRITIQUES` | Cross-agent review |
| 3-5 | Revision | `MAX_CONCURRENT_REVISIONS` | Convergence detection per round |
| 6 | Refinement | Parallel | Incorporate all feedback |
| 7 | Voting | Parallel | Majority / supermajority consensus |
| 8 | Judgment | Single judge | ELO update, early stop check |

## Agent Roles

- **Proposers** -- Generate and revise solutions through rounds 1-6.
- **Critics** -- Evaluate proposals in round 2; all agents may critique.
- **Voters** -- All agents vote in round 7 on final proposals.
- **Judge** -- Single designated agent scores and ranks in round 8.
- **Trickster** (optional) -- Detects hollow consensus when enabled.
- **Rhetorical Observer** (optional) -- Monitors argument quality.

## Convergence Detection

Semantic similarity is measured between proposals at rounds 3, 4, and 5.
When the convergence score exceeds the configured threshold, the debate
short-circuits to consensus, saving compute and time. The threshold is
configurable via `DebateProtocol.convergence_threshold`.

## Post-Debate

After the debate concludes:

1. Results are routed to the originating channel (`debate_origin.py`).
2. ELO ratings are updated based on judge scores.
3. Memory coordinator atomically writes outcomes to all tiers.
4. Knowledge Mound stores the decision, evidence, and receipts.
5. Optional post-debate workflows trigger (e.g., notifications, exports).
