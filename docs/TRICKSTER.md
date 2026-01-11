# Evidence-Powered Trickster

The Trickster is a quality enforcement system that detects and challenges "hollow consensus" - situations where agents agree without substantive evidence backing their positions. It maintains intellectual rigor in debates by injecting targeted challenges when quality drops.

## Overview

The Trickster operates passively during debates, monitoring evidence quality and convergence patterns. When it detects hollow consensus forming, it intervenes with:

1. **Challenge Prompts** - Injected questions demanding evidence
2. **Role Assignments** - Assigning the QUALITY_CHALLENGER role to agents
3. **Extended Rounds** - Adding extra rounds for evidence gathering
4. **Breakpoints** - Triggering human review for severe cases

## Enabling the Trickster

### Via Protocol Flag

```python
from aragora import Arena, Environment, DebateProtocol

protocol = DebateProtocol(
    rounds=5,
    enable_trickster=True  # Enable hollow consensus detection
)

arena = Arena(env, agents, protocol)
result = await arena.run()
```

### Via API

```json
POST /api/debates
{
  "task": "Should we migrate to Kubernetes?",
  "agents": ["claude", "gpt4"],
  "protocol": {
    "rounds": 5,
    "enable_trickster": true
  }
}
```

### Configuration Options

```python
from aragora.debate.trickster import TricksterConfig, EvidencePoweredTrickster

config = TricksterConfig(
    # Quality thresholds
    min_quality_threshold=0.65,      # Minimum acceptable evidence quality (0-1)
    hollow_detection_threshold=0.5,  # Alert severity to trigger intervention

    # Intervention cooldown
    intervention_cooldown_rounds=1,  # Rounds between interventions

    # Feature flags
    enable_challenge_prompts=True,
    enable_role_assignment=True,
    enable_extended_rounds=True,
    enable_breakpoints=True,

    # Limits
    max_challenges_per_round=3,
    max_interventions_total=5
)

trickster = EvidencePoweredTrickster(config=config)
```

## Detection Mechanisms

### Evidence Quality Analysis

The Trickster analyzes each agent's response for:

| Metric | Weight | Description |
|--------|--------|-------------|
| `citation_density` | 0.25 | Ratio of claims with sources |
| `specificity_score` | 0.25 | Concrete numbers vs vague language |
| `logical_chain_score` | 0.25 | Premise-to-conclusion reasoning |
| `evidence_diversity` | 0.25 | Variety of evidence types |

### Hollow Consensus Detection

Hollow consensus is detected when:

1. **High Convergence** - Agents are semantically agreeing (similarity > 0.7)
2. **Low Quality** - Average evidence quality is below threshold
3. **Quality Variance** - Wide variance suggests superficial agreement

```
Severity = (1 - avg_quality) * convergence_similarity * (1 + quality_variance)
```

### Cross-Proposal Analysis

The Trickster also performs cross-agent analysis:

- **Evidence Gaps** - Claims made by multiple agents without any supporting evidence
- **Echo Chamber** - Agents citing the same limited sources (redundancy > 0.7)
- **Corroboration** - Independent evidence supporting the same conclusion

## Intervention Types

### 1. Challenge Prompts

The most common intervention. Injects a structured challenge:

```markdown
## QUALITY CHALLENGE - Evidence Review Required

The current discussion shows signs of **hollow consensus** -
positions are converging without sufficient evidence backing.

### Specific Challenges:
- Provide specific citations or data sources
- Replace vague language with concrete numbers
- Give real examples that demonstrate your points

### Evidence Gaps by Agent:
- **claude**: Missing citations, specificity
- **gpt4**: Missing reasoning, evidence_diversity

### Before Proceeding:
1. Provide specific citations or data sources
2. Replace vague language with concrete numbers
3. Give real examples that demonstrate your points
4. Explain the logical chain from premise to conclusion

*This challenge was triggered by the Evidence-Powered Trickster system.*
```

### 2. Quality Role Assignment

Assigns the `QUALITY_CHALLENGER` cognitive role to an agent:

```python
role = trickster.get_quality_challenger_assignment(
    agent_name="claude",
    round_num=3
)
# Agent receives special prompt to challenge evidence quality
```

### 3. Evidence Gap Challenges

When cross-proposal analysis finds unsupported claims:

```markdown
## EVIDENCE GAP DETECTED

Multiple agents are making claims **without supporting evidence**.
Before reaching consensus, please address these gaps:

- **Claim by claude, gpt4**: "Microservices improve scalability..."
  → No evidence provided by any agent

### Required Actions:
1. Provide specific sources or data supporting these claims
2. If no evidence exists, reconsider the claim
3. Distinguish between speculation and supported conclusions
```

### 4. Echo Chamber Warnings

When agents cite the same limited sources:

```markdown
## ECHO CHAMBER WARNING

Agents are citing the **same limited evidence** (85% redundancy).

- Unique evidence sources: 3
- Total citations: 15

This suggests agents may be reinforcing each other's views
without independent validation.

### Required Actions:
1. Each agent should seek **independent** evidence sources
2. Consider alternative interpretations of the shared evidence
3. Challenge assumptions that are based on repeated assertions
4. Look for evidence that might **contradict** the emerging consensus
```

### 5. Novelty Challenges

When the NoveltyTracker detects stale proposals:

```markdown
## NOVELTY CHALLENGE - Seek Alternative Perspectives

Your current proposals are **too similar** to ideas already discussed
in previous rounds. The debate risks converging to mediocrity.

### Agents Needing Fresh Perspectives:
- **claude**: Novelty 35% (below threshold)
- **gpt4**: Novelty 42% (below threshold)

### To Increase Novelty:
1. Consider angles you haven't explored yet
2. Challenge assumptions from prior rounds
3. Introduce new evidence or frameworks
4. Play devil's advocate to your own position
5. Think about edge cases or minority viewpoints
```

## Callbacks and Monitoring

### Intervention Callback

```python
def on_intervention(intervention: TricksterIntervention):
    print(f"Trickster intervened at round {intervention.round_num}")
    print(f"Type: {intervention.intervention_type}")
    print(f"Targets: {intervention.target_agents}")
    print(f"Priority: {intervention.priority}")

trickster = EvidencePoweredTrickster(
    on_intervention=on_intervention
)
```

### Alert Callback

```python
def on_alert(alert: HollowConsensusAlert):
    if alert.detected:
        print(f"Hollow consensus detected! Severity: {alert.severity}")
        print(f"Avg quality: {alert.avg_quality}")
        print(f"Reason: {alert.reason}")

trickster = EvidencePoweredTrickster(
    on_alert=on_alert
)
```

### Statistics

```python
stats = trickster.get_stats()
print(f"Total interventions: {stats['total_interventions']}")
print(f"Hollow alerts detected: {stats['hollow_alerts_detected']}")
print(f"Quality per round: {stats['avg_quality_per_round']}")
```

## Integration with Arena

The Trickster integrates with the Arena through the debate protocol:

```python
# In debate rounds phase
if self.protocol.enable_trickster:
    intervention = self.trickster.check_and_intervene(
        responses=round_responses,
        convergence_similarity=convergence_score,
        round_num=current_round
    )

    if intervention:
        if intervention.intervention_type == InterventionType.CHALLENGE_PROMPT:
            # Inject challenge into next round context
            context.append({"role": "system", "content": intervention.challenge_text})

        elif intervention.intervention_type == InterventionType.QUALITY_ROLE:
            # Assign quality challenger role
            for agent_name in intervention.target_agents:
                role = self.trickster.get_quality_challenger_assignment(
                    agent_name, current_round
                )
                self.role_manager.assign(role)

        elif intervention.intervention_type == InterventionType.BREAKPOINT:
            # Pause for human review
            await self.event_emitter.emit("trickster_breakpoint", {
                "reason": intervention.challenge_text,
                "severity": intervention.priority
            })
```

## Best Practices

### Tuning Thresholds

Start conservative and adjust based on your use case:

```python
# For highly technical debates (strict quality)
config = TricksterConfig(
    min_quality_threshold=0.7,
    hollow_detection_threshold=0.4
)

# For brainstorming sessions (more lenient)
config = TricksterConfig(
    min_quality_threshold=0.4,
    hollow_detection_threshold=0.7
)
```

### Avoiding False Positives

- Set appropriate cooldown between interventions
- Use role assignment before challenge prompts
- Enable breakpoints only for high-stakes debates

### Monitoring Quality Over Time

Track evidence quality trends across debates:

```python
quality_trend = []
for debate_result in debate_history:
    stats = debate_result.trickster_stats
    quality_trend.append(stats['avg_quality_per_round'])

# Identify agents consistently flagged
flagged_agents = Counter()
for stats in debate_stats:
    for intervention in stats['interventions']:
        for agent in intervention['targets']:
            flagged_agents[agent] += 1
```

## Related Features

- **Convergence Detection** - `aragora.debate.convergence` - Semantic similarity measurement
- **Evidence Quality** - `aragora.debate.evidence_quality` - Quality scoring system
- **Cognitive Roles** - `aragora.debate.roles` - Role-based debate dynamics
- **Breakpoints** - `aragora.debate.breakpoints` - Human-in-the-loop triggers
