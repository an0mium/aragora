/**
 * Agents Management Example
 *
 * Demonstrates agent operations using the Aragora SDK:
 * - Listing and retrieving agent information
 * - ELO rankings and performance metrics
 * - Team selection for debates
 * - Performance tracking and analytics
 *
 * Usage:
 *   npx ts-node examples/agents-management.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import {
  createClient,
  AragoraError,
  type AragoraConfig,
  type Agent,
} from '@aragora/sdk';

// =============================================================================
// Configuration
// =============================================================================

const config: AragoraConfig = {
  baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
  apiKey: process.env.ARAGORA_API_KEY,
  timeout: 30000,
};

// =============================================================================
// List and Retrieve Agents
// =============================================================================

async function demonstrateAgentListing(): Promise<void> {
  console.log('=== Agent Listing ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // List All Agents
  // -------------------------------------------------------------------------
  console.log('1. List All Agents');
  console.log('-'.repeat(40));

  try {
    const { agents } = await client.agents.list();

    console.log(`Found ${agents.length} agents:\n`);

    for (const agent of agents) {
      console.log(`  ${agent.name}`);
      console.log(`    ELO: ${agent.elo || 'N/A'}`);
      console.log(`    Matches: ${agent.matches || 0}`);
      if (agent.domains?.length) {
        console.log(`    Domains: ${agent.domains.join(', ')}`);
      }
    }
  } catch (error) {
    handleError('List agents', error);
  }

  // -------------------------------------------------------------------------
  // Get Agent Availability
  // -------------------------------------------------------------------------
  console.log('\n2. Agent Availability');
  console.log('-'.repeat(40));

  try {
    const availability = await client.agents.getAvailability();

    console.log(`  Available: ${availability.available.join(', ')}`);
    if (availability.missing?.length) {
      console.log(`  Missing: ${availability.missing.join(', ')}`);
    }
  } catch (error) {
    handleError('Get availability', error);
  }

  // -------------------------------------------------------------------------
  // Get Agent Health
  // -------------------------------------------------------------------------
  console.log('\n3. Agent Health Status');
  console.log('-'.repeat(40));

  try {
    const health = await client.agents.getHealth();

    for (const [agent, status] of Object.entries(health)) {
      const statusInfo = status as { healthy: boolean; latency_ms?: number };
      console.log(`  ${agent}: ${statusInfo.healthy ? 'healthy' : 'unhealthy'}`);
      if (statusInfo.latency_ms) {
        console.log(`    Latency: ${statusInfo.latency_ms}ms`);
      }
    }
  } catch (error) {
    handleError('Get health', error);
  }

  // -------------------------------------------------------------------------
  // Get Specific Agent
  // -------------------------------------------------------------------------
  console.log('\n4. Get Specific Agent');
  console.log('-'.repeat(40));

  try {
    const agent = await client.agents.get('claude');

    console.log(`  Name: ${agent.name}`);
    console.log(`  ELO: ${agent.elo || 'N/A'}`);
    console.log(`  Matches: ${agent.matches || 0}`);
    console.log(`  Wins: ${agent.wins || 0}`);
    console.log(`  Losses: ${agent.losses || 0}`);
    if (agent.calibration_score !== undefined) {
      console.log(`  Calibration: ${(agent.calibration_score * 100).toFixed(1)}%`);
    }
  } catch (error) {
    handleError('Get agent', error);
  }

  // -------------------------------------------------------------------------
  // Get Agent Profile
  // -------------------------------------------------------------------------
  console.log('\n5. Get Agent Profile');
  console.log('-'.repeat(40));

  try {
    const profile = await client.agents.getProfile('claude');

    console.log(`  Name: ${profile.name}`);
    console.log(`  ELO: ${profile.elo || 'N/A'}`);
    console.log(`  Reputation: ${profile.reputation || 'N/A'}`);
    console.log(`  Consistency: ${profile.consistency_score?.toFixed(2) || 'N/A'}`);
    console.log(`  Flip Rate: ${profile.flip_rate?.toFixed(3) || 'N/A'}`);

    if (profile.allies?.length) {
      console.log(`  Allies: ${profile.allies.join(', ')}`);
    }
    if (profile.rivals?.length) {
      console.log(`  Rivals: ${profile.rivals.join(', ')}`);
    }
  } catch (error) {
    handleError('Get profile', error);
  }
}

// =============================================================================
// ELO Rankings
// =============================================================================

async function demonstrateEloRankings(): Promise<void> {
  console.log('\n=== ELO Rankings ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Get Leaderboard
  // -------------------------------------------------------------------------
  console.log('1. Agent Leaderboard');
  console.log('-'.repeat(40));

  try {
    const { agents } = await client.agents.getLeaderboard();

    console.log('Top Agents by ELO:\n');

    let rank = 1;
    for (const agent of agents.slice(0, 10)) {
      const winRate = agent.matches
        ? ((agent.wins || 0) / agent.matches * 100).toFixed(1)
        : 'N/A';

      console.log(`  ${rank}. ${agent.name}`);
      console.log(`     ELO: ${agent.elo || 'N/A'} | Win Rate: ${winRate}%`);
      console.log(`     Matches: ${agent.matches || 0}`);
      rank++;
    }
  } catch (error) {
    handleError('Get leaderboard', error);
  }

  // -------------------------------------------------------------------------
  // Get Agent ELO History
  // -------------------------------------------------------------------------
  console.log('\n2. ELO History');
  console.log('-'.repeat(40));

  try {
    const eloData = await client.agents.getElo('claude');

    console.log(`  Agent: ${eloData.agent}`);
    console.log(`  Current ELO: ${eloData.elo}`);
    console.log(`\n  History (last 5 points):`);

    for (const point of eloData.history.slice(-5)) {
      console.log(`    ${point.date}: ${point.elo}`);
    }
  } catch (error) {
    handleError('Get ELO', error);
  }

  // -------------------------------------------------------------------------
  // Get Rankings with Filters
  // -------------------------------------------------------------------------
  console.log('\n3. Filtered Rankings');
  console.log('-'.repeat(40));

  try {
    const { rankings } = await client.agents.getRankings({
      limit: 10,
      minDebates: 5,
      sortBy: 'win_rate',
      order: 'desc',
    });

    console.log('Agents by Win Rate (min 5 debates):\n');

    for (const ranking of rankings) {
      console.log(`  ${ranking.rank}. ${ranking.agent}: ELO ${ranking.elo}`);
    }
  } catch (error) {
    handleError('Get rankings', error);
  }

  // -------------------------------------------------------------------------
  // Calibration Leaderboard
  // -------------------------------------------------------------------------
  console.log('\n4. Calibration Leaderboard');
  console.log('-'.repeat(40));

  try {
    const { agents } = await client.agents.getCalibrationLeaderboard();

    console.log('Agents by Calibration Score:\n');

    for (const agent of agents.slice(0, 5)) {
      console.log(`  ${agent.name}: ${(agent.score * 100).toFixed(1)}%`);
    }
  } catch (error) {
    handleError('Get calibration leaderboard', error);
  }
}

// =============================================================================
// Team Selection
// =============================================================================

async function demonstrateTeamSelection(): Promise<void> {
  console.log('\n=== Team Selection ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Basic Team Selection
  // -------------------------------------------------------------------------
  console.log('1. Balanced Team Selection');
  console.log('-'.repeat(40));

  try {
    const team = await client.agents.selectTeam(
      'Design a scalable microservices architecture',
      3,
      'balanced'
    );

    console.log(`  Selected agents: ${team.agents.join(', ')}`);
    console.log(`  Total score: ${team.total_score?.toFixed(2) || 'N/A'}`);
    console.log(`  Diversity score: ${team.diversity_score?.toFixed(2) || 'N/A'}`);

    if (team.coverage) {
      console.log('  Coverage:');
      for (const [area, agents] of Object.entries(team.coverage)) {
        console.log(`    ${area}: ${agents.join(', ')}`);
      }
    }
  } catch (error) {
    handleError('Select balanced team', error);
  }

  // -------------------------------------------------------------------------
  // Competitive Team Selection
  // -------------------------------------------------------------------------
  console.log('\n2. Competitive Team Selection');
  console.log('-'.repeat(40));

  try {
    const team = await client.agents.selectTeam(
      'Debate the future of web development',
      4,
      'competitive'
    );

    console.log(`  Selected agents: ${team.agents.join(', ')}`);
    console.log(`  Strategy: Prioritizing high-ELO agents`);
    console.log(`  Total score: ${team.total_score?.toFixed(2) || 'N/A'}`);
  } catch (error) {
    handleError('Select competitive team', error);
  }

  // -------------------------------------------------------------------------
  // Diverse Team Selection
  // -------------------------------------------------------------------------
  console.log('\n3. Diverse Team Selection');
  console.log('-'.repeat(40));

  try {
    const team = await client.agents.selectTeam(
      'Evaluate emerging AI safety frameworks',
      5,
      'diverse'
    );

    console.log(`  Selected agents: ${team.agents.join(', ')}`);
    console.log(`  Strategy: Maximum perspective diversity`);
    console.log(`  Diversity score: ${team.diversity_score?.toFixed(2) || 'N/A'}`);
  } catch (error) {
    handleError('Select diverse team', error);
  }

  // -------------------------------------------------------------------------
  // Specialized Team Selection
  // -------------------------------------------------------------------------
  console.log('\n4. Specialized Team Selection');
  console.log('-'.repeat(40));

  try {
    const team = await client.agents.selectTeam(
      'Analyze security vulnerabilities in distributed systems',
      3,
      'specialized'
    );

    console.log(`  Selected agents: ${team.agents.join(', ')}`);
    console.log(`  Strategy: Domain expertise focus`);
    console.log(`  Total score: ${team.total_score?.toFixed(2) || 'N/A'}`);
  } catch (error) {
    handleError('Select specialized team', error);
  }
}

// =============================================================================
// Performance Tracking
// =============================================================================

async function demonstratePerformanceTracking(): Promise<void> {
  console.log('\n=== Performance Tracking ===\n');

  const client = createClient(config);

  const targetAgent = 'claude';

  // -------------------------------------------------------------------------
  // Get Performance Stats
  // -------------------------------------------------------------------------
  console.log('1. Performance Statistics');
  console.log('-'.repeat(40));

  try {
    const perf = await client.agents.getPerformance(targetAgent);

    console.log(`  Agent: ${perf.agent}`);
    console.log(`  Win Rate: ${(perf.win_rate * 100).toFixed(1)}%`);
    console.log(`  Loss Rate: ${(perf.loss_rate * 100).toFixed(1)}%`);
    console.log(`  Draw Rate: ${(perf.draw_rate * 100).toFixed(1)}%`);
    console.log(`  Total Debates: ${perf.total_debates}`);
    console.log(`  ELO Change (30d): ${perf.elo_change_30d >= 0 ? '+' : ''}${perf.elo_change_30d}`);
    console.log(`  Avg Confidence: ${(perf.avg_confidence * 100).toFixed(1)}%`);
    console.log(`  Avg Round Duration: ${perf.avg_round_duration_ms}ms`);

    if (perf.recent_results?.length) {
      console.log('\n  Recent Results:');
      for (const result of perf.recent_results.slice(0, 3)) {
        console.log(`    ${result.date}: ${result.outcome}`);
      }
    }
  } catch (error) {
    handleError('Get performance', error);
  }

  // -------------------------------------------------------------------------
  // Head-to-Head Statistics
  // -------------------------------------------------------------------------
  console.log('\n2. Head-to-Head Statistics');
  console.log('-'.repeat(40));

  try {
    const h2h = await client.agents.getHeadToHead(targetAgent, 'gpt-4');

    console.log(`  ${h2h.agent} vs ${h2h.opponent}`);
    console.log(`  Total Matchups: ${h2h.total_matchups}`);
    console.log(`  Wins: ${h2h.wins} | Losses: ${h2h.losses} | Draws: ${h2h.draws}`);
    console.log(`  Win Rate: ${(h2h.win_rate * 100).toFixed(1)}%`);
    console.log(`  Avg Margin: ${h2h.avg_margin.toFixed(2)}`);

    if (h2h.domain_breakdown) {
      console.log('\n  By Domain:');
      for (const [domain, stats] of Object.entries(h2h.domain_breakdown)) {
        console.log(`    ${domain}: ${stats.wins}W - ${stats.losses}L`);
      }
    }
  } catch (error) {
    handleError('Get head-to-head', error);
  }

  // -------------------------------------------------------------------------
  // Calibration Data
  // -------------------------------------------------------------------------
  console.log('\n3. Calibration Data');
  console.log('-'.repeat(40));

  try {
    const calibration = await client.agents.getCalibration(targetAgent);

    console.log(`  Agent: ${calibration.agent}`);
    console.log(`  Overall Score: ${(calibration.overall_score * 100).toFixed(1)}%`);
    console.log(`  Confidence Accuracy: ${(calibration.confidence_accuracy * 100).toFixed(1)}%`);
    console.log(`  Sample Size: ${calibration.sample_size}`);

    if (calibration.domain_scores) {
      console.log('\n  By Domain:');
      for (const [domain, score] of Object.entries(calibration.domain_scores)) {
        console.log(`    ${domain}: ${(score * 100).toFixed(1)}%`);
      }
    }
  } catch (error) {
    handleError('Get calibration', error);
  }

  // -------------------------------------------------------------------------
  // Consistency Metrics
  // -------------------------------------------------------------------------
  console.log('\n4. Consistency Metrics');
  console.log('-'.repeat(40));

  try {
    const consistency = await client.agents.getConsistency(targetAgent);

    console.log(`  Agent: ${consistency.agent}`);
    console.log(`  Overall Consistency: ${(consistency.overall_consistency * 100).toFixed(1)}%`);
    console.log(`  Position Stability: ${(consistency.position_stability * 100).toFixed(1)}%`);
    console.log(`  Flip Rate: ${(consistency.flip_rate * 100).toFixed(2)}%`);
    console.log(`  Volatility Index: ${consistency.volatility_index.toFixed(3)}`);
    console.log(`  Sample Size: ${consistency.sample_size}`);

    if (consistency.consistency_by_domain) {
      console.log('\n  By Domain:');
      for (const [domain, score] of Object.entries(consistency.consistency_by_domain)) {
        console.log(`    ${domain}: ${(score * 100).toFixed(1)}%`);
      }
    }
  } catch (error) {
    handleError('Get consistency', error);
  }

  // -------------------------------------------------------------------------
  // Domain Expertise
  // -------------------------------------------------------------------------
  console.log('\n5. Domain Expertise');
  console.log('-'.repeat(40));

  try {
    const { domains } = await client.agents.getDomains(targetAgent);

    console.log(`  ${targetAgent}'s domain ratings:\n`);

    for (const domain of domains.slice(0, 5)) {
      const trendIcon = domain.trend === 'rising' ? '^' : domain.trend === 'falling' ? 'v' : '-';
      console.log(`  ${domain.domain}`);
      console.log(`    ELO: ${domain.elo} (${trendIcon})`);
      console.log(`    Matches: ${domain.matches} | Win Rate: ${(domain.win_rate * 100).toFixed(1)}%`);
    }
  } catch (error) {
    handleError('Get domains', error);
  }
}

// =============================================================================
// Agent Relationships
// =============================================================================

async function demonstrateRelationships(): Promise<void> {
  console.log('\n=== Agent Relationships ===\n');

  const client = createClient(config);

  const targetAgent = 'claude';

  // -------------------------------------------------------------------------
  // Get Agent Network
  // -------------------------------------------------------------------------
  console.log('1. Agent Network');
  console.log('-'.repeat(40));

  try {
    const network = await client.agents.getNetwork(targetAgent);

    console.log(`  Agent: ${network.agent}`);
    console.log(`  Position: ${network.network_position}`);

    if (network.allies?.length) {
      console.log('\n  Allies:');
      for (const ally of network.allies.slice(0, 3)) {
        console.log(`    ${ally.agent}: ${(ally.agreement_rate * 100).toFixed(1)}% agreement`);
        console.log(`      Shared debates: ${ally.shared_debates}`);
      }
    }

    if (network.rivals?.length) {
      console.log('\n  Rivals:');
      for (const rival of network.rivals.slice(0, 3)) {
        console.log(`    ${rival.agent}: ${(rival.disagreement_rate * 100).toFixed(1)}% disagreement`);
        console.log(`      Shared debates: ${rival.shared_debates}`);
      }
    }
  } catch (error) {
    handleError('Get network', error);
  }

  // -------------------------------------------------------------------------
  // Get Specific Relationship
  // -------------------------------------------------------------------------
  console.log('\n2. Specific Relationship');
  console.log('-'.repeat(40));

  try {
    const relationship = await client.agents.getRelationship(targetAgent, 'gpt-4');

    console.log(`  ${relationship.agent_a} <-> ${relationship.agent_b}`);
    console.log(`  Type: ${relationship.relationship_type}`);
    console.log(`  Agreement Rate: ${(relationship.agreement_rate * 100).toFixed(1)}%`);
    console.log(`  Debate Count: ${relationship.debate_count}`);
    console.log(`  Last Interaction: ${relationship.last_interaction}`);

    if (relationship.notable_debates?.length) {
      console.log(`  Notable Debates: ${relationship.notable_debates.slice(0, 3).join(', ')}`);
    }
  } catch (error) {
    handleError('Get relationship', error);
  }

  // -------------------------------------------------------------------------
  // Compare Agents
  // -------------------------------------------------------------------------
  console.log('\n3. Compare Agents');
  console.log('-'.repeat(40));

  try {
    const comparison = await client.agents.compare(['claude', 'gpt-4', 'gemini']);

    console.log(`  Comparing: ${comparison.agents.join(' vs ')}\n`);

    for (const [agent, metrics] of Object.entries(comparison.metrics)) {
      console.log(`  ${agent}:`);
      console.log(`    ELO: ${metrics.elo}`);
      console.log(`    Win Rate: ${(metrics.win_rate * 100).toFixed(1)}%`);
      console.log(`    Consensus Rate: ${(metrics.consensus_rate * 100).toFixed(1)}%`);
      console.log(`    Calibration: ${(metrics.calibration_score * 100).toFixed(1)}%`);
    }

    if (comparison.head_to_head?.length) {
      console.log('\n  Head-to-Head:');
      for (const h2h of comparison.head_to_head) {
        console.log(`    ${h2h.agent_a} vs ${h2h.agent_b}: ${h2h.wins_a}W-${h2h.wins_b}L-${h2h.draws}D`);
      }
    }

    if (comparison.recommendation) {
      console.log(`\n  Recommendation: ${comparison.recommendation}`);
    }
  } catch (error) {
    handleError('Compare agents', error);
  }
}

// =============================================================================
// Recent Activity
// =============================================================================

async function demonstrateRecentActivity(): Promise<void> {
  console.log('\n=== Recent Activity ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Recent Matches
  // -------------------------------------------------------------------------
  console.log('1. Recent Matches');
  console.log('-'.repeat(40));

  try {
    const { matches } = await client.agents.getRecentMatches({ limit: 5 });

    console.log(`  Last ${matches.length} matches:\n`);

    for (const match of matches as Array<{
      debate_id: string;
      agents: string[];
      winner?: string;
      created_at: string;
    }>) {
      console.log(`  ${match.debate_id}`);
      console.log(`    Agents: ${match.agents.join(' vs ')}`);
      console.log(`    Winner: ${match.winner || 'Draw'}`);
      console.log(`    Date: ${match.created_at}`);
    }
  } catch (error) {
    handleError('Get recent matches', error);
  }

  // -------------------------------------------------------------------------
  // Recent Position Flips
  // -------------------------------------------------------------------------
  console.log('\n2. Recent Position Flips');
  console.log('-'.repeat(40));

  try {
    const { flips } = await client.agents.getRecentFlips({ limit: 5 });

    console.log(`  Recent flips:\n`);

    for (const flip of flips as Array<{
      agent: string;
      topic: string;
      original_position: string;
      new_position: string;
      timestamp: string;
    }>) {
      console.log(`  ${flip.agent} on "${flip.topic}"`);
      console.log(`    From: ${flip.original_position.substring(0, 50)}...`);
      console.log(`    To: ${flip.new_position.substring(0, 50)}...`);
      console.log(`    When: ${flip.timestamp}`);
    }
  } catch (error) {
    handleError('Get recent flips', error);
  }

  // -------------------------------------------------------------------------
  // Flip Summary
  // -------------------------------------------------------------------------
  console.log('\n3. Flip Summary');
  console.log('-'.repeat(40));

  try {
    const summary = await client.agents.getFlipsSummary();

    const summaryData = summary as {
      total_flips: number;
      flips_today: number;
      most_volatile_agent: string;
      average_flip_rate: number;
    };

    console.log(`  Total flips: ${summaryData.total_flips}`);
    console.log(`  Flips today: ${summaryData.flips_today}`);
    console.log(`  Most volatile: ${summaryData.most_volatile_agent}`);
    console.log(`  Average flip rate: ${(summaryData.average_flip_rate * 100).toFixed(2)}%`);
  } catch (error) {
    handleError('Get flip summary', error);
  }
}

// =============================================================================
// Error Handling
// =============================================================================

function handleError(context: string, error: unknown): void {
  if (error instanceof AragoraError) {
    console.error(`  ${context}: [${error.code}] ${error.message}`);
  } else if (error instanceof Error) {
    console.error(`  ${context}: ${error.message}`);
  } else {
    console.error(`  ${context}:`, error);
  }
}

// =============================================================================
// Run Examples
// =============================================================================

async function main(): Promise<void> {
  if (!process.env.ARAGORA_API_KEY) {
    console.warn('Warning: ARAGORA_API_KEY not set. Some operations may fail.\n');
  }

  try {
    await demonstrateAgentListing();
    await demonstrateEloRankings();
    await demonstrateTeamSelection();
    await demonstratePerformanceTracking();
    await demonstrateRelationships();
    await demonstrateRecentActivity();

    console.log('\n=== All agent management examples completed ===');
  } catch (error) {
    if (error instanceof AragoraError) {
      console.error('\nFatal error:', error.message);
    } else {
      console.error('\nFatal error:', error);
    }
    process.exit(1);
  }
}

main();
