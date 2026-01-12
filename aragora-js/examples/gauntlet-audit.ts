/**
 * Gauntlet Security Audit Example
 *
 * Demonstrates how to run a comprehensive security audit using the Gauntlet API.
 * The Gauntlet stress-tests AI systems with adversarial probes and red-team scenarios.
 */

import { AragoraClient } from '../src';

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('Starting Gauntlet security audit...\n');

  // Run a comprehensive security gauntlet
  const receipt = await client.gauntlet.runAndWait({
    target: 'Your AI System Description',
    playbook: 'security-red-team',
    config: {
      intensity: 'thorough', // 'quick' | 'standard' | 'thorough' | 'exhaustive'
      categories: [
        'prompt-injection',
        'jailbreak',
        'data-extraction',
        'boundary-testing',
        'role-confusion',
      ],
      max_probes: 50,
      parallel_probes: 5,
    },
    personas: [
      { name: 'security-researcher', role: 'adversary' },
      { name: 'compliance-auditor', role: 'validator' },
      { name: 'devil-advocate', role: 'critic' },
    ],
  });

  // Display the receipt
  console.log('=== Gauntlet Receipt ===');
  console.log(`ID: ${receipt.gauntlet_id}`);
  console.log(`Status: ${receipt.status}`);
  console.log(`Duration: ${receipt.duration_seconds}s`);
  console.log(`Total Probes: ${receipt.total_probes}`);

  // Display findings by severity
  console.log('\n=== Findings by Severity ===');
  const findings = receipt.findings || {};
  console.log(`Critical: ${findings.critical?.length || 0}`);
  console.log(`High: ${findings.high?.length || 0}`);
  console.log(`Medium: ${findings.medium?.length || 0}`);
  console.log(`Low: ${findings.low?.length || 0}`);

  // Display critical findings
  if (findings.critical?.length) {
    console.log('\n=== Critical Findings ===');
    for (const finding of findings.critical) {
      console.log(`\n[${finding.category}] ${finding.title}`);
      console.log(`  Description: ${finding.description}`);
      console.log(`  Recommendation: ${finding.recommendation}`);
    }
  }

  // Display overall score
  console.log('\n=== Security Score ===');
  console.log(`Overall: ${receipt.score?.overall || 'N/A'}/100`);
  console.log(`Robustness: ${receipt.score?.robustness || 'N/A'}/100`);
  console.log(`Compliance: ${receipt.score?.compliance || 'N/A'}/100`);

  // Export receipt
  console.log(`\nFull receipt available at: /api/gauntlet/${receipt.gauntlet_id}/receipt`);
}

main().catch(console.error);
