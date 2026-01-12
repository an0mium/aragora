/**
 * Error Handling Example
 *
 * Demonstrates robust error handling patterns with the Aragora SDK.
 */

import { AragoraClient, AragoraError } from '../src';

const BASE_URL = process.env.ARAGORA_API_URL || 'http://localhost:8080';

/**
 * Example 1: Basic try-catch with AragoraError
 */
async function basicErrorHandling() {
  console.log('=== Basic Error Handling ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  try {
    // This might fail if server is down or debate doesn't exist
    const debate = await client.debates.get('non-existent-id');
    console.log('Debate:', debate);
  } catch (error) {
    if (error instanceof AragoraError) {
      console.log('Aragora API Error:');
      console.log(`  Code: ${error.code}`);
      console.log(`  Message: ${error.message}`);
      console.log(`  Status: ${error.status}`);

      // Handle specific error codes
      switch (error.code) {
        case 'NOT_FOUND':
          console.log('  -> Debate does not exist');
          break;
        case 'UNAUTHORIZED':
          console.log('  -> Check your API key');
          break;
        case 'RATE_LIMITED':
          console.log('  -> Wait and retry');
          break;
        default:
          console.log('  -> Unexpected error');
      }
    } else {
      // Network errors, etc.
      console.log('Unknown error:', error);
    }
  }
}

/**
 * Example 2: Retry with exponential backoff
 */
async function retryWithBackoff<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;

      // Check if error is retryable
      const isRetryable =
        error instanceof AragoraError &&
        (error.code === 'RATE_LIMITED' ||
          error.code === 'SERVICE_UNAVAILABLE' ||
          (error.status && error.status >= 500));

      if (!isRetryable || attempt === maxRetries) {
        throw error;
      }

      // Exponential backoff with jitter
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
      console.log(`  Retry ${attempt + 1}/${maxRetries} after ${delay.toFixed(0)}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

async function retryExample() {
  console.log('\n=== Retry with Backoff ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  try {
    const debate = await retryWithBackoff(
      () =>
        client.debates.run({
          task: 'Test debate for retry example',
          agents: ['claude-sonnet', 'gpt-4'],
          max_rounds: 1,
        }),
      3,
      1000
    );

    console.log('Debate completed:', debate.debate_id);
  } catch (error) {
    console.log('Failed after retries:', (error as Error).message);
  }
}

/**
 * Example 3: Timeout handling
 */
async function timeoutExample() {
  console.log('\n=== Timeout Handling ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
    timeout: 5000, // 5 second global timeout
  });

  try {
    // Long-running debate with per-request timeout
    const debate = await client.debates.run(
      {
        task: 'Complex topic that might take a while',
        agents: ['claude-sonnet', 'gpt-4'],
        max_rounds: 5,
      },
      { timeout: 30000 } // 30 second timeout for this request
    );

    console.log('Completed:', debate.debate_id);
  } catch (error) {
    if (error instanceof AragoraError && error.code === 'TIMEOUT') {
      console.log('Request timed out - debate may still be running on server');
      console.log('Consider polling for status instead');
    } else {
      throw error;
    }
  }
}

/**
 * Example 4: Validation error handling
 */
async function validationExample() {
  console.log('\n=== Validation Errors ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  try {
    // Invalid request - no agents specified
    await client.debates.create({
      task: '', // Empty task
      agents: [], // No agents
      max_rounds: -1, // Invalid rounds
    });
  } catch (error) {
    if (error instanceof AragoraError && error.code === 'VALIDATION_ERROR') {
      console.log('Validation failed:');
      console.log(`  Message: ${error.message}`);
      console.log(`  Details: ${error.details || 'none'}`);

      // Provide user-friendly feedback
      if (error.message.includes('task')) {
        console.log('  -> Please provide a debate topic');
      }
      if (error.message.includes('agents')) {
        console.log('  -> At least 2 agents are required');
      }
    } else {
      throw error;
    }
  }
}

/**
 * Example 5: Graceful degradation
 */
async function gracefulDegradation() {
  console.log('\n=== Graceful Degradation ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Try to get health, fall back to basic connectivity check
  let serverAvailable = false;

  try {
    const health = await client.health();
    serverAvailable = health.status === 'healthy';
    console.log('Server health:', health.status);
  } catch {
    console.log('Health check failed, trying basic ping...');

    try {
      // Try a simpler endpoint
      await client.agents.list();
      serverAvailable = true;
      console.log('Server is reachable (health endpoint may be degraded)');
    } catch {
      console.log('Server unreachable');
    }
  }

  if (!serverAvailable) {
    console.log('\nFalling back to offline mode...');
    console.log('(In a real app, you might use cached data or queue requests)');
    return;
  }

  // Continue with normal operation
  console.log('\nProceeding with online operation...');
}

/**
 * Example 6: Abort controller for cancellation
 */
async function cancellationExample() {
  console.log('\n=== Request Cancellation ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Create an abort controller
  const controller = new AbortController();

  // Set up cancellation after 2 seconds
  const timeoutId = setTimeout(() => {
    console.log('Cancelling request...');
    controller.abort();
  }, 2000);

  try {
    const debate = await client.debates.run(
      {
        task: 'A debate that might get cancelled',
        agents: ['claude-sonnet', 'gpt-4'],
        max_rounds: 5,
      },
      { signal: controller.signal }
    );

    clearTimeout(timeoutId);
    console.log('Completed before cancellation:', debate.debate_id);
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('Request was cancelled by user');
      console.log('(Debate may still be running on server - check status)');
    } else {
      throw error;
    }
  }
}

/**
 * Example 7: Error aggregation for batch operations
 */
async function batchErrorHandling() {
  console.log('\n=== Batch Error Handling ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  const tasks = [
    'Should we use TypeScript?',
    'Is GraphQL better than REST?',
    '', // Invalid - will fail
    'What is the best CI/CD tool?',
  ];

  const results: Array<{ task: string; success: boolean; debateId?: string; error?: string }> = [];

  // Process all tasks, collecting errors
  await Promise.all(
    tasks.map(async (task, index) => {
      try {
        const debate = await client.debates.create({
          task,
          agents: ['claude-sonnet', 'gpt-4'],
        });

        results[index] = {
          task,
          success: true,
          debateId: debate.debate_id,
        };
      } catch (error) {
        results[index] = {
          task: task || '(empty)',
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    })
  );

  // Report results
  console.log('Batch Results:');
  for (const result of results) {
    if (result.success) {
      console.log(`  [OK] "${result.task}" -> ${result.debateId}`);
    } else {
      console.log(`  [FAIL] "${result.task}" -> ${result.error}`);
    }
  }

  const successCount = results.filter((r) => r.success).length;
  console.log(`\nTotal: ${successCount}/${tasks.length} succeeded`);
}

// Run all examples
async function main() {
  await basicErrorHandling();
  await retryExample();
  await timeoutExample();
  await validationExample();
  await gracefulDegradation();
  await cancellationExample();
  await batchErrorHandling();
}

main().catch(console.error);
