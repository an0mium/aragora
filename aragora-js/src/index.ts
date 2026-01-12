/**
 * Aragora SDK
 *
 * TypeScript/JavaScript SDK for the Aragora multi-agent debate framework.
 *
 * @example
 * ```typescript
 * import { AragoraClient, streamDebate } from '@aragora/sdk';
 *
 * // Create client
 * const client = new AragoraClient({
 *   baseUrl: 'http://localhost:8080',
 *   apiKey: 'your-api-key',
 * });
 *
 * // Run a debate
 * const debate = await client.debates.run({
 *   task: 'Should we use microservices?',
 *   agents: ['anthropic-api', 'openai-api'],
 * });
 * console.log('Consensus:', debate.consensus?.conclusion);
 *
 * // Stream debate events
 * const stream = streamDebate('http://localhost:8080', debate.debate_id);
 * for await (const event of stream) {
 *   console.log(event.type, event.data);
 * }
 *
 * // Verify a claim
 * const result = await client.verification.verify({
 *   claim: 'All primes > 2 are odd',
 *   backend: 'z3',
 * });
 * console.log('Valid:', result.status === 'valid');
 *
 * // Graph debate with branching
 * const graphResult = await client.graphDebates.create({
 *   task: 'Design a distributed cache',
 *   max_branches: 5,
 * });
 *
 * // Matrix debate with scenarios
 * const matrixResult = await client.matrixDebates.create({
 *   task: 'Evaluate database options',
 *   scenarios: [
 *     { name: 'read_heavy', parameters: { read_ratio: 0.9 } },
 *     { name: 'write_heavy', parameters: { read_ratio: 0.1 } },
 *   ],
 * });
 * ```
 *
 * @packageDocumentation
 */

// Client
export { AragoraClient, AragoraError } from './client';
export { default } from './client';

// WebSocket
export { DebateStream, streamDebate } from './websocket';

// Types
export * from './types';
