/**
 * Playground Namespace API
 *
 * Provides methods for the interactive debate playground:
 * - Create playground debates
 * - Stream live debates
 * - Cost estimation
 * - Status and TTS
 */

import type { AragoraClient } from '../client';

/**
 * Playground API namespace.
 */
export class PlaygroundAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Create a playground debate.
   * @route POST /api/playground/debate
   */
  async createDebate(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/debate', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a live-streaming playground debate.
   * @route POST /api/playground/debate/live
   */
  async createLiveDebate(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/debate/live', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get cost estimate for a live playground debate before starting it.
   * @route POST /api/playground/debate/live/cost-estimate
   */
  async estimateLiveCost(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/debate/live/cost-estimate', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get playground system status.
   * @route GET /api/playground/status
   */
  async getStatus(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/playground/status') as Promise<Record<string, unknown>>;
  }

  /**
   * Convert debate text to speech audio.
   * @route POST /api/playground/tts
   */
  async textToSpeech(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/tts', {
      body,
    }) as Promise<Record<string, unknown>>;
  }
}
