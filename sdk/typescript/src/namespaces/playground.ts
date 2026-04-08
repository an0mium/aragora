/**
 * Playground Namespace API
 *
 * Provides methods for the interactive debate playground:
 * - Create playground debates
 * - Assess landing-page questions before debate
 * - Record landing telemetry and feedback
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
   * Assess whether a landing-page question is ready for debate.
   * @route POST /api/playground/assess
   */
  async assessQuestion(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/assess', {
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
   * Record bounded landing telemetry.
   * @route POST /api/playground/landing/events
   */
  async recordLandingEvent(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/landing/events', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Summarize recent landing telemetry.
   * @route GET /api/playground/landing/events/summary
   */
  async getLandingEventSummary(options?: {
    window?: number;
    limit?: number;
  }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/playground/landing/events/summary', {
      params: {
        ...(options?.window !== undefined ? { window: options.window } : {}),
        ...(options?.limit !== undefined ? { limit: options.limit } : {}),
      },
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Submit a bounded landing wrong-answer report.
   * @route POST /api/playground/landing/feedback
   */
  async submitLandingFeedback(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/landing/feedback', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * List recent landing wrong-answer reports.
   * @route GET /api/playground/landing/feedback
   */
  async listLandingFeedback(options?: {
    window?: number;
    limit?: number;
  }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/playground/landing/feedback', {
      params: {
        ...(options?.window !== undefined ? { window: options.window } : {}),
        ...(options?.limit !== undefined ? { limit: options.limit } : {}),
      },
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Update review state for a landing feedback report.
   * @route POST /api/playground/landing/feedback/review
   */
  async reviewLandingFeedback(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/playground/landing/feedback/review', {
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
