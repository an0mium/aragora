/**
 * Voice Namespace API
 *
 * Provides endpoints for voice/TTS integration including
 * voice sessions, speech-to-text, and debate audio synthesis.
 */

import type { AragoraClient } from '../client';

/** Voice session state */
export interface VoiceSession {
  id: string;
  status: 'active' | 'completed' | 'failed';
  caller_number?: string;
  debate_id?: string;
  started_at: string;
  ended_at?: string;
  duration_seconds?: number;
}

/** TTS synthesis request */
export interface SynthesizeRequest {
  text: string;
  voice?: string;
  format?: 'mp3' | 'wav' | 'ogg';
  speed?: number;
}

/** TTS synthesis result */
export interface SynthesizeResult {
  audio_url: string;
  duration_seconds: number;
  format: string;
  voice: string;
}

/** Voice configuration */
export interface VoiceConfig {
  enabled: boolean;
  default_voice: string;
  available_voices: string[];
  tts_provider: string;
}

/**
 * Voice namespace for TTS and voice integration.
 *
 * @example
 * ```typescript
 * const config = await client.voice.getConfig();
 * const audio = await client.voice.synthesize({ text: 'Hello world' });
 * ```
 */
export class VoiceNamespace {
  constructor(private client: AragoraClient) {}

  /** Get voice configuration. */
  async getConfig(): Promise<VoiceConfig> {
    return this.client.request<VoiceConfig>('GET', '/api/v1/voice/config');
  }

  /** List voice sessions. */
  async listSessions(options?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<VoiceSession[]> {
    const response = await this.client.request<{ sessions: VoiceSession[] }>(
      'GET',
      '/api/v1/voice/sessions',
      { params: options }
    );
    return response.sessions;
  }

  /** Get a voice session by ID. */
  async getSession(sessionId: string): Promise<VoiceSession> {
    return this.client.request<VoiceSession>(
      'GET',
      `/api/v1/voice/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /** Synthesize text to speech. */
  async synthesize(request: SynthesizeRequest): Promise<SynthesizeResult> {
    return this.client.request<SynthesizeResult>(
      'POST',
      '/api/v1/voice/synthesize',
      { body: request }
    );
  }

  /** Synthesize a debate to audio. */
  async synthesizeDebate(
    debateId: string,
    options?: { voice?: string; format?: string }
  ): Promise<SynthesizeResult> {
    return this.client.request<SynthesizeResult>(
      'POST',
      `/api/v1/voice/debates/${encodeURIComponent(debateId)}/synthesize`,
      { body: options }
    );
  }
}
