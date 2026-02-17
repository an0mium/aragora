/**
 * Verification Namespace API
 *
 * Provides a namespaced interface for claim and debate verification operations.
 */

import type { AragoraClient } from '../client';

/**
 * Verification result.
 */
export interface VerificationResult {
  verified: boolean;
  confidence: number;
  evidence?: Array<{ source: string; relevance: number; excerpt: string }>;
  counterfactuals?: Array<{ claim: string; likelihood: number }>;
}

/**
 * Verification status.
 */
export interface VerificationStatus {
  available: boolean;
  backends: string[];
  last_check?: string;
}

/**
 * Verification report.
 */
export interface VerificationReport {
  debate_id: string;
  verified_claims: number;
  unverified_claims: number;
  confidence_score: number;
  details: Array<{
    claim: string;
    verified: boolean;
    confidence: number;
  }>;
}

/**
 * Verify claim request.
 */
export interface VerifyClaimRequest {
  claim: string;
  context?: string;
  sources?: string[];
}

/**
 * Options for debate conclusion verification.
 */
export interface DebateConclusionVerifyOptions {
  /** Include detailed evidence */
  include_evidence?: boolean;
  /** Include counterfactual analysis */
  include_counterfactuals?: boolean;
  /** Verification depth level */
  depth?: 'shallow' | 'standard' | 'deep';
}

/**
 * Verification API namespace.
 *
 * Provides methods for verifying claims and debate conclusions:
 * - Independent claim verification
 * - System verification status
 */
export class VerificationAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Get verification system status.
   * @route GET /api/v1/verification/status
   */
  async getStatus(): Promise<VerificationStatus> {
    return this.client.request('GET', '/api/v1/verification/status') as Promise<VerificationStatus>;
  }

  /**
   * Formally verify a claim.
   * @route POST /api/v1/verification/formal-verify
   */
  async formalVerify(request: VerifyClaimRequest): Promise<VerificationResult> {
    return this.client.request('POST', '/api/v1/verification/formal-verify', {
      body: request,
    }) as Promise<VerificationResult>;
  }
}
