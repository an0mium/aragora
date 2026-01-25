/**
 * Verification Namespace API
 *
 * Provides a namespaced interface for claim and debate verification operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  VerificationResult,
  VerificationStatus,
  VerificationReport,
  VerifyClaimRequest,
} from '../types';

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
 * Interface for the internal client methods used by VerificationAPI.
 */
interface VerificationClientInterface {
  verifyClaim(request: VerifyClaimRequest): Promise<VerificationResult>;
  getVerificationStatus(): Promise<VerificationStatus>;
  verifyDebateConclusion(debateId: string, options?: DebateConclusionVerifyOptions): Promise<VerificationResult>;
  getVerificationReport(debateId: string): Promise<VerificationReport>;
}

/**
 * Verification API namespace.
 *
 * Provides methods for verifying claims and debate conclusions:
 * - Independent claim verification
 * - Debate conclusion verification
 * - Verification reports
 * - System verification status
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Verify a claim
 * const result = await client.verification.verifyClaim({
 *   claim: 'The Earth is approximately 4.5 billion years old',
 *   context: 'Geological science',
 * });
 *
 * // Verify debate conclusion
 * const verified = await client.verification.verifyConclusion(debateId, {
 *   include_evidence: true,
 *   depth: 'deep',
 * });
 *
 * // Get verification report for a debate
 * const report = await client.verification.getReport(debateId);
 * ```
 */
export class VerificationAPI {
  constructor(private client: VerificationClientInterface) {}

  /**
   * Verify a claim independently.
   */
  async verifyClaim(request: VerifyClaimRequest): Promise<VerificationResult> {
    return this.client.verifyClaim(request);
  }

  /**
   * Get the current verification system status.
   */
  async status(): Promise<VerificationStatus> {
    return this.client.getVerificationStatus();
  }

  /**
   * Verify a debate's conclusion.
   */
  async verifyConclusion(
    debateId: string,
    options?: DebateConclusionVerifyOptions
  ): Promise<VerificationResult> {
    return this.client.verifyDebateConclusion(debateId, options);
  }

  /**
   * Get the verification report for a debate.
   */
  async getReport(debateId: string): Promise<VerificationReport> {
    return this.client.getVerificationReport(debateId);
  }
}
