/**
 * Training Namespace API
 *
 * Provides model training data export and job management:
 * - SFT (Supervised Fine-Tuning) data export
 * - DPO (Direct Preference Optimization) data export
 * - Gauntlet adversarial data export
 * - Training job management and metrics
 */

/**
 * Export format types.
 */
export type ExportFormat = 'json' | 'jsonl';

/**
 * Export types.
 */
export type ExportType = 'sft' | 'dpo' | 'gauntlet';

/**
 * Training job status.
 */
export type JobStatus = 'pending' | 'training' | 'completed' | 'failed' | 'cancelled';

/**
 * Gauntlet persona types.
 */
export type GauntletPersona = 'gdpr' | 'hipaa' | 'ai_act' | 'all';

/**
 * SFT export parameters.
 */
export interface SFTExportParams {
  /** Minimum confidence threshold (0.0-1.0, default 0.7) */
  min_confidence?: number;
  /** Minimum success rate threshold (0.0-1.0, default 0.6) */
  min_success_rate?: number;
  /** Maximum records to export (default 1000) */
  limit?: number;
  /** Offset for pagination (default 0) */
  offset?: number;
  /** Include critique data (default true) */
  include_critiques?: boolean;
  /** Include pattern data (default true) */
  include_patterns?: boolean;
  /** Include debate data (default true) */
  include_debates?: boolean;
  /** Export format (default json) */
  format?: ExportFormat;
}

/**
 * DPO export parameters.
 */
export interface DPOExportParams {
  /** Minimum confidence difference between chosen/rejected (0.0-1.0, default 0.1) */
  min_confidence_diff?: number;
  /** Maximum records to export (default 500) */
  limit?: number;
  /** Export format */
  format?: ExportFormat;
}

/**
 * Gauntlet export parameters.
 */
export interface GauntletExportParams {
  /** Persona type to export (default all) */
  persona?: GauntletPersona;
  /** Minimum severity threshold */
  min_severity?: number;
  /** Maximum records to export */
  limit?: number;
  /** Export format */
  format?: ExportFormat;
}

/**
 * SFT training record.
 */
export interface SFTRecord {
  instruction: string;
  response: string;
  metadata: {
    source: 'debate' | 'pattern' | 'critique';
    confidence: number;
    debate_id?: string;
  };
}

/**
 * DPO training record.
 */
export interface DPORecord {
  prompt: string;
  chosen: string;
  rejected: string;
  metadata: {
    chosen_confidence: number;
    rejected_confidence: number;
    confidence_diff: number;
  };
}

/**
 * Training export result.
 */
export interface TrainingExportResult {
  export_type: ExportType;
  total_records: number;
  parameters: Record<string, unknown>;
  exported_at: string;
  format: ExportFormat;
  records?: SFTRecord[] | DPORecord[] | unknown[];
  data?: string; // JSONL format as string
}

/**
 * Training statistics.
 */
export interface TrainingStats {
  total_debates: number;
  total_patterns: number;
  total_critiques: number;
  sft_eligible: number;
  dpo_eligible: number;
  gauntlet_eligible: number;
  last_updated: string;
}

/**
 * Training formats info.
 */
export interface TrainingFormats {
  supported_formats: string[];
  sft_schema: Record<string, unknown>;
  dpo_schema: Record<string, unknown>;
  gauntlet_schema: Record<string, unknown>;
}

/**
 * Training job.
 */
export interface TrainingJob {
  id: string;
  vertical?: string;
  status: JobStatus;
  base_model?: string;
  adapter_name?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  training_data_examples: number;
  training_data_debates?: number;
  error_message?: string;
}

/**
 * Training job details.
 */
export interface TrainingJobDetails extends TrainingJob {
  config?: Record<string, unknown>;
  hyperparameters?: Record<string, unknown>;
  checkpoints?: string[];
}

/**
 * List jobs parameters.
 */
export interface ListJobsParams {
  limit?: number;
  offset?: number;
  status?: JobStatus;
  vertical?: string;
}

/**
 * Complete job data.
 */
export interface CompleteJobData {
  final_loss?: number;
  elo_rating?: number;
  win_rate?: number;
  vertical_accuracy?: number;
  artifacts?: Record<string, string>;
}

/**
 * Training metrics.
 */
export interface TrainingMetrics {
  job_id: string;
  status: JobStatus;
  training_data_examples: number;
  training_data_debates?: number;
  final_loss?: number;
  elo_rating?: number;
  win_rate?: number;
  vertical_accuracy?: number;
  metrics_history?: Array<{
    step: number;
    loss: number;
    timestamp: string;
  }>;
}

/**
 * Training artifacts.
 */
export interface TrainingArtifacts {
  job_id: string;
  adapter_path?: string;
  config_path?: string;
  metrics_path?: string;
  checkpoints: string[];
  total_size_bytes: number;
}

/**
 * Client interface for training operations.
 */
interface TrainingClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Training API namespace.
 *
 * Provides methods for exporting training data and managing training jobs:
 * - Export SFT, DPO, and Gauntlet data for model training
 * - Manage training jobs lifecycle
 * - Retrieve training metrics and artifacts
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Export SFT training data
 * const sftData = await client.training.exportSFT({
 *   min_confidence: 0.8,
 *   limit: 5000,
 *   format: 'jsonl',
 * });
 *
 * // Export DPO preference pairs
 * const dpoData = await client.training.exportDPO({
 *   min_confidence_diff: 0.2,
 * });
 *
 * // Manage training jobs
 * const jobs = await client.training.listJobs({ status: 'completed' });
 * ```
 */
export class TrainingAPI {
  constructor(private client: TrainingClientInterface) {}

  // =========================================================================
  // Data Export
  // =========================================================================

  /**
   * Export SFT (Supervised Fine-Tuning) training data.
   */
  async exportSFT(params?: SFTExportParams): Promise<TrainingExportResult> {
    return this.client.request('POST', '/api/v1/training/export/sft', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Export DPO (Direct Preference Optimization) training data.
   */
  async exportDPO(params?: DPOExportParams): Promise<TrainingExportResult> {
    return this.client.request('POST', '/api/v1/training/export/dpo', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Export Gauntlet adversarial training data.
   */
  async exportGauntlet(params?: GauntletExportParams): Promise<TrainingExportResult> {
    return this.client.request('POST', '/api/v1/training/export/gauntlet', {
      params: params as Record<string, unknown>,
    });
  }

  // =========================================================================
  // Statistics and Formats
  // =========================================================================

  /**
   * Get training data statistics.
   */
  async getStats(): Promise<{ stats: TrainingStats }> {
    return this.client.request('GET', '/api/v1/training/stats');
  }

  /**
   * Get supported training formats and schemas.
   */
  async getFormats(): Promise<{ formats: TrainingFormats }> {
    return this.client.request('GET', '/api/v1/training/formats');
  }

  // =========================================================================
  // Job Management
  // =========================================================================

  /**
   * List training jobs.
   */
  async listJobs(
    params?: ListJobsParams
  ): Promise<{ jobs: TrainingJob[]; total: number; limit: number; offset: number }> {
    return this.client.request('GET', '/api/v1/training/jobs', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get training job details.
   */
  async getJob(jobId: string): Promise<TrainingJobDetails> {
    return this.client.request('GET', `/api/v1/training/jobs/${jobId}`);
  }

  /**
   * Cancel a training job.
   */
  async cancelJob(jobId: string): Promise<{ success: boolean; message: string }> {
    return this.client.request('DELETE', `/api/v1/training/jobs/${jobId}`);
  }

  /**
   * Export training job data.
   */
  async exportJobData(
    jobId: string
  ): Promise<{ success: boolean; examples_exported: number; export_path?: string }> {
    return this.client.request('POST', `/api/v1/training/jobs/${jobId}/export`);
  }

  /**
   * Start a training job.
   */
  async startJob(
    jobId: string
  ): Promise<{ success: boolean; training_job_id: string; message: string }> {
    return this.client.request('POST', `/api/v1/training/jobs/${jobId}/start`);
  }

  /**
   * Mark a training job as complete (typically called by webhook).
   */
  async completeJob(
    jobId: string,
    data?: CompleteJobData
  ): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', `/api/v1/training/jobs/${jobId}/complete`, {
      json: data,
    });
  }

  /**
   * Get training job metrics.
   */
  async getJobMetrics(jobId: string): Promise<TrainingMetrics> {
    return this.client.request('GET', `/api/v1/training/jobs/${jobId}/metrics`);
  }

  /**
   * Get training job artifacts.
   */
  async getJobArtifacts(jobId: string): Promise<TrainingArtifacts> {
    return this.client.request('GET', `/api/v1/training/jobs/${jobId}/artifacts`);
  }
}
