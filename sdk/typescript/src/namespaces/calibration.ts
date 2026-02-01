/**
 * Calibration Namespace API
 *
 * Provides access to agent calibration leaderboards, curves, and summaries.
 */

export interface CalibrationLeaderboardEntry {
  agent: string;
  score: number;
  samples?: number;
  rank?: number;
}

export interface CalibrationVisualization {
  agents: Array<{
    name: string;
    calibration_curve: Array<{ confidence: number; accuracy: number; count: number }>;
    overall_score: number;
  }>;
}

export interface CalibrationCurve {
  agent: string;
  data_points: Array<{
    confidence: number;
    accuracy: number;
    sample_count: number;
  }>;
  overall_score: number;
  total_predictions: number;
}

export interface CalibrationSummary {
  agent: string;
  calibration_score: number;
  overconfidence_bias: number;
  underconfidence_bias: number;
  total_predictions: number;
  accuracy: number;
  last_updated?: string;
}

export interface AgentCalibration {
  agent: string;
  scores: Record<string, number>;
  history?: Array<{ timestamp: string; score: number }>;
}

interface CalibrationClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class CalibrationAPI {
  constructor(private client: CalibrationClientInterface) {}

  /**
   * Get the calibration leaderboard.
   */
  async getLeaderboard(params?: {
    limit?: number;
  }): Promise<{ leaderboard: CalibrationLeaderboardEntry[]; total?: number }> {
    return this.client.request('GET', '/api/v1/calibration/leaderboard', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get calibration visualization data for all agents.
   */
  async getVisualization(): Promise<CalibrationVisualization> {
    return this.client.request('GET', '/api/v1/calibration/visualization');
  }

  /**
   * Get the calibration curve for a specific agent.
   */
  async getCurve(agentName: string): Promise<CalibrationCurve> {
    return this.client.request('GET', `/api/v1/agent/${agentName}/calibration-curve`);
  }

  /**
   * Get the calibration summary for a specific agent.
   */
  async getSummary(agentName: string): Promise<CalibrationSummary> {
    return this.client.request('GET', `/api/v1/agent/${agentName}/calibration-summary`);
  }

  /**
   * Get general calibration scores for a specific agent.
   */
  async getAgentCalibration(agentName: string): Promise<AgentCalibration> {
    return this.client.request('GET', `/api/v1/agent/${agentName}/calibration`);
  }
}
