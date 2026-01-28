/**
 * Calibration Namespace API
 *
 * Provides access to calibration leaderboards.
 */

export interface CalibrationLeaderboardEntry {
  agent: string;
  score: number;
  samples?: number;
}

interface CalibrationClientInterface {
  get<T>(path: string): Promise<T>;
}

export class CalibrationAPI {
  constructor(private client: CalibrationClientInterface) {}

  async getLeaderboard(): Promise<{ leaderboard: CalibrationLeaderboardEntry[]; total?: number }> {
    return this.client.get('/api/v1/calibration/leaderboard');
  }
}
