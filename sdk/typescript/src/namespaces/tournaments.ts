/**
 * Tournaments Namespace API
 *
 * Provides a namespaced interface for tournament management operations
 * including brackets, matches, standings, and results.
 *
 * @module tournaments
 */

import type {
  Tournament,
  TournamentStandings,
  TournamentBracket,
  TournamentMatch,
  CreateTournamentRequest,
  PaginationParams,
} from '../types';

// =============================================================================
// Tournament-specific Types (matching Python SDK)
// =============================================================================

/**
 * Tournament format types.
 */
export type TournamentFormat = 'single_elimination' | 'double_elimination' | 'round_robin' | 'swiss';

/**
 * Tournament status types.
 */
export type TournamentStatus = 'pending' | 'active' | 'completed' | 'cancelled';

/**
 * Match status types.
 */
export type MatchStatus = 'pending' | 'in_progress' | 'completed' | 'bye';

/**
 * Match result submission.
 */
export interface MatchResultSubmission {
  /** Winner agent name */
  winner: string;
  /** Loser agent name */
  loser: string;
  /** Optional match score */
  score?: { winner: number; loser: number };
  /** Optional match notes */
  notes?: string;
}

/**
 * Tournament list parameters.
 */
export interface TournamentListParams extends PaginationParams {
  /** Filter by tournament status */
  status?: TournamentStatus;
}

/**
 * Match list parameters.
 */
export interface MatchListParams {
  /** Filter by round number */
  round?: number;
  /** Filter by match status */
  status?: MatchStatus | string;
}

/**
 * Tournament advancement result.
 */
export interface TournamentAdvanceResult {
  /** Whether advancement was successful */
  advanced: boolean;
  /** The next round number */
  next_round: number;
  /** Matches created for the next round */
  matches_created?: number;
  /** Message about the advancement */
  message?: string;
}

/**
 * Tournament start result.
 */
export interface TournamentStartResult {
  /** Whether the tournament started successfully */
  started: boolean;
  /** Updated tournament status */
  status: TournamentStatus;
  /** Number of initial matches created */
  matches_created?: number;
  /** Timestamp when tournament started */
  started_at?: string;
  /** Message about the start */
  message?: string;
}

/**
 * Tournament cancellation result.
 */
export interface TournamentCancelResult {
  /** Whether cancellation was successful */
  cancelled: boolean;
  /** Updated tournament status */
  status: TournamentStatus;
  /** Cancellation reason if provided */
  reason?: string;
  /** Timestamp when cancelled */
  cancelled_at?: string;
  /** Message about the cancellation */
  message?: string;
}

/**
 * Tournament statistics.
 */
export interface TournamentStats {
  /** Tournament ID */
  tournament_id: string;
  /** Total participants */
  total_participants: number;
  /** Completed matches */
  completed_matches: number;
  /** Remaining matches */
  remaining_matches: number;
  /** Current round */
  current_round: number;
  /** Total rounds */
  total_rounds: number;
  /** Average match duration in seconds */
  avg_match_duration_seconds?: number;
  /** Participants still active */
  active_participants: number;
  /** Eliminated participants */
  eliminated_participants: number;
}

/**
 * Tournament participant registration.
 */
export interface TournamentRegistration {
  /** Tournament ID */
  tournament_id: string;
  /** Participant/agent name */
  participant: string;
  /** Registration timestamp */
  registered_at: string;
  /** Seed number if assigned */
  seed?: number;
  /** Registration status */
  status: 'registered' | 'confirmed' | 'withdrawn';
}

/**
 * Tournament seeding configuration.
 */
export interface TournamentSeeding {
  /** Seeding method */
  method: 'random' | 'elo' | 'manual';
  /** Seeds assigned to participants */
  seeds: Array<{ participant: string; seed: number }>;
}

/**
 * Interface for the internal client methods used by TournamentsAPI.
 */
interface TournamentsClientInterface {
  listTournaments(params?: TournamentListParams): Promise<{ tournaments: Tournament[] }>;
  getTournament(tournamentId: string): Promise<Tournament>;
  createTournament(request: CreateTournamentRequest): Promise<Tournament>;
  getTournamentStandings(tournamentId: string): Promise<TournamentStandings>;
  getTournamentBracket(tournamentId: string): Promise<TournamentBracket>;
  listTournamentMatches(tournamentId: string, params?: MatchListParams): Promise<{ matches: TournamentMatch[] }>;
  submitMatchResult(tournamentId: string, matchId: string, result: MatchResultSubmission): Promise<TournamentMatch>;
  advanceTournament(tournamentId: string): Promise<TournamentAdvanceResult>;

  // Generic request method for additional endpoints
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

/**
 * Tournaments API namespace.
 *
 * Provides methods for managing agent tournaments including:
 * - Creating and listing tournaments
 * - Viewing standings and brackets
 * - Managing matches and rounds
 * - Submitting match results
 * - Advancing tournament rounds
 * - Registration and participation
 * - Statistics and rankings
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create a new tournament
 * const tournament = await client.tournaments.create({
 *   name: 'Weekly Championship',
 *   format: 'single_elimination',
 *   participants: ['claude', 'gpt-4', 'gemini', 'grok'],
 * });
 *
 * // Start the tournament
 * await client.tournaments.start(tournament.id);
 *
 * // Get tournament standings
 * const standings = await client.tournaments.getStandings(tournament.id);
 *
 * // List active tournaments
 * const { tournaments } = await client.tournaments.list({ status: 'active' });
 *
 * // Get a specific match
 * const match = await client.tournaments.getMatch(tournamentId, matchId);
 *
 * // Submit a match result
 * await client.tournaments.submitResult(tournamentId, matchId, {
 *   winner: 'claude',
 *   loser: 'gpt-4',
 *   score: { winner: 3, loser: 1 },
 * });
 *
 * // Advance to next round
 * await client.tournaments.advance(tournament.id);
 *
 * // Cancel a tournament
 * await client.tournaments.cancel(tournament.id, 'Insufficient participants');
 * ```
 */
export class TournamentsAPI {
  constructor(private client: TournamentsClientInterface) {}

  // ===========================================================================
  // Tournament CRUD Operations
  // ===========================================================================

  /**
   * List tournaments with optional filtering.
   *
   * @param params - Optional filter and pagination parameters
   * @returns List of tournaments with pagination info
   *
   * @example
   * ```typescript
   * // List all active tournaments
   * const { tournaments } = await client.tournaments.list({ status: 'active' });
   *
   * // List with pagination
   * const { tournaments } = await client.tournaments.list({
   *   status: 'completed',
   *   limit: 10,
   *   offset: 20
   * });
   * ```
   */
  async list(params?: TournamentListParams): Promise<{ tournaments: Tournament[] }> {
    return this.client.listTournaments(params);
  }

  /**
   * Get a tournament by ID.
   *
   * @param tournamentId - Tournament identifier
   * @returns Tournament details
   *
   * @example
   * ```typescript
   * const tournament = await client.tournaments.get('tournament-123');
   * console.log(`${tournament.name}: ${tournament.status}`);
   * ```
   */
  async get(tournamentId: string): Promise<Tournament> {
    return this.client.getTournament(tournamentId);
  }

  /**
   * Create a new tournament.
   *
   * @param request - Tournament creation parameters
   * @returns Created tournament
   *
   * @example
   * ```typescript
   * const tournament = await client.tournaments.create({
   *   name: 'AI Championship 2024',
   *   description: 'Annual AI agent tournament',
   *   format: 'double_elimination',
   *   participants: ['claude', 'gpt-4', 'gemini', 'grok', 'mistral'],
   *   config: {
   *     debate_rounds: 3,
   *     consensus_type: 'majority',
   *     auto_advance: false
   *   }
   * });
   * ```
   */
  async create(request: CreateTournamentRequest): Promise<Tournament> {
    return this.client.createTournament(request);
  }

  /**
   * Update a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @param updates - Fields to update
   * @returns Updated tournament
   *
   * @example
   * ```typescript
   * const tournament = await client.tournaments.update('tournament-123', {
   *   name: 'Updated Tournament Name',
   *   description: 'New description'
   * });
   * ```
   */
  async update(
    tournamentId: string,
    updates: {
      name?: string;
      description?: string;
      metadata?: Record<string, unknown>;
    }
  ): Promise<Tournament> {
    return this.client.request<Tournament>(
      'PATCH',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}`,
      { body: updates }
    );
  }

  /**
   * Delete a tournament.
   *
   * Note: Only pending tournaments can be deleted. Use cancel() for active tournaments.
   *
   * @param tournamentId - Tournament identifier
   * @returns Deletion confirmation
   *
   * @example
   * ```typescript
   * await client.tournaments.delete('tournament-123');
   * ```
   */
  async delete(tournamentId: string): Promise<{ deleted: boolean; message: string }> {
    return this.client.request<{ deleted: boolean; message: string }>(
      'DELETE',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}`
    );
  }

  // ===========================================================================
  // Tournament Lifecycle
  // ===========================================================================

  /**
   * Start a pending tournament.
   *
   * Initializes the tournament, generates the bracket, and creates initial matches.
   *
   * @param tournamentId - Tournament identifier
   * @returns Start result with tournament status
   *
   * @example
   * ```typescript
   * const result = await client.tournaments.start('tournament-123');
   * if (result.started) {
   *   console.log(`Tournament started with ${result.matches_created} initial matches`);
   * }
   * ```
   */
  async start(tournamentId: string): Promise<TournamentStartResult> {
    return this.client.request<TournamentStartResult>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/start`
    );
  }

  /**
   * Cancel a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @param reason - Optional cancellation reason
   * @returns Cancellation result
   *
   * @example
   * ```typescript
   * const result = await client.tournaments.cancel(
   *   'tournament-123',
   *   'Insufficient participants'
   * );
   * ```
   */
  async cancel(tournamentId: string, reason?: string): Promise<TournamentCancelResult> {
    const body: { reason?: string } = {};
    if (reason) {
      body.reason = reason;
    }
    return this.client.request<TournamentCancelResult>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/cancel`,
      { body: Object.keys(body).length > 0 ? body : undefined }
    );
  }

  /**
   * Advance the tournament to the next round.
   *
   * Only succeeds if all matches in the current round are completed.
   *
   * @param tournamentId - Tournament identifier
   * @returns Advancement result with next round info
   *
   * @example
   * ```typescript
   * const result = await client.tournaments.advance('tournament-123');
   * if (result.advanced) {
   *   console.log(`Advanced to round ${result.next_round}`);
   * }
   * ```
   */
  async advance(tournamentId: string): Promise<TournamentAdvanceResult> {
    return this.client.advanceTournament(tournamentId);
  }

  // ===========================================================================
  // Standings and Brackets
  // ===========================================================================

  /**
   * Get tournament standings.
   *
   * Returns current rankings of all participants based on wins, losses, and points.
   *
   * @param tournamentId - Tournament identifier
   * @returns Tournament standings with rankings
   *
   * @example
   * ```typescript
   * const standings = await client.tournaments.getStandings('tournament-123');
   * for (const entry of standings.standings) {
   *   console.log(`#${entry.rank} ${entry.agent}: ${entry.wins}W-${entry.losses}L`);
   * }
   * ```
   */
  async getStandings(tournamentId: string): Promise<TournamentStandings> {
    return this.client.getTournamentStandings(tournamentId);
  }

  /**
   * Get tournament bracket.
   *
   * Returns the full bracket structure with all rounds and matches.
   *
   * @param tournamentId - Tournament identifier
   * @returns Tournament bracket structure
   *
   * @example
   * ```typescript
   * const bracket = await client.tournaments.getBracket('tournament-123');
   * for (const round of bracket.rounds) {
   *   console.log(`Round ${round.round_number}: ${round.matches.length} matches`);
   * }
   * ```
   */
  async getBracket(tournamentId: string): Promise<TournamentBracket> {
    return this.client.getTournamentBracket(tournamentId);
  }

  // ===========================================================================
  // Match Management
  // ===========================================================================

  /**
   * List matches in a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @param params - Optional filter parameters
   * @returns List of matches
   *
   * @example
   * ```typescript
   * // List all matches
   * const { matches } = await client.tournaments.listMatches('tournament-123');
   *
   * // List matches in a specific round
   * const { matches } = await client.tournaments.listMatches('tournament-123', {
   *   round: 2,
   *   status: 'pending'
   * });
   * ```
   */
  async listMatches(
    tournamentId: string,
    params?: MatchListParams
  ): Promise<{ matches: TournamentMatch[] }> {
    return this.client.listTournamentMatches(tournamentId, params);
  }

  /**
   * Get a specific match.
   *
   * @param tournamentId - Tournament identifier
   * @param matchId - Match identifier
   * @returns Match details
   *
   * @example
   * ```typescript
   * const match = await client.tournaments.getMatch('tournament-123', 'match-456');
   * console.log(`${match.participant_a} vs ${match.participant_b}: ${match.status}`);
   * ```
   */
  async getMatch(tournamentId: string, matchId: string): Promise<TournamentMatch> {
    return this.client.request<TournamentMatch>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/matches/${encodeURIComponent(matchId)}`
    );
  }

  /**
   * Submit a match result.
   *
   * @param tournamentId - Tournament identifier
   * @param matchId - Match identifier
   * @param result - Match result details
   * @returns Updated match with result
   *
   * @example
   * ```typescript
   * const match = await client.tournaments.submitResult(
   *   'tournament-123',
   *   'match-456',
   *   {
   *     winner: 'claude',
   *     loser: 'gpt-4',
   *     score: { winner: 3, loser: 1 },
   *     notes: 'Decisive victory in round 3'
   *   }
   * );
   * ```
   */
  async submitResult(
    tournamentId: string,
    matchId: string,
    result: MatchResultSubmission
  ): Promise<TournamentMatch> {
    return this.client.submitMatchResult(tournamentId, matchId, result);
  }

  /**
   * Schedule a match.
   *
   * @param tournamentId - Tournament identifier
   * @param matchId - Match identifier
   * @param scheduledAt - ISO timestamp for when the match should occur
   * @returns Updated match with scheduled time
   *
   * @example
   * ```typescript
   * const match = await client.tournaments.scheduleMatch(
   *   'tournament-123',
   *   'match-456',
   *   '2024-03-15T14:00:00Z'
   * );
   * ```
   */
  async scheduleMatch(
    tournamentId: string,
    matchId: string,
    scheduledAt: string
  ): Promise<TournamentMatch> {
    return this.client.request<TournamentMatch>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/matches/${encodeURIComponent(matchId)}/schedule`,
      { body: { scheduled_at: scheduledAt } }
    );
  }

  // ===========================================================================
  // Registration and Participation
  // ===========================================================================

  /**
   * Register a participant for a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @param participant - Agent/participant name
   * @param options - Optional registration options
   * @returns Registration confirmation
   *
   * @example
   * ```typescript
   * const registration = await client.tournaments.register(
   *   'tournament-123',
   *   'claude',
   *   { seed: 1 }
   * );
   * ```
   */
  async register(
    tournamentId: string,
    participant: string,
    options?: { seed?: number; metadata?: Record<string, unknown> }
  ): Promise<TournamentRegistration> {
    return this.client.request<TournamentRegistration>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/register`,
      {
        body: {
          participant,
          seed: options?.seed,
          metadata: options?.metadata,
        },
      }
    );
  }

  /**
   * Withdraw a participant from a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @param participant - Agent/participant name
   * @param reason - Optional withdrawal reason
   * @returns Withdrawal confirmation
   *
   * @example
   * ```typescript
   * await client.tournaments.withdraw('tournament-123', 'gpt-4', 'API quota exceeded');
   * ```
   */
  async withdraw(
    tournamentId: string,
    participant: string,
    reason?: string
  ): Promise<{ withdrawn: boolean; message: string }> {
    const body: { participant: string; reason?: string } = { participant };
    if (reason) {
      body.reason = reason;
    }
    return this.client.request<{ withdrawn: boolean; message: string }>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/withdraw`,
      { body }
    );
  }

  /**
   * List participants in a tournament.
   *
   * @param tournamentId - Tournament identifier
   * @returns List of participants with their status
   *
   * @example
   * ```typescript
   * const { participants } = await client.tournaments.listParticipants('tournament-123');
   * for (const p of participants) {
   *   console.log(`${p.participant}: ${p.status}`);
   * }
   * ```
   */
  async listParticipants(
    tournamentId: string
  ): Promise<{ participants: TournamentRegistration[] }> {
    return this.client.request<{ participants: TournamentRegistration[] }>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/participants`
    );
  }

  // ===========================================================================
  // Seeding and Configuration
  // ===========================================================================

  /**
   * Get tournament seeding.
   *
   * @param tournamentId - Tournament identifier
   * @returns Current seeding configuration
   *
   * @example
   * ```typescript
   * const seeding = await client.tournaments.getSeeding('tournament-123');
   * console.log(`Seeding method: ${seeding.method}`);
   * ```
   */
  async getSeeding(tournamentId: string): Promise<TournamentSeeding> {
    return this.client.request<TournamentSeeding>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/seeding`
    );
  }

  /**
   * Set tournament seeding.
   *
   * @param tournamentId - Tournament identifier
   * @param seeding - Seeding configuration
   * @returns Updated seeding
   *
   * @example
   * ```typescript
   * // Seed by ELO rating
   * await client.tournaments.setSeeding('tournament-123', {
   *   method: 'elo'
   * });
   *
   * // Manual seeding
   * await client.tournaments.setSeeding('tournament-123', {
   *   method: 'manual',
   *   seeds: [
   *     { participant: 'claude', seed: 1 },
   *     { participant: 'gpt-4', seed: 2 },
   *     { participant: 'gemini', seed: 3 },
   *     { participant: 'grok', seed: 4 }
   *   ]
   * });
   * ```
   */
  async setSeeding(
    tournamentId: string,
    seeding: Partial<TournamentSeeding>
  ): Promise<TournamentSeeding> {
    return this.client.request<TournamentSeeding>(
      'PUT',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/seeding`,
      { body: seeding }
    );
  }

  // ===========================================================================
  // Statistics and Results
  // ===========================================================================

  /**
   * Get tournament statistics.
   *
   * @param tournamentId - Tournament identifier
   * @returns Tournament statistics
   *
   * @example
   * ```typescript
   * const stats = await client.tournaments.getStats('tournament-123');
   * console.log(`Progress: ${stats.completed_matches}/${stats.completed_matches + stats.remaining_matches} matches`);
   * ```
   */
  async getStats(tournamentId: string): Promise<TournamentStats> {
    return this.client.request<TournamentStats>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/stats`
    );
  }

  /**
   * Get tournament results.
   *
   * Returns final results for a completed tournament.
   *
   * @param tournamentId - Tournament identifier
   * @returns Tournament results with final standings and match history
   *
   * @example
   * ```typescript
   * const results = await client.tournaments.getResults('tournament-123');
   * console.log(`Winner: ${results.winner}`);
   * ```
   */
  async getResults(tournamentId: string): Promise<{
    tournament_id: string;
    winner?: string;
    runner_up?: string;
    standings: TournamentStandings['standings'];
    total_matches: number;
    total_participants: number;
    duration_seconds?: number;
    completed_at?: string;
  }> {
    return this.client.request<{
      tournament_id: string;
      winner?: string;
      runner_up?: string;
      standings: TournamentStandings['standings'];
      total_matches: number;
      total_participants: number;
      duration_seconds?: number;
      completed_at?: string;
    }>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/results`
    );
  }

  /**
   * Get participant's tournament history.
   *
   * @param tournamentId - Tournament identifier
   * @param participant - Agent/participant name
   * @returns Participant's match history in this tournament
   *
   * @example
   * ```typescript
   * const history = await client.tournaments.getParticipantHistory(
   *   'tournament-123',
   *   'claude'
   * );
   * console.log(`Claude: ${history.wins}W-${history.losses}L`);
   * ```
   */
  async getParticipantHistory(
    tournamentId: string,
    participant: string
  ): Promise<{
    tournament_id: string;
    participant: string;
    wins: number;
    losses: number;
    draws: number;
    elo_change: number;
    matches: TournamentMatch[];
  }> {
    return this.client.request<{
      tournament_id: string;
      participant: string;
      wins: number;
      losses: number;
      draws: number;
      elo_change: number;
      matches: TournamentMatch[];
    }>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/participants/${encodeURIComponent(participant)}/history`
    );
  }

  // ===========================================================================
  // Export and Reporting
  // ===========================================================================

  /**
   * Export tournament data.
   *
   * @param tournamentId - Tournament identifier
   * @param format - Export format (json, csv, or markdown)
   * @returns Export data in the requested format
   *
   * @example
   * ```typescript
   * const exportData = await client.tournaments.export('tournament-123', 'markdown');
   * console.log(exportData.content);
   * ```
   */
  async export(
    tournamentId: string,
    format: 'json' | 'csv' | 'markdown' = 'json'
  ): Promise<{
    tournament_id: string;
    format: string;
    content: string;
    filename: string;
    generated_at: string;
  }> {
    return this.client.request<{
      tournament_id: string;
      format: string;
      content: string;
      filename: string;
      generated_at: string;
    }>(
      'GET',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/export`,
      { params: { format } }
    );
  }

  /**
   * Generate tournament report.
   *
   * @param tournamentId - Tournament identifier
   * @returns Comprehensive tournament report
   *
   * @example
   * ```typescript
   * const report = await client.tournaments.generateReport('tournament-123');
   * console.log(report.summary);
   * ```
   */
  async generateReport(tournamentId: string): Promise<{
    tournament_id: string;
    summary: string;
    highlights: string[];
    statistics: TournamentStats;
    standings: TournamentStandings['standings'];
    notable_matches: TournamentMatch[];
    generated_at: string;
  }> {
    return this.client.request<{
      tournament_id: string;
      summary: string;
      highlights: string[];
      statistics: TournamentStats;
      standings: TournamentStandings['standings'];
      notable_matches: TournamentMatch[];
      generated_at: string;
    }>(
      'POST',
      `/api/v1/tournaments/${encodeURIComponent(tournamentId)}/report`
    );
  }
}
