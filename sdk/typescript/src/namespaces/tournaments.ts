/**
 * Tournaments Namespace API
 *
 * Provides a namespaced interface for tournament management operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  Tournament,
  TournamentStandings,
  TournamentBracket,
  TournamentMatch,
  CreateTournamentRequest,
  PaginationParams,
} from '../types';

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
 * Interface for the internal client methods used by TournamentsAPI.
 */
interface TournamentsClientInterface {
  listTournaments(params?: { status?: string } & PaginationParams): Promise<{ tournaments: Tournament[] }>;
  getTournament(tournamentId: string): Promise<Tournament>;
  createTournament(request: CreateTournamentRequest): Promise<Tournament>;
  getTournamentStandings(tournamentId: string): Promise<TournamentStandings>;
  getTournamentBracket(tournamentId: string): Promise<TournamentBracket>;
  listTournamentMatches(tournamentId: string, params?: { round?: number; status?: string }): Promise<{ matches: TournamentMatch[] }>;
  submitMatchResult(tournamentId: string, matchId: string, result: MatchResultSubmission): Promise<TournamentMatch>;
  advanceTournament(tournamentId: string): Promise<{ advanced: boolean; next_round: number }>;
}

/**
 * Tournaments API namespace.
 *
 * Provides methods for managing tournaments:
 * - Creating and listing tournaments
 * - Viewing standings and brackets
 * - Submitting match results
 * - Advancing tournament rounds
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
 * // Get tournament standings
 * const standings = await client.tournaments.getStandings(tournament.id);
 *
 * // List active tournaments
 * const { tournaments } = await client.tournaments.list({ status: 'active' });
 *
 * // Submit a match result
 * await client.tournaments.submitResult(tournamentId, matchId, {
 *   winner: 'claude',
 *   loser: 'gpt-4',
 * });
 * ```
 */
export class TournamentsAPI {
  constructor(private client: TournamentsClientInterface) {}

  /**
   * List tournaments with optional filtering.
   */
  async list(params?: { status?: string } & PaginationParams): Promise<{ tournaments: Tournament[] }> {
    return this.client.listTournaments(params);
  }

  /**
   * Get a tournament by ID.
   */
  async get(tournamentId: string): Promise<Tournament> {
    return this.client.getTournament(tournamentId);
  }

  /**
   * Create a new tournament.
   */
  async create(request: CreateTournamentRequest): Promise<Tournament> {
    return this.client.createTournament(request);
  }

  /**
   * Get tournament standings.
   */
  async getStandings(tournamentId: string): Promise<TournamentStandings> {
    return this.client.getTournamentStandings(tournamentId);
  }

  /**
   * Get tournament bracket.
   */
  async getBracket(tournamentId: string): Promise<TournamentBracket> {
    return this.client.getTournamentBracket(tournamentId);
  }

  /**
   * List matches in a tournament.
   */
  async listMatches(
    tournamentId: string,
    params?: { round?: number; status?: string }
  ): Promise<{ matches: TournamentMatch[] }> {
    return this.client.listTournamentMatches(tournamentId, params);
  }

  /**
   * Submit a match result.
   */
  async submitResult(
    tournamentId: string,
    matchId: string,
    result: MatchResultSubmission
  ): Promise<TournamentMatch> {
    return this.client.submitMatchResult(tournamentId, matchId, result);
  }

  /**
   * Advance the tournament to the next round.
   */
  async advance(tournamentId: string): Promise<{ advanced: boolean; next_round: number }> {
    return this.client.advanceTournament(tournamentId);
  }
}
