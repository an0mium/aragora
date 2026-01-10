'use client';

import { useState } from 'react';
import type { GenesisStats, GenesisEvent, Genome } from './types';

interface EvolutionTabProps {
  genesisStats: GenesisStats | null;
  events: GenesisEvent[];
  topGenomes: Genome[];
  loading: boolean;
}

type SubTab = 'stats' | 'events' | 'genomes';

const getEventIcon = (eventType: string): string => {
  switch (eventType.toLowerCase()) {
    case 'birth':
    case 'spawn':
      return '🌱';
    case 'death':
    case 'cull':
      return '💀';
    case 'mutation':
      return '🧬';
    case 'crossover':
      return '🔀';
    case 'fitness_update':
      return '📈';
    default:
      return '⚡';
  }
};

export function EvolutionTab({ genesisStats, events, topGenomes, loading }: EvolutionTabProps) {
  const [subTab, setSubTab] = useState<SubTab>('stats');

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      {loading && !genesisStats && (
        <div className="text-center text-text-muted py-4 font-mono text-sm">
          Loading evolution data...
        </div>
      )}

      {!loading && !genesisStats && (
        <div className="text-center text-text-muted py-4 font-mono text-sm">
          No evolution data available yet.
        </div>
      )}

      {genesisStats && (
        <>
          {/* Sub-tab Navigation */}
          <div className="flex space-x-1 bg-bg border border-border rounded p-0.5 text-xs">
            {(['stats', 'events', 'genomes'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setSubTab(tab)}
                className={`px-3 py-1 rounded font-mono transition-colors flex-1 capitalize ${
                  subTab === tab
                    ? 'bg-yellow-500 text-bg font-medium'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {tab}
                {tab === 'events' && events.length > 0 && (
                  <span className="ml-1 text-yellow-300">({events.length})</span>
                )}
                {tab === 'genomes' && topGenomes.length > 0 && (
                  <span className="ml-1 text-yellow-300">({topGenomes.length})</span>
                )}
              </button>
            ))}
          </div>

          {/* Stats Sub-tab */}
          {subTab === 'stats' && (
            <>
              {/* Population Stats */}
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 bg-bg border border-border rounded-lg text-center">
                  <div className="text-2xl font-mono text-green-400">{genesisStats.total_births}</div>
                  <div className="text-xs text-text-muted">Births</div>
                </div>
                <div className="p-3 bg-bg border border-border rounded-lg text-center">
                  <div className="text-2xl font-mono text-red-400">{genesisStats.total_deaths}</div>
                  <div className="text-xs text-text-muted">Deaths</div>
                </div>
                <div className="p-3 bg-bg border border-border rounded-lg text-center">
                  <div
                    className={`text-2xl font-mono ${genesisStats.net_population_change >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {genesisStats.net_population_change >= 0 ? '+' : ''}
                    {genesisStats.net_population_change}
                  </div>
                  <div className="text-xs text-text-muted">Net Change</div>
                </div>
              </div>

              {/* Fitness Trend */}
              <div className="p-3 bg-bg border border-border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-mono text-text-muted">Avg Fitness Change (Recent)</span>
                  <span
                    className={`text-lg font-mono ${genesisStats.avg_fitness_change_recent >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {genesisStats.avg_fitness_change_recent >= 0 ? '+' : ''}
                    {genesisStats.avg_fitness_change_recent.toFixed(4)}
                  </span>
                </div>
                <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
                  <div
                    className={`h-full ${genesisStats.avg_fitness_change_recent >= 0 ? 'bg-green-400' : 'bg-red-400'}`}
                    style={{ width: `${Math.min(100, Math.abs(genesisStats.avg_fitness_change_recent) * 500)}%` }}
                  />
                </div>
              </div>

              {/* Event Breakdown */}
              {genesisStats.event_counts && Object.keys(genesisStats.event_counts).length > 0 && (
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-sm font-mono text-text-muted mb-3">Event Types</div>
                  <div className="space-y-2">
                    {Object.entries(genesisStats.event_counts)
                      .filter(([, count]) => count > 0)
                      .sort(([, a], [, b]) => b - a)
                      .map(([type, count]) => (
                        <div key={type} className="flex items-center justify-between text-xs font-mono">
                          <span className="text-text-muted">{type.replace(/_/g, ' ')}</span>
                          <span className="text-yellow-400">{count}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Integrity Status */}
              <div className="flex items-center justify-between p-2 bg-bg border border-border rounded-lg text-xs font-mono">
                <span className="text-text-muted">Ledger Integrity</span>
                <span className={genesisStats.integrity_verified ? 'text-green-400' : 'text-red-400'}>
                  {genesisStats.integrity_verified ? 'VERIFIED' : 'UNVERIFIED'}
                </span>
              </div>
            </>
          )}

          {/* Events Sub-tab */}
          {subTab === 'events' && (
            <div className="space-y-2">
              {events.length === 0 ? (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  No recent evolution events.
                </div>
              ) : (
                events.map((event, idx) => (
                  <div
                    key={idx}
                    className="p-2 bg-bg border border-border rounded-lg flex items-center gap-3"
                  >
                    <span className="text-lg">{getEventIcon(event.event_type)}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-mono text-text">
                        {event.event_type.replace(/_/g, ' ')}
                      </div>
                      <div className="text-xs text-text-muted truncate">
                        Genome: {event.genome_id.slice(0, 12)}...
                        {event.parent_id && (
                          <span className="ml-2">← {event.parent_id.slice(0, 8)}...</span>
                        )}
                      </div>
                    </div>
                    {event.fitness_change !== undefined && (
                      <span
                        className={`text-xs font-mono ${
                          event.fitness_change >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}
                      >
                        {event.fitness_change >= 0 ? '+' : ''}
                        {event.fitness_change.toFixed(3)}
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Genomes Sub-tab */}
          {subTab === 'genomes' && (
            <div className="space-y-2">
              {topGenomes.length === 0 ? (
                <div className="text-center text-text-muted py-4 font-mono text-sm">
                  No genomes available.
                </div>
              ) : (
                <>
                  <div className="text-xs text-text-muted font-mono mb-2">
                    Top {topGenomes.length} Genomes by Fitness
                  </div>
                  {topGenomes.map((genome, idx) => (
                    <div
                      key={genome.genome_id}
                      className="p-3 bg-bg border border-border rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-yellow-400 font-mono text-sm">#{idx + 1}</span>
                          <span className="text-text font-medium">{genome.agent_name}</span>
                        </div>
                        <span className="text-lg font-mono text-green-400">
                          {genome.fitness.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-text-muted font-mono">
                        <span>Gen {genome.generation}</span>
                        <span className="truncate">ID: {genome.genome_id.slice(0, 12)}...</span>
                      </div>
                      {/* Fitness bar */}
                      <div className="mt-2 h-1.5 bg-surface rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-yellow-500 to-green-500"
                          style={{ width: `${Math.min(100, genome.fitness * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
