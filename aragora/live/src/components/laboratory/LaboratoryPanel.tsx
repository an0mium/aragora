'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { ErrorWithRetry } from '../RetryButton';
import { fetchWithRetry } from '@/utils/retry';
import { RefreshButton, ExpandToggle } from '../shared';
import { TraitsTab } from './TraitsTab';
import { PollinationsTab } from './PollinationsTab';
import { EvolutionTab } from './EvolutionTab';
import { PatternsTab } from './PatternsTab';
import type {
  EmergentTrait,
  CrossPollination,
  GenesisStats,
  GenesisEvent,
  Genome,
  CritiquePattern,
  LaboratoryPanelProps,
  LaboratoryTab,
} from './types';

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

const TAB_STYLES: Record<LaboratoryTab, string> = {
  traits: 'bg-acid-cyan text-bg font-medium',
  pollinations: 'bg-acid-green text-bg font-medium',
  evolution: 'bg-yellow-500 text-bg font-medium',
  patterns: 'bg-purple-500 text-bg font-medium',
};

const TAB_LABELS: Record<LaboratoryTab, string> = {
  traits: 'EMERGENT TRAITS',
  pollinations: 'POLLINATIONS',
  evolution: 'EVOLUTION',
  patterns: 'PATTERNS',
};

export function LaboratoryPanel({ apiBase = DEFAULT_API_BASE }: LaboratoryPanelProps) {
  const [traits, setTraits] = useState<EmergentTrait[]>([]);
  const [pollinations, setPollinations] = useState<CrossPollination[]>([]);
  const [genesisStats, setGenesisStats] = useState<GenesisStats | null>(null);
  const [genesisEvents, setGenesisEvents] = useState<GenesisEvent[]>([]);
  const [topGenomes, setTopGenomes] = useState<Genome[]>([]);
  const [patterns, setPatterns] = useState<CritiquePattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<LaboratoryTab>('traits');
  const [expanded, setExpanded] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    // Use allSettled to handle partial failures gracefully
    const results = await Promise.allSettled([
      fetchWithRetry(`${apiBase}/api/laboratory/emergent-traits?min_confidence=0.3&limit=10`, undefined, { maxRetries: 2 }),
      fetchWithRetry(`${apiBase}/api/laboratory/cross-pollinations/suggest`, undefined, { maxRetries: 2 }),
      fetchWithRetry(`${apiBase}/api/genesis/stats`, undefined, { maxRetries: 2 }),
      fetchWithRetry(`${apiBase}/api/genesis/events?limit=10`, undefined, { maxRetries: 2 }),
      fetchWithRetry(`${apiBase}/api/genesis/genomes/top?limit=5`, undefined, { maxRetries: 2 }),
      fetchWithRetry(`${apiBase}/api/critiques/patterns?limit=15&min_success=0.5`, undefined, { maxRetries: 2 }),
    ]);

    const [traitsResult, pollinationsResult, genesisResult, eventsResult, genomesResult, patternsResult] = results;
    let hasError = false;

    if (traitsResult.status === 'fulfilled' && traitsResult.value.ok) {
      const data = await traitsResult.value.json();
      setTraits(data.emergent_traits || []);
    } else {
      hasError = true;
    }

    if (pollinationsResult.status === 'fulfilled' && pollinationsResult.value.ok) {
      const data = await pollinationsResult.value.json();
      setPollinations(data.suggestions || []);
    } else {
      hasError = true;
    }

    if (genesisResult.status === 'fulfilled' && genesisResult.value.ok) {
      const data = await genesisResult.value.json();
      setGenesisStats(data);
    } else {
      hasError = true;
    }

    if (eventsResult.status === 'fulfilled' && eventsResult.value.ok) {
      const data = await eventsResult.value.json();
      setGenesisEvents(data.events || []);
    }

    if (genomesResult.status === 'fulfilled' && genomesResult.value.ok) {
      const data = await genomesResult.value.json();
      setTopGenomes(data.genomes || []);
    }

    if (patternsResult.status === 'fulfilled' && patternsResult.value.ok) {
      const data = await patternsResult.value.json();
      setPatterns(data.patterns || []);
    } else {
      hasError = true;
    }

    if (hasError) {
      setError('Some data failed to load. Partial results shown.');
    }
    setLoading(false);
  }, [apiBase]);

  // Use ref to store latest fetchData to avoid interval recreation
  const fetchDataRef = useRef(fetchData);
  fetchDataRef.current = fetchData;

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Separate effect for interval - runs once, uses ref
  useEffect(() => {
    const interval = setInterval(() => {
      fetchDataRef.current();
    }, 300000);
    return () => clearInterval(interval);
  }, []);

  const handleTabChange = useCallback((tabId: LaboratoryTab) => {
    setActiveTab(tabId);
  }, []);

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text font-mono">Persona Laboratory</h3>
        <div className="flex items-center gap-2">
          <RefreshButton onClick={fetchData} loading={loading} />
          <ExpandToggle expanded={expanded} onToggle={() => setExpanded(!expanded)} />
        </div>
      </div>

      {/* Summary Stats */}
      <div className="flex items-center gap-4 text-xs font-mono text-text-muted mb-4 border-b border-border pb-3 flex-wrap">
        <span>
          Traits: <span className="text-acid-cyan">{traits.length}</span>
        </span>
        <span>
          Pollinations: <span className="text-acid-green">{pollinations.length}</span>
        </span>
        <span>
          Patterns: <span className="text-purple-400">{patterns.length}</span>
        </span>
        {genesisStats && (
          <>
            <span>
              Population:{' '}
              <span className={genesisStats.net_population_change >= 0 ? 'text-green-400' : 'text-red-400'}>
                {genesisStats.net_population_change >= 0 ? '+' : ''}
                {genesisStats.net_population_change}
              </span>
            </span>
            <span>
              Events: <span className="text-yellow-400">{genesisStats.total_events}</span>
            </span>
          </>
        )}
      </div>

      {error && <ErrorWithRetry error={error} onRetry={fetchData} className="mb-4" />}

      {expanded && (
        <>
          {/* Tab Navigation */}
          <div className="flex space-x-1 bg-bg border border-border rounded p-1 mb-4">
            {(Object.keys(TAB_LABELS) as LaboratoryTab[]).map((tabId) => (
              <button
                key={tabId}
                role="tab"
                aria-selected={activeTab === tabId}
                onClick={() => handleTabChange(tabId)}
                className={`px-3 py-1 rounded text-sm font-mono transition-colors flex-1 ${
                  activeTab === tabId ? TAB_STYLES[tabId] : 'text-text-muted hover:text-text'
                }`}
              >
                {TAB_LABELS[tabId]}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'traits' && <TraitsTab traits={traits} loading={loading} />}
          {activeTab === 'pollinations' && <PollinationsTab pollinations={pollinations} loading={loading} />}
          {activeTab === 'evolution' && (
            <EvolutionTab
              genesisStats={genesisStats}
              events={genesisEvents}
              topGenomes={topGenomes}
              loading={loading}
            />
          )}
          {activeTab === 'patterns' && <PatternsTab patterns={patterns} loading={loading} />}
        </>
      )}

      {/* Help text when collapsed */}
      {!expanded && (
        <div className="text-xs font-mono text-text-muted">
          <p>
            <span className="text-acid-cyan">Traits:</span> Discovered specializations |{' '}
            <span className="text-acid-green">Pollinations:</span> Trait transfers |{' '}
            <span className="text-yellow-400">Evolution:</span> Population dynamics
          </p>
        </div>
      )}
    </div>
  );
}
