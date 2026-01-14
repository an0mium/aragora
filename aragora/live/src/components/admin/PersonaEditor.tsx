'use client';

import { useState, useEffect, useCallback } from 'react';

interface Persona {
  agent_name: string;
  description: string;
  traits: string[];
  expertise: string[];
  created_at: string;
  updated_at: string;
}

interface PersonaEditorProps {
  apiBase?: string;
}

type ViewMode = 'grid' | 'list';

export function PersonaEditor({ apiBase = '/api' }: PersonaEditorProps) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');

  const fetchPersonas = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/personas`);
      if (!response.ok) {
        throw new Error(`Failed to fetch personas: ${response.status}`);
      }
      const data = await response.json();
      setPersonas(data.personas || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load personas');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchPersonas();
  }, [fetchPersonas]);

  const filteredPersonas = personas.filter((persona) => {
    const query = searchQuery.toLowerCase();
    return (
      persona.agent_name.toLowerCase().includes(query) ||
      persona.description.toLowerCase().includes(query) ||
      persona.traits.some((t) => t.toLowerCase().includes(query)) ||
      persona.expertise.some((e) => e.toLowerCase().includes(query))
    );
  });

  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return (
      <div className="bg-surface border border-acid-green/30 p-8">
        <div className="flex items-center justify-center gap-2">
          <div className="w-2 h-2 bg-acid-green rounded-full animate-pulse" />
          <span className="text-xs font-mono text-acid-green">LOADING PERSONAS...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-surface border border-crimson/30 p-4">
        <div className="flex items-center gap-2">
          <span className="text-crimson text-xs font-mono">ERROR:</span>
          <span className="text-text-primary text-xs font-mono">{error}</span>
        </div>
        <button
          onClick={fetchPersonas}
          className="mt-3 px-3 py-1.5 text-xs font-mono bg-crimson/20 text-crimson border border-crimson/40 hover:bg-crimson/30 transition-colors"
        >
          RETRY
        </button>
      </div>
    );
  }

  return (
    <div className="bg-surface border border-acid-green/30">
      {/* Header */}
      <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
            {'>'} PERSONA MANAGER
          </span>
          <span className="text-xs font-mono text-text-muted">
            {personas.length} agent{personas.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-2 py-1 text-xs font-mono border ${
              viewMode === 'grid'
                ? 'bg-acid-green/20 text-acid-green border-acid-green/40'
                : 'text-text-muted border-border hover:border-acid-green/40'
            }`}
          >
            GRID
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`px-2 py-1 text-xs font-mono border ${
              viewMode === 'list'
                ? 'bg-acid-green/20 text-acid-green border-acid-green/40'
                : 'text-text-muted border-border hover:border-acid-green/40'
            }`}
          >
            LIST
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="px-4 py-3 border-b border-acid-green/10">
        <input
          type="text"
          placeholder="Search personas by name, traits, or expertise..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full bg-bg border border-border px-3 py-2 text-xs font-mono text-text-primary placeholder-text-muted focus:border-acid-green/50 focus:outline-none"
        />
      </div>

      {/* Content */}
      <div className="p-4">
        {filteredPersonas.length === 0 ? (
          <div className="text-center py-8">
            <span className="text-xs font-mono text-text-muted">
              {searchQuery ? 'No personas match your search' : 'No personas configured'}
            </span>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredPersonas.map((persona) => (
              <PersonaCard
                key={persona.agent_name}
                persona={persona}
                isSelected={selectedPersona?.agent_name === persona.agent_name}
                onClick={() =>
                  setSelectedPersona(
                    selectedPersona?.agent_name === persona.agent_name ? null : persona
                  )
                }
                formatDate={formatDate}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredPersonas.map((persona) => (
              <PersonaRow
                key={persona.agent_name}
                persona={persona}
                isSelected={selectedPersona?.agent_name === persona.agent_name}
                onClick={() =>
                  setSelectedPersona(
                    selectedPersona?.agent_name === persona.agent_name ? null : persona
                  )
                }
                formatDate={formatDate}
              />
            ))}
          </div>
        )}
      </div>

      {/* Detail Panel */}
      {selectedPersona && (
        <PersonaDetailPanel
          persona={selectedPersona}
          onClose={() => setSelectedPersona(null)}
          formatDate={formatDate}
        />
      )}
    </div>
  );
}

interface PersonaCardProps {
  persona: Persona;
  isSelected: boolean;
  onClick: () => void;
  formatDate: (d: string) => string;
}

function PersonaCard({ persona, isSelected, onClick, formatDate }: PersonaCardProps) {
  return (
    <div
      onClick={onClick}
      className={`p-4 border cursor-pointer transition-all ${
        isSelected
          ? 'border-acid-green bg-acid-green/10'
          : 'border-border hover:border-acid-green/40 bg-bg/30'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-sm font-mono text-acid-cyan font-medium">
          {persona.agent_name}
        </span>
        <span className="text-xs font-mono text-text-muted">{formatDate(persona.updated_at)}</span>
      </div>

      <p className="text-xs font-mono text-text-primary mb-3 line-clamp-2">
        {persona.description || 'No description'}
      </p>

      {persona.traits.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {persona.traits.slice(0, 3).map((trait) => (
            <span
              key={trait}
              className="px-1.5 py-0.5 text-xs font-mono bg-purple/10 text-purple border border-purple/30"
            >
              {trait}
            </span>
          ))}
          {persona.traits.length > 3 && (
            <span className="px-1.5 py-0.5 text-xs font-mono text-text-muted">
              +{persona.traits.length - 3}
            </span>
          )}
        </div>
      )}

      {persona.expertise.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {persona.expertise.slice(0, 3).map((exp) => (
            <span
              key={exp}
              className="px-1.5 py-0.5 text-xs font-mono bg-acid-green/10 text-acid-green border border-acid-green/30"
            >
              {exp}
            </span>
          ))}
          {persona.expertise.length > 3 && (
            <span className="px-1.5 py-0.5 text-xs font-mono text-text-muted">
              +{persona.expertise.length - 3}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

interface PersonaRowProps {
  persona: Persona;
  isSelected: boolean;
  onClick: () => void;
  formatDate: (d: string) => string;
}

function PersonaRow({ persona, isSelected, onClick, formatDate }: PersonaRowProps) {
  return (
    <div
      onClick={onClick}
      className={`p-3 border cursor-pointer transition-all flex items-center gap-4 ${
        isSelected
          ? 'border-acid-green bg-acid-green/10'
          : 'border-border hover:border-acid-green/40 bg-bg/30'
      }`}
    >
      <div className="flex-shrink-0 w-32">
        <span className="text-sm font-mono text-acid-cyan font-medium">{persona.agent_name}</span>
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-xs font-mono text-text-primary truncate">
          {persona.description || 'No description'}
        </p>
      </div>

      <div className="flex-shrink-0 flex items-center gap-2">
        <span className="text-xs font-mono text-purple">
          {persona.traits.length} traits
        </span>
        <span className="text-xs font-mono text-acid-green">
          {persona.expertise.length} expertise
        </span>
      </div>

      <div className="flex-shrink-0 w-24 text-right">
        <span className="text-xs font-mono text-text-muted">{formatDate(persona.updated_at)}</span>
      </div>
    </div>
  );
}

interface PersonaDetailPanelProps {
  persona: Persona;
  onClose: () => void;
  formatDate: (d: string) => string;
}

function PersonaDetailPanel({ persona, onClose, formatDate }: PersonaDetailPanelProps) {
  return (
    <div className="border-t border-acid-green/20 bg-bg/50 p-4">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
          PERSONA DETAILS: {persona.agent_name}
        </span>
        <button
          onClick={onClose}
          className="px-2 py-1 text-xs font-mono text-text-muted hover:text-crimson border border-border hover:border-crimson/40 transition-colors"
        >
          CLOSE
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h4 className="text-xs font-mono text-text-muted mb-2">DESCRIPTION</h4>
          <p className="text-xs font-mono text-text-primary bg-surface p-3 border border-border">
            {persona.description || 'No description provided'}
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <h4 className="text-xs font-mono text-text-muted mb-2">TRAITS</h4>
            <div className="flex flex-wrap gap-1">
              {persona.traits.length > 0 ? (
                persona.traits.map((trait) => (
                  <span
                    key={trait}
                    className="px-2 py-1 text-xs font-mono bg-purple/10 text-purple border border-purple/30"
                  >
                    {trait}
                  </span>
                ))
              ) : (
                <span className="text-xs font-mono text-text-muted">No traits defined</span>
              )}
            </div>
          </div>

          <div>
            <h4 className="text-xs font-mono text-text-muted mb-2">EXPERTISE</h4>
            <div className="flex flex-wrap gap-1">
              {persona.expertise.length > 0 ? (
                persona.expertise.map((exp) => (
                  <span
                    key={exp}
                    className="px-2 py-1 text-xs font-mono bg-acid-green/10 text-acid-green border border-acid-green/30"
                  >
                    {exp}
                  </span>
                ))
              ) : (
                <span className="text-xs font-mono text-text-muted">No expertise defined</span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-border flex gap-6">
        <div>
          <span className="text-xs font-mono text-text-muted">CREATED: </span>
          <span className="text-xs font-mono text-text-primary">{formatDate(persona.created_at)}</span>
        </div>
        <div>
          <span className="text-xs font-mono text-text-muted">UPDATED: </span>
          <span className="text-xs font-mono text-text-primary">{formatDate(persona.updated_at)}</span>
        </div>
      </div>
    </div>
  );
}

export default PersonaEditor;
