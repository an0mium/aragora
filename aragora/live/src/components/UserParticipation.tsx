'use client';

import { useState } from 'react';
import type { StreamEvent } from '@/types/events';

interface UserParticipationProps {
  events: StreamEvent[];
  onVote: (choice: string) => void;
  onSuggest: (suggestion: string) => void;
}

export function UserParticipation({ events, onVote, onSuggest }: UserParticipationProps) {
  const [voteChoice, setVoteChoice] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [hasVoted, setHasVoted] = useState(false);

  // Extract current proposals from recent agent messages
  const recentProposals = events
    .filter(e => e.type === 'agent_message' && e.data.role === 'proposer')
    .slice(-4) // Last 4 proposals
    .map(e => ({
      agent: e.data.agent as string,
      content: (e.data.content as string).substring(0, 200) + '...',
    }));

  const handleVote = () => {
    if (voteChoice && !hasVoted) {
      onVote(voteChoice);
      setHasVoted(true);
      setVoteChoice('');
    }
  };

  const handleSuggest = () => {
    if (suggestion.trim()) {
      onSuggest(suggestion.trim());
      setSuggestion('');
    }
  };

  return (
    <div className="bg-card rounded-lg p-4 border border-border">
      <h2 className="text-sm font-medium text-text-muted uppercase tracking-wider mb-3">
        User Participation
      </h2>

      {/* Vote Section */}
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Vote on Proposals</h3>
        {recentProposals.length > 0 ? (
          <div className="space-y-2 mb-3">
            {recentProposals.map((proposal, index) => (
              <div key={index} className="flex items-start gap-2">
                <input
                  type="radio"
                  name="vote"
                  value={proposal.agent}
                  checked={voteChoice === proposal.agent}
                  onChange={(e) => setVoteChoice(e.target.value)}
                  disabled={hasVoted}
                  className="mt-1"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium text-accent">{proposal.agent}</div>
                  <div className="text-xs text-text-muted">{proposal.content}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-text-muted text-sm mb-3">Waiting for proposals...</p>
        )}

        <button
          onClick={handleVote}
          disabled={!voteChoice || hasVoted}
          className="w-full px-3 py-2 bg-accent text-white rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-accent/90"
        >
          {hasVoted ? 'Vote Submitted ✓' : 'Submit Vote'}
        </button>
      </div>

      {/* Suggestion Section */}
      <div>
        <h3 className="text-sm font-medium mb-2">Suggest Counterpoint</h3>
        <textarea
          value={suggestion}
          onChange={(e) => setSuggestion(e.target.value)}
          placeholder="Share your thoughts or suggest an improvement..."
          className="w-full h-20 p-2 bg-surface border border-border rounded text-sm resize-none"
          maxLength={500}
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-xs text-text-muted">{suggestion.length}/500</span>
          <button
            onClick={handleSuggest}
            disabled={!suggestion.trim()}
            className="px-3 py-1 bg-secondary text-white rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-secondary/90"
          >
            Suggest
          </button>
        </div>
      </div>
    </div>
  );
}