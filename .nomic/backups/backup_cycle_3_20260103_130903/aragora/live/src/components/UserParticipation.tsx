'use client';

import { useState, useEffect, useMemo } from 'react';
import type { StreamEvent, AudienceSummaryData } from '@/types/events';

interface UserParticipationProps {
  events: StreamEvent[];
  onVote: (choice: string) => void;
  onSuggest: (suggestion: string) => void;
  onAck?: (callback: (msgType: string) => void) => () => void;
  onError?: (callback: (message: string) => void) => () => void;
}

type SubmissionState = 'idle' | 'pending' | 'success' | 'error' | 'rate_limited';

export function UserParticipation({ events, onVote, onSuggest, onAck, onError }: UserParticipationProps) {
  const [voteChoice, setVoteChoice] = useState('');
  const [suggestion, setSuggestion] = useState('');
  const [hasVoted, setHasVoted] = useState(false);
  const [voteState, setVoteState] = useState<SubmissionState>('idle');
  const [suggestionState, setSuggestionState] = useState<SubmissionState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Get the latest audience summary from events
  const audienceSummary = useMemo(() => {
    const summaryEvents = events.filter(e => e.type === 'audience_summary');
    if (summaryEvents.length === 0) return null;
    const latest = summaryEvents[summaryEvents.length - 1];
    return latest.data as unknown as AudienceSummaryData;
  }, [events]);

  // Handle acknowledgments
  useEffect(() => {
    if (!onAck) return;
    const unsubscribeAck = onAck((msgType) => {
      if (msgType === 'user_vote') {
        setVoteState('success');
        setTimeout(() => setVoteState('idle'), 2000);
      } else if (msgType === 'user_suggestion') {
        setSuggestionState('success');
        setTimeout(() => setSuggestionState('idle'), 2000);
      }
    });

    return unsubscribeAck;
  }, [onAck]);

  // Handle errors (including rate limiting)
  useEffect(() => {
    if (!onError) return;
    const unsubscribeError = onError((message) => {
      const isRateLimited = message.toLowerCase().includes('rate limit');
      const newState: SubmissionState = isRateLimited ? 'rate_limited' : 'error';

      if (voteState === 'pending') {
        setVoteState(newState);
        setErrorMessage(message);
      } else if (suggestionState === 'pending') {
        setSuggestionState(newState);
        setErrorMessage(message);
      } else {
        // Error came without pending state (e.g., general error)
        setErrorMessage(message);
      }

      setTimeout(() => {
        setVoteState((s) => s === newState ? 'idle' : s);
        setSuggestionState((s) => s === newState ? 'idle' : s);
        setErrorMessage(null);
      }, isRateLimited ? 5000 : 3000);
    });

    return unsubscribeError;
  }, [onError, voteState, suggestionState]);

  // Extract current proposals from recent agent messages
  const recentProposals = events
    .filter(e => e.type === 'agent_message' && e.data.role === 'proposer')
    .slice(-4) // Last 4 proposals
    .map(e => ({
      agent: e.data.agent as string,
      content: e.data.content as string, // Full content, no truncation
    }));

  const handleVote = () => {
    if (voteChoice && !hasVoted && voteState === 'idle') {
      setVoteState('pending');
      onVote(voteChoice);
      setHasVoted(true);
      setVoteChoice('');
    }
  };

  const handleSuggest = () => {
    if (suggestion.trim() && suggestionState === 'idle') {
      setSuggestionState('pending');
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
                  <div className="text-xs text-text-muted whitespace-pre-wrap break-words max-h-48 overflow-y-auto">{proposal.content}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-text-muted text-sm mb-3">Waiting for proposals...</p>
        )}

        <button
          onClick={handleVote}
          disabled={!voteChoice || hasVoted || voteState !== 'idle'}
          className={`w-full px-3 py-2 rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed ${
            voteState === 'success'
              ? 'bg-success text-white'
              : voteState === 'error' || voteState === 'rate_limited'
              ? 'bg-warning text-white'
              : voteState === 'pending'
              ? 'bg-surface text-text animate-pulse'
              : 'bg-accent text-white hover:bg-accent/90'
          }`}
        >
          {voteState === 'pending' ? 'Submitting...' :
           voteState === 'success' ? 'Vote Submitted ✓' :
           voteState === 'rate_limited' ? 'Rate Limited - Wait' :
           voteState === 'error' ? 'Failed - Try Again' :
           hasVoted ? 'Vote Submitted ✓' : 'Submit Vote'}
        </button>
        {voteState === 'error' && errorMessage && (
          <p className="text-xs text-warning mt-1">{errorMessage}</p>
        )}
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
            disabled={!suggestion.trim() || suggestionState !== 'idle'}
            className={`px-3 py-1 rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed ${
              suggestionState === 'success'
                ? 'bg-success text-white'
                : suggestionState === 'error' || suggestionState === 'rate_limited'
                ? 'bg-warning text-white'
                : suggestionState === 'pending'
                ? 'bg-surface text-text animate-pulse'
                : 'bg-secondary text-white hover:bg-secondary/90'
            }`}
          >
            {suggestionState === 'pending' ? '...' :
             suggestionState === 'success' ? 'Sent ✓' :
             suggestionState === 'rate_limited' ? 'Wait' :
             suggestionState === 'error' ? 'Failed' :
             'Suggest'}
          </button>
        </div>
        {suggestionState === 'error' && errorMessage && (
          <p className="text-xs text-warning mt-1">{errorMessage}</p>
        )}
        {suggestionState === 'rate_limited' && (
          <p className="text-xs text-warning mt-1">Rate limited. Please wait before submitting again.</p>
        )}
      </div>

      {/* Audience Pulse Section - shows clustered suggestions */}
      {audienceSummary && audienceSummary.clusters.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <span className="inline-block w-2 h-2 bg-accent rounded-full animate-pulse" />
            Audience Pulse
            <span className="text-xs text-text-muted font-normal">
              ({audienceSummary.total} suggestions)
            </span>
          </h3>
          <div className="space-y-2">
            {audienceSummary.clusters.slice(0, 3).map((cluster, index) => (
              <div
                key={index}
                className="text-sm p-2 bg-surface rounded border border-border"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-text-muted flex-1">
                    &ldquo;{cluster.representative}&rdquo;
                  </span>
                  <span className="text-xs bg-accent/20 text-accent px-1.5 py-0.5 rounded whitespace-nowrap">
                    {cluster.count}x
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}