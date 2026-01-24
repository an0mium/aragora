'use client';

import { useState, useCallback } from 'react';
import { useOnboardingStore } from '@/store';

const EXAMPLE_TOPICS = [
  'Should we adopt TypeScript for our frontend?',
  'Which CRM should we use: Salesforce or HubSpot?',
  'Should we implement a 4-day work week?',
  'Should we build a monolith or use microservices?',
];

export function FirstDebateStep() {
  const {
    selectedTemplate,
    firstDebateTopic,
    firstDebateId,
    debateStatus,
    debateError,
    setFirstDebateTopic,
    setFirstDebateId,
    setDebateStatus,
    setDebateError,
    updateProgress,
  } = useOnboardingStore();

  const [localError, setLocalError] = useState<string | null>(null);

  const handleStartDebate = useCallback(async () => {
    if (!firstDebateTopic.trim()) {
      setLocalError('Please enter a topic for your debate');
      return;
    }

    setLocalError(null);
    setDebateStatus('creating');
    setDebateError(null);

    try {
      // Create the debate via API
      const response = await fetch('/api/debate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: firstDebateTopic,
          agents: ['claude', 'gpt-4'], // Default agents
          rounds: selectedTemplate?.rounds || 2,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to create debate');
      }

      const data = await response.json();
      setFirstDebateId(data.debate_id);
      setDebateStatus('running');
      updateProgress({ firstDebateStarted: true });

      // Simulate debate completion after a short time for onboarding
      // In production, this would use WebSocket streaming
      setTimeout(() => {
        setDebateStatus('completed');
        updateProgress({ firstDebateCompleted: true });
      }, 3000);

    } catch (err) {
      setDebateError(err instanceof Error ? err.message : 'Failed to start debate');
      setDebateStatus('error');
    }
  }, [firstDebateTopic, selectedTemplate, setFirstDebateId, setDebateStatus, setDebateError, updateProgress]);

  const handleUseExample = (topic: string) => {
    setFirstDebateTopic(topic);
    setLocalError(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-mono text-acid-green mb-2">
          Run Your First Debate
        </h3>
        <p className="text-sm text-text-muted">
          Enter a topic to see Aragora in action
        </p>
      </div>

      {/* Topic Input */}
      <div>
        <label className="block text-sm font-mono text-text mb-2">
          What decision do you need to make?
        </label>
        <textarea
          value={firstDebateTopic}
          onChange={(e) => {
            setFirstDebateTopic(e.target.value);
            setLocalError(null);
          }}
          placeholder="e.g., Should we use microservices or a monolith?"
          rows={3}
          disabled={debateStatus === 'running' || debateStatus === 'completed'}
          className="w-full px-4 py-2 bg-bg border border-acid-green/30 rounded text-text font-mono focus:border-acid-green focus:outline-none disabled:opacity-50"
        />
        {(localError || debateError) && (
          <p className="text-xs text-accent-red mt-1">{localError || debateError}</p>
        )}
      </div>

      {/* Example Topics */}
      {debateStatus === 'idle' && (
        <div>
          <label className="block text-xs font-mono text-text-muted mb-2">
            Or try an example:
          </label>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_TOPICS.map((topic) => (
              <button
                key={topic}
                onClick={() => handleUseExample(topic)}
                className="px-3 py-1 text-xs border border-acid-green/20 rounded hover:border-acid-green/50 hover:bg-acid-green/5 text-text-muted hover:text-text transition-colors"
              >
                {topic.length > 40 ? `${topic.slice(0, 40)}...` : topic}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Debate Status */}
      {debateStatus === 'creating' && (
        <div className="text-center py-4">
          <div className="text-acid-green font-mono text-sm animate-pulse">
            Creating debate...
          </div>
        </div>
      )}

      {debateStatus === 'running' && (
        <div className="p-4 border border-acid-cyan/30 rounded-lg bg-acid-cyan/5">
          <div className="text-sm font-mono text-acid-cyan mb-2">
            Debate in progress...
          </div>
          <div className="text-xs text-text-muted">
            AI agents are debating your topic. This usually takes 2-3 minutes.
          </div>
          <div className="mt-3 w-full h-1 bg-acid-cyan/20 rounded-full overflow-hidden">
            <div className="h-full bg-acid-cyan animate-progress-indeterminate" />
          </div>
        </div>
      )}

      {debateStatus === 'completed' && (
        <div className="p-4 border border-acid-green/30 rounded-lg bg-acid-green/5">
          <div className="text-sm font-mono text-acid-green mb-2">
            Debate completed!
          </div>
          <div className="text-xs text-text-muted">
            Your first debate has finished. Click Continue to see your decision receipt.
          </div>
          {firstDebateId && (
            <div className="text-xs text-text-muted mt-2">
              Debate ID: {firstDebateId}
            </div>
          )}
        </div>
      )}

      {/* Start Button */}
      {debateStatus === 'idle' && (
        <button
          onClick={handleStartDebate}
          disabled={!firstDebateTopic.trim()}
          className="w-full px-4 py-3 bg-acid-green text-bg font-mono text-sm hover:bg-acid-green/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          START DEBATE
        </button>
      )}

      {/* Template Info */}
      {selectedTemplate && debateStatus === 'idle' && (
        <div className="text-center text-xs text-text-muted">
          Using template: {selectedTemplate.name} ({selectedTemplate.agentsCount} agents, {selectedTemplate.rounds} rounds)
        </div>
      )}
    </div>
  );
}
