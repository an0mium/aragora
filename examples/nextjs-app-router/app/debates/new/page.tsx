'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { getClientSideClient } from '@/lib/aragora';

const AVAILABLE_AGENTS = [
  { id: 'claude', name: 'Claude', provider: 'Anthropic' },
  { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI' },
  { id: 'gemini', name: 'Gemini', provider: 'Google' },
  { id: 'grok', name: 'Grok', provider: 'xAI' },
  { id: 'mistral', name: 'Mistral Large', provider: 'Mistral' },
];

export default function NewDebatePage() {
  const router = useRouter();
  const [task, setTask] = useState('');
  const [selectedAgents, setSelectedAgents] = useState<string[]>(['claude', 'gpt-4', 'gemini']);
  const [rounds, setRounds] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleAgent = (agentId: string) => {
    setSelectedAgents(prev =>
      prev.includes(agentId)
        ? prev.filter(id => id !== agentId)
        : [...prev, agentId]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim()) {
      setError('Please enter a debate topic');
      return;
    }
    if (selectedAgents.length < 2) {
      setError('Please select at least 2 agents');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const client = getClientSideClient();
      const result = await client.debates.create({
        task: task.trim(),
        agents: selectedAgents,
        rounds,
      });

      router.push(`/debates/${result.debate_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create debate');
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px' }}>
      <h1 style={{ marginBottom: '2rem' }}>Create New Debate</h1>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
            Debate Topic
          </label>
          <textarea
            className="input"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="What question should the agents debate? e.g., 'Should we adopt microservices architecture?'"
            rows={4}
            style={{ resize: 'vertical' }}
          />
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
            Select Agents ({selectedAgents.length} selected)
          </label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {AVAILABLE_AGENTS.map(agent => (
              <button
                key={agent.id}
                type="button"
                onClick={() => toggleAgent(agent.id)}
                style={{
                  padding: '0.5rem 1rem',
                  border: '1px solid var(--border)',
                  borderRadius: '0.5rem',
                  background: selectedAgents.includes(agent.id) ? 'var(--primary)' : 'transparent',
                  color: selectedAgents.includes(agent.id) ? 'white' : 'var(--text)',
                  cursor: 'pointer',
                }}
              >
                {agent.name}
                <span style={{ fontSize: '0.75rem', opacity: 0.7, marginLeft: '0.25rem' }}>
                  ({agent.provider})
                </span>
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
            Number of Rounds: {rounds}
          </label>
          <input
            type="range"
            min={3}
            max={9}
            value={rounds}
            onChange={(e) => setRounds(parseInt(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            <span>3 (Quick)</span>
            <span>9 (Thorough)</span>
          </div>
        </div>

        {error && (
          <div style={{ padding: '1rem', background: '#fee2e2', color: '#991b1b', borderRadius: '0.5rem', marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <button type="submit" className="button" disabled={loading} style={{ width: '100%' }}>
          {loading ? 'Creating...' : 'Start Debate'}
        </button>
      </form>
    </div>
  );
}
