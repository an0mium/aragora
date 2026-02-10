import type { MetaFunction } from '@remix-run/node';
import { Link } from '@remix-run/react';

export const meta: MetaFunction = () => {
  return [
    { title: 'Aragora - Multi-Agent Debate Platform' },
    { name: 'description', content: 'Orchestrate AI debates for defensible decisions' },
  ];
};

export default function Index() {
  return (
    <div>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>
        Welcome to Aragora
      </h1>
      <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
        Multi-agent debate platform for defensible decisions.
      </p>

      <div className="grid">
        <div className="card">
          <h3 style={{ marginBottom: '0.5rem' }}>Start a Debate</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Create a new debate and let AI agents deliberate on your question.
          </p>
          <Link to="/debates/new" className="button">
            New Debate
          </Link>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: '0.5rem' }}>View Debates</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Browse and monitor your existing debates and their outcomes.
          </p>
          <Link to="/debates" className="button">
            View All
          </Link>
        </div>
      </div>
    </div>
  );
}
