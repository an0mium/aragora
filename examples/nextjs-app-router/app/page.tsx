import Link from 'next/link';

export default function Home() {
  return (
    <div>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
        Welcome to Aragora
      </h1>
      <p style={{ color: 'var(--text-muted)', marginBottom: '2rem', maxWidth: '600px' }}>
        Aragora orchestrates multi-agent debates to help you make defensible decisions.
        15+ AI models collaborate through structured deliberation to reach consensus.
      </p>

      <div className="grid" style={{ marginBottom: '3rem' }}>
        <div className="card">
          <h3 style={{ marginBottom: '0.5rem' }}>Start a Debate</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Create a new debate and let AI agents deliberate on your question.
          </p>
          <Link href="/debates/new" className="button">
            New Debate
          </Link>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: '0.5rem' }}>View Debates</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Browse and monitor your existing debates and their outcomes.
          </p>
          <Link href="/debates" className="button">
            View All
          </Link>
        </div>
      </div>

      <h2 style={{ marginBottom: '1rem' }}>Features</h2>
      <ul style={{ color: 'var(--text-muted)', paddingLeft: '1.5rem' }}>
        <li>Multi-agent orchestration with 15+ AI models</li>
        <li>Structured propose-critique-revise workflow</li>
        <li>Real-time streaming of debate progress</li>
        <li>Consensus detection and proof generation</li>
        <li>Knowledge mound integration for context</li>
      </ul>
    </div>
  );
}
