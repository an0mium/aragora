'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function AboutPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="https://live.aragora.ai"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [LIVE DASHBOARD]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="py-16 px-4">
          <div className="container mx-auto max-w-4xl text-center">
            <h1 className="text-3xl sm:text-4xl font-mono mb-4">
              <span className="text-acid-green">aragora</span>
            </h1>
            <p className="text-xl text-acid-cyan font-mono mb-4">Self-Improving AI</p>
            <p className="text-text-muted font-mono mb-8">
              Multi-agent debate with autonomous self-modification.<br />
              Watch AI agents reason, critique, and evolve the system in real-time.
            </p>
            <div className="flex justify-center items-center gap-2 text-xs text-text-muted font-mono mb-8">
              <span className="text-acid-green">ar-</span>
              <span>(Latin: toward, enhanced)</span>
              <span className="text-acid-green">+</span>
              <span className="text-acid-green">agora</span>
              <span>(Greek: marketplace of ideas)</span>
            </div>
            <div className="flex justify-center gap-4 flex-wrap">
              <Link
                href="https://live.aragora.ai"
                className="px-6 py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors"
              >
                Watch Live
              </Link>
              <a
                href="https://pypi.org/project/aragora/"
                className="px-6 py-2 border border-acid-green/50 text-acid-green font-mono hover:bg-acid-green/10 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                PyPI v0.7.0
              </a>
              <a
                href="https://github.com/aragora"
                className="px-6 py-2 border border-acid-green/50 text-acid-green font-mono hover:bg-acid-green/10 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
          </div>
        </section>

        {/* Nomic Loop Section */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <div className="text-center mb-12">
              <h2 className="text-2xl font-mono text-acid-green mb-4">The Nomic Loop</h2>
              <p className="text-text-muted font-mono max-w-2xl mx-auto">
                aragora improves itself through structured cycles. Agents propose changes,
                debate their merits, implement them, and verify the results.
                The rules can change the rules.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-5 gap-4">
              {[
                { num: 1, name: 'Debate', desc: 'Multiple agents propose and critique improvements. Consensus emerges from structured discussion.' },
                { num: 2, name: 'Design', desc: 'Winning proposals are refined into actionable specifications with clear success criteria.' },
                { num: 3, name: 'Implement', desc: 'Code changes are generated and applied. Each modification is tracked and reversible.' },
                { num: 4, name: 'Verify', desc: 'Tests run, linters check, and agents evaluate. Only verified changes proceed.' },
                { num: 5, name: 'Commit', desc: 'Approved changes are committed. Failed cycles trigger rollback to last known good state.' },
              ].map((phase) => (
                <div key={phase.num} className="border border-acid-green/30 p-4 bg-bg/50">
                  <div className="text-acid-green font-mono text-xl mb-2">{phase.num}</div>
                  <div className="text-acid-cyan font-mono font-bold mb-2">{phase.name}</div>
                  <p className="text-text-muted text-xs font-mono">{phase.desc}</p>
                </div>
              ))}
            </div>

            <blockquote className="mt-8 text-center text-text-muted font-mono italic border-l-4 border-acid-green/50 pl-4 py-2">
              &quot;In Nomic, the rules can change the rules. aragora applies this to AI -
              the system debates and implements its own evolution.&quot;
            </blockquote>
          </div>
        </section>

        {/* Core Features */}
        <section className="py-16 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Core Features</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { title: 'Heterogeneous Agents', desc: 'Mix Claude, GPT, Gemini, Mistral (EU perspective), Grok, and Chinese models like DeepSeek, Qwen, and Kimi, plus local models. Different biases create stronger adversarial coverage.' },
                { title: 'Structured Debate', desc: 'Propose, Critique, Revise loop. Configurable rounds and consensus mechanisms (majority, unanimous, judge).' },
                { title: 'Evidence Provenance', desc: 'Cryptographic chain tracking sources. Every claim linked to evidence with reliability scoring.' },
                { title: 'Formal Verification', desc: 'Z3-powered proof checking. Verify logical claims with SMT solver integration.' },
                { title: 'Decision-to-PR Pipeline', desc: 'Turn debate outcomes into GitHub PRs. Risk registers, test plans, implementation specs.' },
                { title: 'Red-Team Mode', desc: 'Adversarial testing with steelman/strawman attacks. Find weaknesses before production.' },
              ].map((feature) => (
                <div key={feature.title} className="border border-acid-green/20 p-4 bg-surface/30">
                  <h3 className="text-acid-cyan font-mono text-sm mb-2">{feature.title}</h3>
                  <p className="text-text-muted text-xs font-mono">{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Why Adversarial Stress-Testing */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Why Adversarial Stress-Testing?</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
              <div className="border border-warning/30 p-6 bg-warning/5">
                <h3 className="text-warning font-mono mb-4">Single Model</h3>
                <ul className="space-y-2 text-text-muted text-sm font-mono">
                  <li className="flex items-center gap-2"><span className="text-warning">✕</span> One perspective</li>
                  <li className="flex items-center gap-2"><span className="text-warning">✕</span> Hallucinations go unchallenged</li>
                  <li className="flex items-center gap-2"><span className="text-warning">✕</span> Black box reasoning</li>
                  <li className="flex items-center gap-2"><span className="text-warning">✕</span> Single point of failure</li>
                  <li className="flex items-center gap-2"><span className="text-warning">✕</span> No dissent recorded</li>
                </ul>
              </div>
              <div className="border border-acid-green/30 p-6 bg-acid-green/5">
                <h3 className="text-acid-green font-mono mb-4">Aragora (AI Red Team)</h3>
                <ul className="space-y-2 text-text-muted text-sm font-mono">
                  <li className="flex items-center gap-2"><span className="text-acid-green">✓</span> Heterogeneous viewpoints</li>
                  <li className="flex items-center gap-2"><span className="text-acid-green">✓</span> Adversarial critique baked in</li>
                  <li className="flex items-center gap-2"><span className="text-acid-green">✓</span> Audit-ready stress-test transcript</li>
                  <li className="flex items-center gap-2"><span className="text-acid-green">✓</span> Decision receipts with dissent</li>
                  <li className="flex items-center gap-2"><span className="text-acid-green">✓</span> Minority views preserved</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* CLI Examples */}
        <section className="py-16 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Quick Start</h2>
            <div className="bg-bg border border-acid-green/30 p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-text-muted">
{`$ pip install aragora

$ aragora ask "Design a rate limiter for 1M req/sec"
[Round 1] claude_proposer: Token bucket with Redis cluster...
[Round 1] gemini_critic: Race condition in distributed counter
[Round 1] gpt_critic: Missing backpressure mechanism
[Round 2] claude_proposer: Revised with CAS operations...
[Round 2] gemini_critic: Addresses race condition ✓
[Round 2] gpt_critic: Added circuit breaker ✓
Consensus reached (confidence: 87%)

$ aragora nomic --cycles 3  # Run self-improvement loop
[Cycle 1] Debating improvements...
[Cycle 1] Implementing: "Add caching to debate storage"
[Cycle 1] Verified ✓ Committed`}
              </pre>
            </div>
          </div>
        </section>

        {/* Use Cases */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Use Cases</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {[
                { icon: '🔍', title: 'Code Review', desc: 'Adversarial security & quality analysis' },
                { icon: '🛠', title: 'System Design', desc: 'Stress-test architectural decisions' },
                { icon: '🔥', title: 'Incident Response', desc: 'Red-team RCA and mitigations' },
                { icon: '📚', title: 'Research Synthesis', desc: 'Challenge claims, surface risks' },
                { icon: '🔒', title: 'Security Testing', desc: 'AI red-team your proposals' },
                { icon: '✅', title: 'Decision Making', desc: 'Decision receipts with dissent tracking' },
              ].map((useCase) => (
                <div key={useCase.title} className="border border-acid-green/20 p-4 bg-bg/50 text-center">
                  <div className="text-2xl mb-2">{useCase.icon}</div>
                  <h3 className="text-acid-cyan font-mono text-sm mb-1">{useCase.title}</h3>
                  <p className="text-text-muted text-xs font-mono">{useCase.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Philosophical Foundations */}
        <section className="py-16 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-4 text-center">Philosophical Foundations</h2>
            <p className="text-center text-text-muted font-mono italic mb-8">
              &quot;Truth emerges from the marketplace of ideas, not central authority.&quot;
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { icon: '⚖', title: 'Voluntary Exchange', desc: 'Agents participate freely. Consensus emerges from debate.' },
                { icon: '🔗', title: 'Counter to Monolithic AI', desc: 'Alternative to "trust the one big model."' },
                { icon: '🌏', title: 'Decentralized Coordination', desc: 'No single authority decides truth.' },
                { icon: '💡', title: 'Emergent Order', desc: 'Better answers emerge from competition.' },
              ].map((principle) => (
                <div key={principle.title} className="border border-acid-green/20 p-4 bg-surface/30">
                  <div className="text-xl mb-2">{principle.icon}</div>
                  <h3 className="text-acid-cyan font-mono text-sm mb-2">{principle.title}</h3>
                  <p className="text-text-muted text-xs font-mono">{principle.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Technical Inspiration */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Technical Inspiration</h2>
            <div className="flex flex-wrap justify-center gap-4 text-sm font-mono">
              {[
                { name: 'Stanford Generative Agents', url: 'https://github.com/joonspk-research/generative_agents' },
                { name: 'LLM Multi-Agent Debate', url: 'https://github.com/Tsinghua-MARS-Lab/DebateLLM' },
                { name: 'ChatArena', url: 'https://github.com/chatarena/chatarena' },
                { name: 'Project Sid', url: 'https://github.com/' },
                { name: 'Nomic (Peter Suber)', url: 'https://en.wikipedia.org/wiki/Nomic' },
                { name: 'Agorism (SEK3)', url: 'https://en.wikipedia.org/wiki/Agorism' },
              ].map((ref) => (
                <a
                  key={ref.name}
                  href={ref.url}
                  className="px-3 py-1 border border-acid-green/30 text-acid-cyan hover:text-acid-green hover:border-acid-green/50 transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {ref.name}
                </a>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-acid-green/50 mb-4">
              {'═'.repeat(40)}
            </div>
            <div className="flex justify-center gap-6 mb-6">
              <Link href="https://live.aragora.ai" className="text-acid-cyan hover:text-acid-green transition-colors">
                Watch Live
              </Link>
              <a href="https://github.com/aragora" className="text-acid-cyan hover:text-acid-green transition-colors" target="_blank" rel="noopener noreferrer">
                View on GitHub
              </a>
              <a href="https://pypi.org/project/aragora/" className="text-acid-cyan hover:text-acid-green transition-colors" target="_blank" rel="noopener noreferrer">
                PyPI Package
              </a>
            </div>
            <p className="text-text-muted mb-2">
              Built for the open marketplace of ideas.
            </p>
            <p className="text-text-muted/60">
              Released under MIT License.
            </p>
            <div className="text-acid-green/50 mt-4">
              {'═'.repeat(40)}
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
