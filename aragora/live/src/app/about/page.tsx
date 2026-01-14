'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBanner } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

// Practical use cases with real examples
const PRACTICAL_USE_CASES = [
  {
    icon: '🏗️',
    title: 'Architecture Stress-Test',
    subtitle: 'Find critical flaws before launch',
    examples: [
      'MedConnect: Found 11 issues, prevented 3 critical outages',
      'Scale review: 50K to 500K concurrent users validated',
    ],
    cta: { label: 'See Case Study', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/architecture-stress-test.md' },
  },
  {
    icon: '🔐',
    title: 'API Security Review',
    subtitle: 'AI red-team your endpoints',
    examples: [
      'CloudPay: 7 critical issues found in 23 minutes',
      'BOLA, rate limiting gaps, PCI-DSS violations caught',
    ],
    cta: { label: 'See Case Study', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/security-api-review.md' },
  },
  {
    icon: '📋',
    title: 'Compliance Audit',
    subtitle: 'GDPR, HIPAA, SOC 2 readiness',
    examples: [
      'Automated persona-based compliance checks',
      'Audit-ready transcripts with minority views preserved',
    ],
    cta: { label: 'See Case Study', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/gdpr-compliance-audit.md' },
  },
  {
    icon: '🔍',
    title: 'Code Review',
    subtitle: 'Multi-model adversarial analysis',
    examples: [
      'Security vulnerabilities, logic errors, edge cases',
      'Cross-model consensus on critical issues',
    ],
    cta: null,
  },
  {
    icon: '🔥',
    title: 'Incident Response',
    subtitle: 'Red-team RCA and mitigations',
    examples: [
      'Root cause analysis with competing hypotheses',
      'Mitigation strategies stress-tested by adversarial agents',
    ],
    cta: null,
  },
  {
    icon: '⚖️',
    title: 'Decision Making',
    subtitle: 'Defensible decisions with receipts',
    examples: [
      'Decision transcripts with dissenting views recorded',
      'Confidence scores and evidence chains for audit',
    ],
    cta: null,
  },
];

// Document library categories
const DOC_CATEGORIES = [
  {
    title: 'Getting Started',
    icon: '🚀',
    docs: [
      { name: 'Quick Start', href: 'https://github.com/aragora/aragora/blob/main/docs/QUICKSTART.md' },
      { name: 'Installation', href: 'https://github.com/aragora/aragora/blob/main/docs/GETTING_STARTED.md' },
      { name: 'CLI Reference', href: 'https://github.com/aragora/aragora/blob/main/docs/CLI_REFERENCE.md' },
    ],
  },
  {
    title: 'Core Features',
    icon: '⚙️',
    docs: [
      { name: 'Nomic Loop', href: 'https://github.com/aragora/aragora/blob/main/docs/NOMIC_LOOP.md' },
      { name: 'Gauntlet', href: 'https://github.com/aragora/aragora/blob/main/docs/GAUNTLET.md' },
      { name: 'Graph Debates', href: 'https://github.com/aragora/aragora/blob/main/docs/GRAPH_DEBATES.md' },
      { name: 'Matrix Debates', href: 'https://github.com/aragora/aragora/blob/main/docs/MATRIX_DEBATES.md' },
    ],
  },
  {
    title: 'API & Integration',
    icon: '🔌',
    docs: [
      { name: 'API Reference', href: 'https://github.com/aragora/aragora/blob/main/docs/API_REFERENCE.md' },
      { name: 'API Examples', href: 'https://github.com/aragora/aragora/blob/main/docs/API_EXAMPLES.md' },
      { name: 'TypeScript SDK', href: 'https://github.com/aragora/aragora/blob/main/docs/SDK_TYPESCRIPT.md' },
      { name: 'MCP Integration', href: 'https://github.com/aragora/aragora/blob/main/docs/MCP_GUIDE.md' },
    ],
  },
  {
    title: 'Agent Development',
    icon: '🤖',
    docs: [
      { name: 'Custom Agents', href: 'https://github.com/aragora/aragora/blob/main/docs/CUSTOM_AGENTS.md' },
      { name: 'Agent Selection', href: 'https://github.com/aragora/aragora/blob/main/docs/AGENT_SELECTION.md' },
      { name: 'Formal Verification', href: 'https://github.com/aragora/aragora/blob/main/docs/FORMAL_VERIFICATION.md' },
    ],
  },
  {
    title: 'Deployment',
    icon: '🚢',
    docs: [
      { name: 'Deployment Guide', href: 'https://github.com/aragora/aragora/blob/main/docs/DEPLOYMENT.md' },
      { name: 'Production Checklist', href: 'https://github.com/aragora/aragora/blob/main/docs/PRODUCTION_CHECKLIST.md' },
      { name: 'Scaling', href: 'https://github.com/aragora/aragora/blob/main/docs/SCALING.md' },
    ],
  },
  {
    title: 'Case Studies',
    icon: '📊',
    docs: [
      { name: 'Architecture Stress-Test', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/architecture-stress-test.md' },
      { name: 'API Security Review', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/security-api-review.md' },
      { name: 'GDPR Compliance', href: 'https://github.com/aragora/aragora/blob/main/docs/case-studies/gdpr-compliance-audit.md' },
    ],
  },
];

// Platform capabilities
const CAPABILITIES = [
  { label: 'AI Providers', value: '13+', desc: 'Claude, GPT, Gemini, Mistral, Grok, DeepSeek, Qwen, Kimi...' },
  { label: 'Consensus Algorithms', value: '7', desc: 'Majority, unanimous, weighted, judge, supermajority...' },
  { label: 'Debate Types', value: '3', desc: 'Standard, graph, matrix' },
  { label: 'Memory Tiers', value: '4', desc: 'Fast, medium, slow, glacial' },
  { label: 'REST Endpoints', value: '134+', desc: 'Full API coverage' },
  { label: 'Test Definitions', value: '27K+', desc: 'Comprehensive coverage' },
];

export default function AboutPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="text-acid-green font-mono font-bold hover:text-acid-cyan transition-colors">
              [ARAGORA]
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

        {/* Hero with Full ASCII Art */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            <AsciiBanner subtitle="documentation" showStatus={false} />

            <div className="text-center mt-8">
              <p className="text-xl text-acid-cyan font-mono mb-4">Self-Improving AI Red Team</p>
              <p className="text-text-muted font-mono mb-6 max-w-2xl mx-auto">
                Multi-agent debate with autonomous self-modification.
                Watch AI agents reason, critique, and stress-test decisions in real-time.
              </p>

              {/* Etymology */}
              <div className="flex justify-center items-center gap-2 text-xs text-text-muted font-mono mb-8">
                <span className="text-acid-green">ar-</span>
                <span>(Latin: toward, enhanced)</span>
                <span className="text-acid-green">+</span>
                <span className="text-acid-green">agora</span>
                <span>(Greek: marketplace of ideas)</span>
              </div>

              {/* Action Buttons */}
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
                  PyPI v0.9.0
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
          </div>
        </section>

        {/* Platform Capabilities */}
        <section className="py-8 px-4 bg-surface/20">
          <div className="container mx-auto max-w-5xl">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              {CAPABILITIES.map((cap) => (
                <div key={cap.label} className="text-center p-3 border border-acid-green/20 bg-bg/50">
                  <div className="text-2xl font-mono text-acid-green font-bold">{cap.value}</div>
                  <div className="text-xs font-mono text-acid-cyan">{cap.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Nomic Loop Section */}
        <section className="py-16 px-4">
          <div className="container mx-auto max-w-4xl">
            <div className="text-center mb-12">
              <h2 className="text-2xl font-mono text-acid-green mb-4">The Nomic Loop</h2>
              <p className="text-text-muted font-mono max-w-2xl mx-auto">
                Aragora improves itself through structured cycles. Agents propose changes,
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
              &quot;In Nomic, the rules can change the rules. Aragora applies this to AI -
              the system debates and implements its own evolution.&quot;
            </blockquote>
          </div>
        </section>

        {/* Practical Use Cases */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-5xl">
            <h2 className="text-2xl font-mono text-acid-green mb-4 text-center">Practical Use Cases</h2>
            <p className="text-center text-text-muted font-mono mb-8 max-w-2xl mx-auto">
              Real-world applications with measurable results
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {PRACTICAL_USE_CASES.map((useCase) => (
                <div key={useCase.title} className="border border-acid-green/20 p-5 bg-bg/50 flex flex-col">
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-2xl">{useCase.icon}</span>
                    <div>
                      <h3 className="text-acid-cyan font-mono text-sm font-bold">{useCase.title}</h3>
                      <p className="text-text-muted/70 text-xs font-mono">{useCase.subtitle}</p>
                    </div>
                  </div>
                  <ul className="space-y-1.5 flex-grow">
                    {useCase.examples.map((example, i) => (
                      <li key={i} className="text-text-muted text-xs font-mono flex items-start gap-2">
                        <span className="text-acid-green mt-0.5">•</span>
                        <span>{example}</span>
                      </li>
                    ))}
                  </ul>
                  {useCase.cta && (
                    <a
                      href={useCase.cta.href}
                      className="mt-4 text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors inline-flex items-center gap-1"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {useCase.cta.label} →
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Why Adversarial Stress-Testing */}
        <section className="py-16 px-4">
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

        {/* Document Library */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-5xl">
            <h2 className="text-2xl font-mono text-acid-green mb-4 text-center">Document Library</h2>
            <p className="text-center text-text-muted font-mono mb-8">
              Comprehensive documentation for every aspect of the platform
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {DOC_CATEGORIES.map((category) => (
                <div key={category.title} className="border border-acid-green/20 p-4 bg-bg/50">
                  <h3 className="text-acid-cyan font-mono text-sm mb-3 flex items-center gap-2">
                    <span>{category.icon}</span>
                    {category.title}
                  </h3>
                  <ul className="space-y-2">
                    {category.docs.map((doc) => (
                      <li key={doc.name}>
                        <a
                          href={doc.href}
                          className="text-text-muted text-xs font-mono hover:text-acid-green transition-colors flex items-center gap-2"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <span className="text-acid-green/50">→</span>
                          {doc.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className="mt-8 text-center">
              <a
                href="https://github.com/aragora/aragora/tree/main/docs"
                className="inline-flex items-center gap-2 px-6 py-2 border border-acid-green/50 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                View All Documentation →
              </a>
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

        {/* Core Features */}
        <section className="py-16 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Core Features</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { title: 'Heterogeneous Agents', desc: 'Mix Claude, GPT, Gemini, Mistral (EU perspective), Grok, and Chinese models like DeepSeek, Qwen, and Kimi. Different biases create stronger adversarial coverage.' },
                { title: 'Structured Debate', desc: 'Propose, Critique, Revise loop. Configurable rounds and consensus mechanisms (majority, unanimous, judge).' },
                { title: 'Evidence Provenance', desc: 'Cryptographic chain tracking sources. Every claim linked to evidence with reliability scoring.' },
                { title: 'Formal Verification', desc: 'Z3-powered proof checking. Verify logical claims with SMT solver integration.' },
                { title: 'Decision-to-PR Pipeline', desc: 'Turn debate outcomes into GitHub PRs. Risk registers, test plans, implementation specs.' },
                { title: 'Red-Team Mode', desc: 'Adversarial testing with steelman/strawman attacks. Find weaknesses before production.' },
              ].map((feature) => (
                <div key={feature.title} className="border border-acid-green/20 p-4 bg-bg/50">
                  <h3 className="text-acid-cyan font-mono text-sm mb-2">{feature.title}</h3>
                  <p className="text-text-muted text-xs font-mono">{feature.desc}</p>
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
              {'═'.repeat(50)}
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
              {'═'.repeat(50)}
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
