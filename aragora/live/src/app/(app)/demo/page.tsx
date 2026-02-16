'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Phase = 'idle' | 'propose' | 'critique' | 'vote' | 'receipt';

interface AgentConfig {
  name: string;
  role: string;
  color: string;       // Tailwind-compatible CSS variable color
  borderColor: string;  // border highlight
  bgColor: string;      // subtle bg tint
}

interface Proposal {
  agent: string;
  text: string;
  evidence: string;
}

interface Critique {
  agent: string;
  target: string;
  severity: 'low' | 'medium' | 'high';
  text: string;
}

interface Vote {
  agent: string;
  choice: 'approve' | 'conditional' | 'reject';
  confidence: number;
  reason: string;
}

interface Receipt {
  id: string;
  question: string;
  verdict: string;
  confidence: number;
  consensus: string;
  dissent: { agent: string; text: string } | null;
  hash: string;
  timestamp: string;
}

interface MockScenario {
  question: string;
  proposals: Proposal[];
  critiques: Critique[];
  votes: Vote[];
  receipt: Receipt;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AGENTS: AgentConfig[] = [
  {
    name: 'Analyst',
    role: 'Domain Expert',
    color: 'var(--acid-cyan)',
    borderColor: 'var(--acid-cyan)',
    bgColor: 'rgba(0, 255, 255, 0.05)',
  },
  {
    name: 'Critic',
    role: 'Devil\'s Advocate',
    color: 'var(--acid-magenta)',
    borderColor: 'var(--acid-magenta)',
    bgColor: 'rgba(255, 0, 255, 0.05)',
  },
  {
    name: 'Judge',
    role: 'Consensus Builder',
    color: 'var(--acid-green)',
    borderColor: 'var(--acid-green)',
    bgColor: 'rgba(57, 255, 20, 0.05)',
  },
];

const PHASE_LABELS: { key: Phase; label: string }[] = [
  { key: 'propose', label: 'PROPOSE' },
  { key: 'critique', label: 'CRITIQUE' },
  { key: 'vote', label: 'VOTE' },
  { key: 'receipt', label: 'RECEIPT' },
];

const PRESET_QUESTIONS = [
  'Should we migrate from REST to GraphQL?',
  'Kubernetes vs staying on VMs?',
  'Should we adopt microservices architecture?',
  'Is it safe to use AI for medical diagnosis?',
];

// ---------------------------------------------------------------------------
// Mock Data Factory
// ---------------------------------------------------------------------------

function buildScenario(question: string): MockScenario {
  const scenarios: Record<string, MockScenario> = {
    'Should we migrate from REST to GraphQL?': {
      question,
      proposals: [
        {
          agent: 'Analyst',
          text: 'GraphQL offers significant advantages for our use case. Client-driven queries reduce over-fetching by an estimated 40% based on our current API traffic analysis. The strong type system provides better contract enforcement between frontend and backend teams, and the introspection capabilities simplify documentation.',
          evidence: 'Ref: Shopify migration study (2023) showed 35-45% reduction in payload sizes; Apollo Federation benchmarks demonstrate <5ms overhead per resolver.',
        },
        {
          agent: 'Critic',
          text: 'Migration introduces non-trivial complexity. N+1 query problems in GraphQL are well-documented and require DataLoader patterns. Our existing REST caching via CDN (Cloudflare) becomes significantly harder to replicate. The learning curve for the team (estimated 3-4 weeks) and tooling migration cost must be weighed against marginal gains.',
          evidence: 'Ref: GitHub API v4 post-mortem noted increased server CPU by 15-20% due to dynamic query parsing; Netflix reverted partial GraphQL adoption in 2022.',
        },
        {
          agent: 'Judge',
          text: 'A hybrid approach is optimal. Introduce GraphQL for the mobile BFF (Backend-for-Frontend) layer where query flexibility matters most, while maintaining REST for server-to-server communication where caching and simplicity are paramount. This limits migration risk to one surface area.',
          evidence: 'Ref: Airbnb hybrid pattern; Microsoft Graph API coexistence model with REST fallback endpoints.',
        },
      ],
      critiques: [
        {
          agent: 'Critic',
          target: 'Analyst',
          severity: 'high',
          text: 'Caching complexity underestimated for existing CDN infrastructure. Cloudflare and Varnish cache REST endpoints trivially via URL-based keys. GraphQL POST requests require custom cache key extraction, adding operational burden.',
        },
        {
          agent: 'Analyst',
          target: 'Critic',
          severity: 'medium',
          text: 'Netflix comparison is outdated; their 2022 issues stemmed from schema federation conflicts since resolved in Federation v2. The N+1 concern is valid but mitigated by persisted queries and DataLoader.',
        },
        {
          agent: 'Judge',
          target: 'Analyst',
          severity: 'low',
          text: 'The 40% over-fetching reduction assumes current endpoints are un-optimized. Sparse fieldsets on REST (JSON:API style) can achieve similar results with lower migration cost.',
        },
      ],
      votes: [
        { agent: 'Analyst', choice: 'conditional', confidence: 0.82, reason: 'Proceed with hybrid approach, starting with mobile BFF' },
        { agent: 'Critic', choice: 'conditional', confidence: 0.65, reason: 'Acceptable if CDN caching strategy is documented first' },
        { agent: 'Judge', choice: 'approve', confidence: 0.88, reason: 'Hybrid approach balances innovation with operational safety' },
      ],
      receipt: {
        id: 'DR-20260216-a3f8c2',
        question,
        verdict: 'Approved With Conditions',
        confidence: 0.78,
        consensus: 'Supermajority (2/3 agreed)',
        dissent: {
          agent: 'Critic',
          text: 'Caching complexity underestimated for existing CDN infrastructure. Require documented caching strategy before proceeding.',
        },
        hash: 'a3f8c2e91b4d7f0a2c8e5b3d6f9a1c4e',
        timestamp: '2026-02-16T14:32:07Z',
      },
    },
    'Kubernetes vs staying on VMs?': {
      question,
      proposals: [
        {
          agent: 'Analyst',
          text: 'Kubernetes provides auto-scaling, self-healing, and declarative infrastructure that reduces operational toil by 60%. Container orchestration enables blue-green deployments natively. With managed K8s offerings (EKS/GKE), the operational overhead is significantly reduced compared to 2020-era self-hosted clusters.',
          evidence: 'Ref: DORA 2025 report links container orchestration to 4x deployment frequency; Datadog container report shows 78% adoption in orgs >500 engineers.',
        },
        {
          agent: 'Critic',
          text: 'For our team size (12 engineers) and service count (8 services), Kubernetes is over-engineering. VM-based deployment with Ansible/Terraform achieves 90% of the benefits at 20% of the complexity. K8s introduces YAML sprawl, networking complexity (CNI plugins), and a steep debugging learning curve.',
          evidence: 'Ref: Basecamp "Majestic Monolith" approach; DHH cost analysis showing 60% cloud cost reduction by leaving K8s.',
        },
        {
          agent: 'Judge',
          text: 'Adopt a phased containerization strategy. Containerize applications with Docker first (gaining portability), deploy to a simple orchestrator like ECS or Nomad, and evaluate K8s adoption once the team reaches 20+ services or needs multi-region deployment.',
          evidence: 'Ref: Spotify incremental container adoption; HashiCorp Nomad simplicity benchmarks.',
        },
      ],
      critiques: [
        {
          agent: 'Critic',
          target: 'Analyst',
          severity: 'high',
          text: 'DORA metrics correlate but do not prove causation. Teams that adopt K8s tend to already have strong DevOps culture. The 60% toil reduction claim lacks control group data for teams our size.',
        },
        {
          agent: 'Analyst',
          target: 'Critic',
          severity: 'medium',
          text: 'Basecamp is an outlier case with a single-product company. The comparison ignores our multi-service architecture where service discovery and networking become primary pain points on bare VMs.',
        },
        {
          agent: 'Judge',
          target: 'Critic',
          severity: 'low',
          text: 'The "YAML sprawl" concern is mitigated by Helm charts and GitOps tooling (ArgoCD) which most production teams adopt.',
        },
      ],
      votes: [
        { agent: 'Analyst', choice: 'conditional', confidence: 0.75, reason: 'Support phased approach but timeline should target K8s within 12 months' },
        { agent: 'Critic', choice: 'approve', confidence: 0.72, reason: 'Phased containerization is pragmatic; avoids premature complexity' },
        { agent: 'Judge', choice: 'approve', confidence: 0.91, reason: 'Graduated approach matches team maturity and growth trajectory' },
      ],
      receipt: {
        id: 'DR-20260216-b7e4d1',
        question,
        verdict: 'Approved With Conditions',
        confidence: 0.79,
        consensus: 'Unanimous (3/3 agreed)',
        dissent: null,
        hash: 'b7e4d19f3a2c8e5b1d6f0a4c7e9b2f5d',
        timestamp: '2026-02-16T14:45:22Z',
      },
    },
    'Should we adopt microservices architecture?': {
      question,
      proposals: [
        {
          agent: 'Analyst',
          text: 'Microservices enable independent deployment cycles, technology heterogeneity, and fault isolation. For our growing platform (15+ developers, 3 teams), the monolith is becoming a coordination bottleneck. Service boundaries aligned with domain contexts (DDD) would reduce merge conflicts by an estimated 70%.',
          evidence: 'Ref: Amazon two-pizza team model; Uber domain-oriented microservice architecture (DOMA) case study.',
        },
        {
          agent: 'Critic',
          text: 'Distributed systems introduce exponential complexity: network partitions, distributed tracing, saga patterns for transactions, and eventual consistency. Our current monolith handles 50K RPM without issues. The "microservices premium" (observability, service mesh, API gateway) adds $15-25K/month in infrastructure costs.',
          evidence: 'Ref: Segment "Goodbye Microservices" blog; Kelsey Hightower\'s "Monoliths are the future" thesis.',
        },
        {
          agent: 'Judge',
          text: 'Extract the 2-3 highest-churn modules (payments, notifications, search) as services first. Keep the core domain as a well-structured modular monolith. This gives 80% of the organizational benefits with 30% of the operational complexity.',
          evidence: 'Ref: Shopify modular monolith success; Martin Fowler\'s "MonolithFirst" pattern.',
        },
      ],
      critiques: [
        {
          agent: 'Critic',
          target: 'Analyst',
          severity: 'high',
          text: 'The 70% merge conflict reduction assumes perfect service boundary decomposition, which rarely happens on first attempt. Poorly drawn boundaries create distributed monoliths -- the worst of both worlds.',
        },
        {
          agent: 'Analyst',
          target: 'Critic',
          severity: 'medium',
          text: 'The Segment case is from 2018 and pre-dates modern service mesh tooling (Istio, Linkerd). Infrastructure costs cited are for self-managed; managed offerings reduce this by 50%.',
        },
        {
          agent: 'Judge',
          target: 'Analyst',
          severity: 'low',
          text: 'The DDD alignment recommendation is sound but requires a domain modeling workshop (2-3 weeks) before any extraction begins.',
        },
      ],
      votes: [
        { agent: 'Analyst', choice: 'conditional', confidence: 0.80, reason: 'Agree with selective extraction; insist on domain modeling first' },
        { agent: 'Critic', choice: 'conditional', confidence: 0.68, reason: 'Only if rollback plan exists and modular monolith is the default path' },
        { agent: 'Judge', choice: 'approve', confidence: 0.85, reason: 'Selective extraction is the lowest-risk path to organizational scalability' },
      ],
      receipt: {
        id: 'DR-20260216-c9f2e3',
        question,
        verdict: 'Approved With Conditions',
        confidence: 0.77,
        consensus: 'Supermajority (2/3 agreed)',
        dissent: {
          agent: 'Critic',
          text: 'Distributed system complexity underestimated. Require rollback plan and modular monolith as default path before any extraction.',
        },
        hash: 'c9f2e3a71b5d8f0c2a6e4b3d9f1a7c5e',
        timestamp: '2026-02-16T15:01:44Z',
      },
    },
    'Is it safe to use AI for medical diagnosis?': {
      question,
      proposals: [
        {
          agent: 'Analyst',
          text: 'AI-assisted diagnosis has demonstrated superior accuracy in specific domains: dermatology (95.5% vs 87% human accuracy), radiology (lung nodule detection AUC 0.97), and pathology. FDA has cleared 800+ AI/ML medical devices. When positioned as clinical decision support (not replacement), AI augments physician capability and reduces diagnostic errors (est. 12 million Americans misdiagnosed annually).',
          evidence: 'Ref: Nature Medicine 2025 meta-analysis of 82 studies; FDA AI/ML Device Registry; Johns Hopkins patient safety study.',
        },
        {
          agent: 'Critic',
          text: 'Safety concerns are profound and multifaceted. Dataset bias leads to underperformance on underrepresented populations (Fitzpatrick skin types V-VI accuracy drops 15-20%). Liability frameworks are unresolved -- who is accountable when AI misdiagnoses? Black-box models lack explainability required for informed consent. Alert fatigue from false positives degrades physician trust over time.',
          evidence: 'Ref: MIT bias in dermatology AI study; AMA liability guidance gaps; EU AI Act high-risk classification for medical AI.',
        },
        {
          agent: 'Judge',
          text: 'Safe deployment requires a robust governance framework: (1) FDA-cleared systems only, (2) mandatory physician-in-the-loop for all diagnoses, (3) continuous bias monitoring with demographic stratification, (4) explainable AI requirements for clinical transparency, and (5) clear liability assignment. Under these conditions, AI diagnosis improves patient outcomes.',
          evidence: 'Ref: WHO AI Ethics Guidance 2024; NHS AI Lab deployment framework; OECD AI Policy Observatory healthcare guidelines.',
        },
      ],
      critiques: [
        {
          agent: 'Critic',
          target: 'Analyst',
          severity: 'high',
          text: 'The 95.5% accuracy figure is from controlled research settings with curated datasets. Real-world clinical deployment accuracy drops 10-15% due to image quality variation, patient demographics, and edge cases not represented in training data.',
        },
        {
          agent: 'Analyst',
          target: 'Critic',
          severity: 'medium',
          text: 'Bias concerns are valid but not unique to AI -- human diagnostic bias is well-documented (race, gender, age). AI systems can be audited and corrected systematically; human bias is harder to measure and remediate at scale.',
        },
        {
          agent: 'Judge',
          target: 'Critic',
          severity: 'low',
          text: 'Liability concern is being addressed by emerging legislation. The EU AI Act and proposed US HEAL Act establish clear accountability chains for AI-assisted medical decisions.',
        },
      ],
      votes: [
        { agent: 'Analyst', choice: 'conditional', confidence: 0.83, reason: 'Yes, with FDA clearance and physician oversight requirements' },
        { agent: 'Critic', choice: 'conditional', confidence: 0.58, reason: 'Only in narrow, well-validated domains with mandatory human review' },
        { agent: 'Judge', choice: 'conditional', confidence: 0.76, reason: 'Safe under comprehensive governance framework with continuous monitoring' },
      ],
      receipt: {
        id: 'DR-20260216-d4a1b7',
        question,
        verdict: 'Conditionally Safe',
        confidence: 0.72,
        consensus: 'Unanimous Conditional (3/3 agreed with conditions)',
        dissent: {
          agent: 'Critic',
          text: 'Real-world accuracy degradation and bias risks require narrow domain scoping and mandatory human-in-the-loop protocols.',
        },
        hash: 'd4a1b7c83e2f9a0d5b6c1e4f8a3d7b9e',
        timestamp: '2026-02-16T15:18:33Z',
      },
    },
  };

  return (
    scenarios[question] || {
      question,
      proposals: [
        {
          agent: 'Analyst',
          text: `After thorough analysis, there are strong arguments in favor of this direction. The potential benefits include improved efficiency, better scalability, and alignment with industry best practices. Key metrics from comparable implementations show 30-45% improvement in target outcomes.`,
          evidence: 'Ref: Industry benchmarks 2025; comparable case studies from similar organizations.',
        },
        {
          agent: 'Critic',
          text: `Significant risks warrant careful consideration. Implementation complexity, team readiness, and hidden costs often exceed initial estimates by 2-3x. The opportunity cost of pursuing this over alternatives must be evaluated against current priorities and resource constraints.`,
          evidence: 'Ref: Gartner implementation failure analysis; Standish Group CHAOS report data.',
        },
        {
          agent: 'Judge',
          text: `A measured, phased approach balances ambition with pragmatism. Start with a time-boxed pilot (4-6 weeks), establish clear success metrics, and make a go/no-go decision based on quantitative outcomes rather than theoretical arguments.`,
          evidence: 'Ref: Lean Startup validated learning methodology; Google DORA framework for measuring outcomes.',
        },
      ],
      critiques: [
        {
          agent: 'Critic',
          target: 'Analyst',
          severity: 'medium',
          text: 'The 30-45% improvement figure lacks context about baseline conditions and control methodology. Selection bias in comparable case studies may inflate expected outcomes.',
        },
        {
          agent: 'Analyst',
          target: 'Critic',
          severity: 'low',
          text: 'While implementation risks are real, the analysis overweights failure cases. Success cases in our specific domain context show more favorable risk-reward ratios.',
        },
        {
          agent: 'Judge',
          target: 'Analyst',
          severity: 'low',
          text: 'The evidence base would be strengthened by including failure mode analysis alongside success metrics.',
        },
      ],
      votes: [
        { agent: 'Analyst', choice: 'conditional', confidence: 0.76, reason: 'Proceed with defined success criteria' },
        { agent: 'Critic', choice: 'conditional', confidence: 0.62, reason: 'Acceptable with risk mitigation plan' },
        { agent: 'Judge', choice: 'approve', confidence: 0.84, reason: 'Pilot approach limits downside' },
      ],
      receipt: {
        id: `DR-20260216-${Math.random().toString(16).slice(2, 8)}`,
        question,
        verdict: 'Approved With Conditions',
        confidence: 0.74,
        consensus: 'Supermajority (2/3 agreed)',
        dissent: {
          agent: 'Critic',
          text: 'Risk mitigation plan required before proceeding. Success criteria must be defined quantitatively.',
        },
        hash: Math.random().toString(16).slice(2, 18) + Math.random().toString(16).slice(2, 18),
        timestamp: new Date().toISOString(),
      },
    }
  );
}

// ---------------------------------------------------------------------------
// Typing animation hook
// ---------------------------------------------------------------------------

function useTypingAnimation(text: string, active: boolean, speed = 12): string {
  const [displayed, setDisplayed] = useState('');
  const indexRef = useRef(0);

  useEffect(() => {
    if (!active) {
      setDisplayed('');
      indexRef.current = 0;
      return;
    }

    setDisplayed('');
    indexRef.current = 0;

    const interval = setInterval(() => {
      if (indexRef.current < text.length) {
        // Type in small chunks for performance
        const chunkSize = Math.max(1, Math.floor(Math.random() * 3) + 1);
        const nextIndex = Math.min(indexRef.current + chunkSize, text.length);
        setDisplayed(text.slice(0, nextIndex));
        indexRef.current = nextIndex;
      } else {
        clearInterval(interval);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, active, speed]);

  return active ? displayed : '';
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PhaseIndicator({ currentPhase }: { currentPhase: Phase }) {
  return (
    <div className="flex items-center gap-1 sm:gap-2 overflow-x-auto pb-2 sm:pb-0">
      {PHASE_LABELS.map((p, i) => {
        const phaseIndex = PHASE_LABELS.findIndex((pl) => pl.key === currentPhase);
        const thisIndex = i;
        const isActive = p.key === currentPhase;
        const isComplete = currentPhase !== 'idle' && thisIndex < phaseIndex;
        const isFuture = currentPhase === 'idle' || thisIndex > phaseIndex;

        return (
          <div key={p.key} className="flex items-center gap-1 sm:gap-2 shrink-0">
            {i > 0 && (
              <div
                className="w-4 sm:w-8 h-px transition-colors duration-500"
                style={{
                  backgroundColor: isComplete
                    ? 'var(--acid-green)'
                    : isFuture
                      ? 'var(--border)'
                      : 'var(--acid-green)',
                }}
              />
            )}
            <div
              className={`
                px-2 sm:px-3 py-1 text-xs font-mono border transition-all duration-500 whitespace-nowrap
                ${isActive
                  ? 'border-[var(--acid-green)] bg-[var(--acid-green)]/20 text-[var(--acid-green)] shadow-[0_0_10px_var(--acid-green)/30]'
                  : isComplete
                    ? 'border-[var(--acid-green)]/50 bg-[var(--acid-green)]/10 text-[var(--acid-green)]/80'
                    : 'border-[var(--border)] text-[var(--text-muted)]'
                }
              `}
            >
              {isComplete ? '\u2713 ' : ''}{p.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AgentCard({ agent }: { agent: AgentConfig }) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-2 border font-mono text-xs"
      style={{
        borderColor: agent.borderColor,
        backgroundColor: agent.bgColor,
      }}
    >
      <div
        className="w-2 h-2 rounded-full shrink-0"
        style={{ backgroundColor: agent.color }}
      />
      <div>
        <span style={{ color: agent.color }}>{agent.name}</span>
        <span className="text-[var(--text-muted)] ml-2">{agent.role}</span>
      </div>
    </div>
  );
}

function ProposalCard({
  proposal,
  agent,
  isTyping,
}: {
  proposal: Proposal;
  agent: AgentConfig;
  isTyping: boolean;
}) {
  const fullText = proposal.text;
  const typedText = useTypingAnimation(fullText, isTyping, 8);
  const displayText = isTyping ? typedText : fullText;
  const showCursor = isTyping && typedText.length < fullText.length;

  return (
    <div
      className="p-4 border transition-all duration-300"
      style={{
        borderColor: agent.borderColor,
        backgroundColor: agent.bgColor,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: agent.color }}
        />
        <span className="font-mono text-sm font-bold" style={{ color: agent.color }}>
          {agent.name}
        </span>
        <span className="text-xs text-[var(--text-muted)] font-mono">PROPOSAL</span>
      </div>
      <p className="text-sm text-[var(--text)] font-mono leading-relaxed mb-3">
        {displayText}
        {showCursor && (
          <span className="inline-block w-2 h-4 ml-0.5 bg-[var(--acid-green)] animate-pulse" />
        )}
      </p>
      {(!isTyping || typedText.length >= fullText.length) && (
        <div className="text-xs text-[var(--text-muted)] font-mono border-t border-[var(--border)] pt-2 mt-2 italic">
          {proposal.evidence}
        </div>
      )}
    </div>
  );
}

function CritiqueCard({ critique, agent }: { critique: Critique; agent: AgentConfig }) {
  const severityStyles: Record<string, string> = {
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    high: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  return (
    <div
      className="p-4 border transition-all duration-300"
      style={{
        borderColor: agent.borderColor,
        backgroundColor: agent.bgColor,
      }}
    >
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <div
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: agent.color }}
        />
        <span className="font-mono text-sm font-bold" style={{ color: agent.color }}>
          {critique.agent}
        </span>
        <span className="text-xs text-[var(--text-muted)] font-mono shrink-0">
          critiques {critique.target}
        </span>
        <span
          className={`px-2 py-0.5 text-xs font-mono border rounded ${severityStyles[critique.severity]}`}
        >
          {critique.severity.toUpperCase()}
        </span>
      </div>
      <p className="text-sm text-[var(--text)] font-mono leading-relaxed">
        &ldquo;{critique.text}&rdquo;
      </p>
    </div>
  );
}

function VoteBar({ vote, agent }: { vote: Vote; agent: AgentConfig }) {
  const choiceStyles: Record<string, { label: string; color: string; bg: string }> = {
    approve: { label: 'APPROVE', color: 'var(--acid-green)', bg: 'var(--acid-green)' },
    conditional: { label: 'CONDITIONAL', color: 'var(--acid-yellow)', bg: 'var(--acid-yellow)' },
    reject: { label: 'REJECT', color: 'var(--crimson)', bg: 'var(--crimson)' },
  };

  const style = choiceStyles[vote.choice];

  return (
    <div
      className="p-4 border transition-all duration-300"
      style={{
        borderColor: agent.borderColor,
        backgroundColor: agent.bgColor,
      }}
    >
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: agent.color }}
          />
          <span className="font-mono text-sm font-bold" style={{ color: agent.color }}>
            {vote.agent}
          </span>
        </div>
        <span
          className="px-2 py-0.5 text-xs font-mono font-bold"
          style={{ color: style.color }}
        >
          {style.label}
        </span>
      </div>
      {/* Confidence bar */}
      <div className="mb-2">
        <div className="flex justify-between text-xs font-mono text-[var(--text-muted)] mb-1">
          <span>Confidence</span>
          <span>{Math.round(vote.confidence * 100)}%</span>
        </div>
        <div className="h-2 bg-[var(--border)] overflow-hidden">
          <div
            className="h-full transition-all duration-1000 ease-out"
            style={{
              width: `${vote.confidence * 100}%`,
              backgroundColor: style.bg,
              opacity: 0.7,
            }}
          />
        </div>
      </div>
      <p className="text-xs text-[var(--text-muted)] font-mono italic">{vote.reason}</p>
    </div>
  );
}

function ReceiptCard({ receipt }: { receipt: Receipt }) {
  return (
    <div className="border-2 border-[var(--acid-green)] bg-[var(--acid-green)]/5 p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-[var(--acid-green)] rounded-full" />
          <span className="font-mono text-lg font-bold text-[var(--acid-green)]">
            DECISION RECEIPT
          </span>
        </div>
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {receipt.id}
        </span>
      </div>

      {/* Question */}
      <div>
        <span className="text-xs font-mono text-[var(--text-muted)] block mb-1">Question</span>
        <p className="font-mono text-sm text-[var(--text)]">{receipt.question}</p>
      </div>

      {/* Verdict */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-3 bg-[var(--acid-green)]/10 border border-[var(--acid-green)]/30">
        <div>
          <span className="text-xs font-mono text-[var(--text-muted)] block mb-1">Verdict</span>
          <span className="font-mono text-lg font-bold text-[var(--acid-green)]">
            {receipt.verdict}
          </span>
        </div>
        <div className="text-right">
          <span className="text-xs font-mono text-[var(--text-muted)] block mb-1">Confidence</span>
          <span className="font-mono text-lg font-bold text-[var(--acid-green)]">
            {Math.round(receipt.confidence * 100)}%
          </span>
        </div>
      </div>

      {/* Consensus */}
      <div>
        <span className="text-xs font-mono text-[var(--text-muted)] block mb-1">Consensus</span>
        <span className="font-mono text-sm text-[var(--acid-cyan)]">{receipt.consensus}</span>
      </div>

      {/* Dissent */}
      {receipt.dissent && (
        <div className="p-3 bg-[var(--acid-magenta)]/5 border border-[var(--acid-magenta)]/30">
          <span className="text-xs font-mono text-[var(--acid-magenta)] block mb-1">
            Dissenting View ({receipt.dissent.agent})
          </span>
          <p className="font-mono text-sm text-[var(--text)] italic">
            &ldquo;{receipt.dissent.text}&rdquo;
          </p>
        </div>
      )}

      {/* Hash */}
      <div className="border-t border-[var(--border)] pt-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs font-mono text-[var(--text-muted)]">
          <span>SHA-256: {receipt.hash.slice(0, 16)}...</span>
          <span>{new Date(receipt.timestamp).toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function DemoPage() {
  const [question, setQuestion] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [scenario, setScenario] = useState<MockScenario | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Track which items have been revealed
  const [revealedProposals, setRevealedProposals] = useState<number[]>([]);
  const [typingProposal, setTypingProposal] = useState<number | null>(null);
  const [revealedCritiques, setRevealedCritiques] = useState<number[]>([]);
  const [revealedVotes, setRevealedVotes] = useState<number[]>([]);
  const [showReceipt, setShowReceipt] = useState(false);

  const resultsRef = useRef<HTMLDivElement>(null);
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach(clearTimeout);
    };
  }, []);

  const scheduleTimeout = useCallback((fn: () => void, delay: number) => {
    const id = setTimeout(fn, delay);
    timeoutsRef.current.push(id);
    return id;
  }, []);

  const resetState = useCallback(() => {
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];
    setPhase('idle');
    setScenario(null);
    setIsRunning(false);
    setRevealedProposals([]);
    setTypingProposal(null);
    setRevealedCritiques([]);
    setRevealedVotes([]);
    setShowReceipt(false);
  }, []);

  const scrollToResults = useCallback(() => {
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  const startDebate = useCallback((q: string) => {
    resetState();
    const s = buildScenario(q);
    setScenario(s);
    setQuestion(q);
    setIsRunning(true);

    let delay = 300;

    // --- PROPOSE phase ---
    scheduleTimeout(() => {
      setPhase('propose');
      scrollToResults();
    }, delay);

    delay += 600;

    // Reveal proposals one by one with typing animation
    s.proposals.forEach((_, i) => {
      scheduleTimeout(() => {
        setTypingProposal(i);
        setRevealedProposals((prev) => [...prev, i]);
        scrollToResults();
      }, delay);
      delay += 2500; // time for typing animation per proposal
    });

    // Finish typing the last proposal
    scheduleTimeout(() => {
      setTypingProposal(null);
    }, delay);
    delay += 300;

    // --- CRITIQUE phase ---
    scheduleTimeout(() => {
      setPhase('critique');
      scrollToResults();
    }, delay);
    delay += 500;

    s.critiques.forEach((_, i) => {
      scheduleTimeout(() => {
        setRevealedCritiques((prev) => [...prev, i]);
        scrollToResults();
      }, delay);
      delay += 1000;
    });

    delay += 300;

    // --- VOTE phase ---
    scheduleTimeout(() => {
      setPhase('vote');
      scrollToResults();
    }, delay);
    delay += 500;

    s.votes.forEach((_, i) => {
      scheduleTimeout(() => {
        setRevealedVotes((prev) => [...prev, i]);
        scrollToResults();
      }, delay);
      delay += 800;
    });

    delay += 500;

    // --- RECEIPT phase ---
    scheduleTimeout(() => {
      setPhase('receipt');
      setShowReceipt(true);
      scrollToResults();
    }, delay);

    scheduleTimeout(() => {
      setIsRunning(false);
    }, delay + 500);
  }, [resetState, scheduleTimeout, scrollToResults]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!question.trim() || isRunning) return;
      startDebate(question.trim());
    },
    [question, isRunning, startDebate],
  );

  const handlePreset = useCallback(
    (q: string) => {
      if (isRunning) return;
      setQuestion(q);
      startDebate(q);
    },
    [isRunning, startDebate],
  );

  const getAgentConfig = (name: string): AgentConfig =>
    AGENTS.find((a) => a.name === name) || AGENTS[0];

  return (
    <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
      <div className="max-w-4xl mx-auto px-4 py-8 sm:py-12 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h1 className="text-2xl sm:text-3xl font-mono font-bold text-[var(--acid-green)]">
                {'>'} DEBATE DEMO
              </h1>
              <p className="text-[var(--text-muted)] font-mono text-sm mt-1">
                Watch 3 AI agents adversarially vet a decision in real-time
              </p>
            </div>
            <Link
              href="/"
              className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors border border-[var(--border)] px-3 py-1.5 hover:border-[var(--acid-green)]/50"
            >
              [BACK TO HOME]
            </Link>
          </div>
        </div>

        {/* Agent Roster */}
        <div className="space-y-2">
          <span className="text-xs font-mono text-[var(--text-muted)] uppercase tracking-wider">
            Agent Panel
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {AGENTS.map((agent) => (
              <AgentCard key={agent.name} agent={agent} />
            ))}
          </div>
        </div>

        {/* Input Section */}
        <div className="space-y-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question for the agents to debate..."
              disabled={isRunning}
              className="flex-1 px-4 py-3 text-sm font-mono bg-[var(--surface)] border border-[var(--border)]
                       text-[var(--text)] placeholder-[var(--text-muted)]/50
                       focus:border-[var(--acid-green)] focus:outline-none
                       disabled:opacity-50 transition-colors"
            />
            <button
              type="submit"
              disabled={isRunning || !question.trim()}
              className="px-4 sm:px-6 py-3 font-mono text-sm font-bold
                       bg-[var(--acid-green)] text-[var(--bg)]
                       hover:bg-[var(--acid-green)]/80 transition-colors
                       disabled:opacity-30 disabled:cursor-not-allowed whitespace-nowrap"
            >
              {isRunning ? 'DEBATING...' : 'START DEBATE'}
            </button>
          </form>

          {/* Preset Questions */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-[var(--text-muted)]">
              Try a preset:
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESET_QUESTIONS.map((pq) => (
                <button
                  key={pq}
                  onClick={() => handlePreset(pq)}
                  disabled={isRunning}
                  className="px-3 py-1.5 text-xs font-mono border border-[var(--acid-green)]/30
                           text-[var(--text-muted)] hover:text-[var(--acid-green)]
                           hover:border-[var(--acid-green)]/60 hover:bg-[var(--acid-green)]/5
                           transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-left"
                >
                  {pq}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Debate Visualization */}
        {phase !== 'idle' && scenario && (
          <div ref={resultsRef} className="space-y-6">
            {/* Phase Progress */}
            <div className="p-4 bg-[var(--surface)] border border-[var(--border)]">
              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                <span className="text-xs font-mono text-[var(--text-muted)] uppercase tracking-wider">
                  Debate Progress
                </span>
                {isRunning && (
                  <span className="flex items-center gap-2 text-xs font-mono text-[var(--acid-green)]">
                    <span className="w-2 h-2 bg-[var(--acid-green)] rounded-full animate-pulse" />
                    LIVE
                  </span>
                )}
                {!isRunning && phase === 'receipt' && (
                  <span className="text-xs font-mono text-[var(--acid-green)]">
                    COMPLETE
                  </span>
                )}
              </div>
              <PhaseIndicator currentPhase={phase} />
            </div>

            {/* PROPOSE Phase */}
            {(phase === 'propose' || phase === 'critique' || phase === 'vote' || phase === 'receipt') && (
              <div className="space-y-3">
                <h2 className="text-sm font-mono text-[var(--acid-cyan)] uppercase tracking-wider flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[var(--acid-cyan)] rounded-full" />
                  Proposals
                </h2>
                <div className="space-y-3">
                  {scenario.proposals.map((proposal, i) =>
                    revealedProposals.includes(i) ? (
                      <ProposalCard
                        key={i}
                        proposal={proposal}
                        agent={getAgentConfig(proposal.agent)}
                        isTyping={typingProposal === i}
                      />
                    ) : null,
                  )}
                </div>
              </div>
            )}

            {/* CRITIQUE Phase */}
            {(phase === 'critique' || phase === 'vote' || phase === 'receipt') && (
              <div className="space-y-3">
                <h2 className="text-sm font-mono text-[var(--acid-magenta)] uppercase tracking-wider flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[var(--acid-magenta)] rounded-full" />
                  Critiques
                </h2>
                <div className="space-y-3">
                  {scenario.critiques.map((critique, i) =>
                    revealedCritiques.includes(i) ? (
                      <CritiqueCard
                        key={i}
                        critique={critique}
                        agent={getAgentConfig(critique.agent)}
                      />
                    ) : null,
                  )}
                </div>
              </div>
            )}

            {/* VOTE Phase */}
            {(phase === 'vote' || phase === 'receipt') && (
              <div className="space-y-3">
                <h2 className="text-sm font-mono text-[var(--acid-yellow)] uppercase tracking-wider flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[var(--acid-yellow)] rounded-full" />
                  Votes
                </h2>
                <div className="space-y-3">
                  {scenario.votes.map((vote, i) =>
                    revealedVotes.includes(i) ? (
                      <VoteBar
                        key={i}
                        vote={vote}
                        agent={getAgentConfig(vote.agent)}
                      />
                    ) : null,
                  )}
                </div>
              </div>
            )}

            {/* RECEIPT Phase */}
            {phase === 'receipt' && showReceipt && (
              <div className="space-y-3">
                <h2 className="text-sm font-mono text-[var(--acid-green)] uppercase tracking-wider flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[var(--acid-green)] rounded-full" />
                  Decision Receipt
                </h2>
                <ReceiptCard receipt={scenario.receipt} />
              </div>
            )}

            {/* Reset button (after completion) */}
            {!isRunning && phase === 'receipt' && (
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-4">
                <button
                  onClick={resetState}
                  className="px-6 py-2 font-mono text-sm border border-[var(--acid-green)]/30
                           text-[var(--acid-green)] hover:bg-[var(--acid-green)]/10 transition-colors"
                >
                  [NEW DEBATE]
                </button>
                <Link
                  href="/arena"
                  className="px-6 py-2 font-mono text-sm bg-[var(--acid-green)] text-[var(--bg)]
                           hover:bg-[var(--acid-green)]/80 transition-colors"
                >
                  TRY REAL DEBATE
                </Link>
              </div>
            )}
          </div>
        )}

        {/* Idle state info */}
        {phase === 'idle' && (
          <div className="border border-[var(--acid-green)]/20 bg-[var(--surface)]/30 p-6 sm:p-8 text-center space-y-4">
            <div className="text-3xl sm:text-4xl font-mono text-[var(--acid-green)]/30">
              {'{ }'}
            </div>
            <h2 className="text-lg font-mono text-[var(--acid-green)]">
              How Aragora Works
            </h2>
            <div className="max-w-2xl mx-auto text-left space-y-3 text-sm font-mono text-[var(--text-muted)]">
              <div className="flex gap-3">
                <span className="text-[var(--acid-cyan)] shrink-0">1.</span>
                <span>
                  <strong className="text-[var(--text)]">PROPOSE</strong> -- Each agent independently analyzes the question and submits a reasoned proposal with evidence.
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-[var(--acid-magenta)] shrink-0">2.</span>
                <span>
                  <strong className="text-[var(--text)]">CRITIQUE</strong> -- Agents adversarially challenge each other&apos;s reasoning, flagging weaknesses and biases.
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-[var(--acid-yellow)] shrink-0">3.</span>
                <span>
                  <strong className="text-[var(--text)]">VOTE</strong> -- Each agent casts a vote with a confidence level, forming consensus or recording dissent.
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-[var(--acid-green)] shrink-0">4.</span>
                <span>
                  <strong className="text-[var(--text)]">RECEIPT</strong> -- A cryptographically-hashed decision receipt is generated with full audit trail.
                </span>
              </div>
            </div>
            <p className="text-xs font-mono text-[var(--text-muted)]/60 pt-2">
              Select a preset question above or type your own to see it in action.
            </p>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-xs font-mono text-[var(--text-muted)]/60 pt-4 border-t border-[var(--border)]">
          This is a simulated demo. In production, 30+ heterogeneous AI agents debate across multiple rounds with real evidence retrieval.
        </div>
      </div>
    </main>
  );
}
