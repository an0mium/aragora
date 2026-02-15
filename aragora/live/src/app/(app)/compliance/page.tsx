'use client';

/**
 * EU AI Act Compliance Demo
 *
 * Interactive demo showing Aragora's EU AI Act compliance artifact generation.
 * Users can classify AI use cases by risk level and generate Article 12/13/14
 * compliance artifacts from decision receipts.
 */

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useAuth } from '@/context/AuthContext';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RiskClassification {
  risk_level: 'unacceptable' | 'high' | 'limited' | 'minimal';
  annex_iii_categories: string[];
  applicable_articles: string[];
  matched_keywords: string[];
  confidence: number;
}

interface ArticleAssessment {
  article: string;
  title: string;
  status: 'compliant' | 'partial' | 'non_compliant' | 'not_applicable';
  findings: string[];
  recommendations: string[];
}

interface ComplianceBundle {
  bundle_id: string;
  generated_at: string;
  integrity_hash: string;
  article_12: Record<string, unknown>;
  article_13: Record<string, unknown>;
  article_14: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Demo Data
// ---------------------------------------------------------------------------

const SAMPLE_USE_CASES = [
  {
    label: 'Hiring Decision AI',
    description:
      'AI system that screens job applicants, scores resumes, and recommends candidates for interview based on historical hiring data and job requirements.',
  },
  {
    label: 'Clinical Decision Support',
    description:
      'AI-assisted diagnostic system that analyzes medical imaging (X-rays, MRIs) and patient history to suggest differential diagnoses for emergency department physicians.',
  },
  {
    label: 'Credit Risk Assessment',
    description:
      'Automated creditworthiness scoring system that evaluates loan applications using financial history, employment data, and behavioral patterns to determine approval and interest rates.',
  },
  {
    label: 'Content Recommendation',
    description:
      'AI system that recommends articles and videos to users based on browsing history and engagement patterns for a news aggregation platform.',
  },
];

const RISK_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  unacceptable: { bg: 'bg-acid-red/10', text: 'text-acid-red', border: 'border-acid-red/40' },
  high: { bg: 'bg-acid-orange/10', text: 'text-acid-orange', border: 'border-acid-orange/40' },
  limited: { bg: 'bg-acid-yellow/10', text: 'text-acid-yellow', border: 'border-acid-yellow/40' },
  minimal: { bg: 'bg-acid-green/10', text: 'text-acid-green', border: 'border-acid-green/40' },
};

const STATUS_ICONS: Record<string, string> = {
  compliant: '[PASS]',
  partial: '[WARN]',
  non_compliant: '[FAIL]',
  not_applicable: '[N/A]',
};

const STATUS_COLORS: Record<string, string> = {
  compliant: 'text-acid-green',
  partial: 'text-acid-yellow',
  non_compliant: 'text-acid-red',
  not_applicable: 'text-muted',
};

// ---------------------------------------------------------------------------
// Fallback demo data
// ---------------------------------------------------------------------------

function getDemoClassification(description: string): RiskClassification {
  const lower = description.toLowerCase();
  const isHigh =
    lower.includes('hiring') ||
    lower.includes('credit') ||
    lower.includes('medical') ||
    lower.includes('diagnostic') ||
    lower.includes('loan');

  if (isHigh) {
    return {
      risk_level: 'high',
      annex_iii_categories: lower.includes('hiring')
        ? ['Employment, workers management']
        : lower.includes('credit') || lower.includes('loan')
          ? ['Access to essential private and public services']
          : ['Biometrics'],
      applicable_articles: ['Article 9', 'Article 12', 'Article 13', 'Article 14', 'Article 15'],
      matched_keywords: lower.includes('hiring')
        ? ['hiring', 'applicant', 'screening']
        : lower.includes('credit')
          ? ['credit', 'loan', 'scoring']
          : ['diagnostic', 'medical', 'imaging'],
      confidence: 0.92,
    };
  }

  return {
    risk_level: 'minimal',
    annex_iii_categories: [],
    applicable_articles: ['Article 52'],
    matched_keywords: [],
    confidence: 0.85,
  };
}

function getDemoAssessments(): ArticleAssessment[] {
  return [
    {
      article: 'Article 9',
      title: 'Risk Management System',
      status: 'compliant',
      findings: [
        'Multi-agent debate provides systematic risk identification through adversarial challenge',
        'Decision receipts document risk assessment outcomes with confidence scores',
      ],
      recommendations: [],
    },
    {
      article: 'Article 12',
      title: 'Record-Keeping & Logging',
      status: 'compliant',
      findings: [
        'Event logging captures all debate phases: propose, critique, revise, vote, synthesize',
        'Audit trail includes agent identities, timestamps, and content hashes',
        'Retention policy configurable per deployment',
      ],
      recommendations: [],
    },
    {
      article: 'Article 13',
      title: 'Transparency & Information',
      status: 'partial',
      findings: [
        'System provides decision explanations via explainability module',
        'Agent model identities disclosed in receipts',
      ],
      recommendations: [
        'Document known limitations for each model provider in user-facing materials',
        'Add accuracy metrics per domain to transparency disclosures',
      ],
    },
    {
      article: 'Article 14',
      title: 'Human Oversight',
      status: 'compliant',
      findings: [
        'Human-in-the-loop approval supported via receipt gating',
        'Override and stop mechanisms available through debate controls',
        'Automation bias safeguards via contrarian agent and dissent tracking',
      ],
      recommendations: [],
    },
    {
      article: 'Article 15',
      title: 'Accuracy, Robustness, Cybersecurity',
      status: 'partial',
      findings: [
        'Multi-model consensus reduces correlated failure modes',
        'Calibration tracking monitors prediction accuracy over time',
      ],
      recommendations: [
        'Implement formal adversarial robustness testing on a quarterly cadence',
        'Add bias monitoring metrics per protected characteristic',
      ],
    },
  ];
}

function getDemoBundle(): ComplianceBundle {
  const now = new Date().toISOString();
  return {
    bundle_id: `CAB-${Date.now().toString(36).toUpperCase()}`,
    generated_at: now,
    integrity_hash: 'sha256:' + Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join(''),
    article_12: {
      event_log: {
        total_events: 47,
        event_types: ['debate_start', 'proposal', 'critique', 'vote', 'consensus', 'receipt_generated'],
        retention_days: 365,
      },
      technical_documentation: {
        annex_iv_sections: ['System description', 'Design specifications', 'Risk management', 'Testing procedures'],
        completeness: 0.85,
      },
    },
    article_13: {
      provider_identity: { name: 'Aragora Platform', contact: 'compliance@aragora.ai' },
      intended_purpose: 'Multi-agent adversarial debate for decision integrity',
      known_risks: ['Model provider outages', 'Training data biases in individual models', 'Prompt injection attacks'],
      output_interpretation: 'Decision receipts with confidence scores, dissent trails, and consensus proofs',
    },
    article_14: {
      oversight_model: 'Human-on-the-loop with override capability',
      bias_safeguards: [
        'Heterogeneous model consensus (different training data)',
        'Contrarian agent prevents groupthink',
        'Dissent tracking surfaces disagreements',
      ],
      override_mechanisms: ['Debate pause/resume', 'Agent removal', 'Manual verdict override', 'Kill switch'],
    },
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CompliancePage() {
  const { config: backendConfig } = useBackend();
  const { tokens } = useAuth();

  // Risk classifier state
  const [useCase, setUseCase] = useState('');
  const [classification, setClassification] = useState<RiskClassification | null>(null);
  const [classifying, setClassifying] = useState(false);

  // Conformity report state
  const [assessments, setAssessments] = useState<ArticleAssessment[]>([]);
  const [showAssessments, setShowAssessments] = useState(false);

  // Bundle state
  const [bundle, setBundle] = useState<ComplianceBundle | null>(null);
  const [generatingBundle, setGeneratingBundle] = useState(false);
  const [activeArticle, setActiveArticle] = useState<'12' | '13' | '14'>('12');

  const classify = useCallback(async () => {
    if (!useCase.trim()) return;
    setClassifying(true);
    setClassification(null);
    setShowAssessments(false);
    setBundle(null);

    try {
      const response = await fetch(`${backendConfig.api}/api/v2/compliance/eu-ai-act/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({ description: useCase }),
      });

      if (response.ok) {
        const data = await response.json();
        setClassification(data.classification || data);
      } else {
        setClassification(getDemoClassification(useCase));
      }
    } catch {
      setClassification(getDemoClassification(useCase));
    } finally {
      setClassifying(false);
    }
  }, [useCase, backendConfig.api, tokens?.access_token]);

  const generateReport = useCallback(async () => {
    setShowAssessments(true);

    try {
      const response = await fetch(`${backendConfig.api}/api/v2/compliance/eu-ai-act/audit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          receipt: {
            question: useCase,
            verdict: 'approved_with_conditions',
            confidence: 0.78,
            consensus: { reached: true, method: 'majority' },
            agents: ['anthropic', 'openai', 'mistral'],
            rounds_used: 2,
          },
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const report = data.conformity_report || data;
        if (report.assessments) {
          setAssessments(report.assessments);
          return;
        }
      }
      setAssessments(getDemoAssessments());
    } catch {
      setAssessments(getDemoAssessments());
    }
  }, [useCase, backendConfig.api, tokens?.access_token]);

  const generateBundle = useCallback(async () => {
    setGeneratingBundle(true);

    try {
      const response = await fetch(`${backendConfig.api}/api/v2/compliance/eu-ai-act/generate-bundle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          receipt: {
            question: useCase,
            verdict: 'approved_with_conditions',
            confidence: 0.78,
            consensus: { reached: true, method: 'majority' },
            agents: ['anthropic', 'openai', 'mistral'],
            rounds_used: 2,
          },
          provider_name: 'Aragora Platform',
          system_name: 'Decision Integrity Engine',
          system_version: '1.0',
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setBundle(data.bundle || data);
      } else {
        setBundle(getDemoBundle());
      }
    } catch {
      setBundle(getDemoBundle());
    } finally {
      setGeneratingBundle(false);
    }
  }, [useCase, backendConfig.api, tokens?.access_token]);

  const riskStyle = classification ? RISK_COLORS[classification.risk_level] || RISK_COLORS.minimal : null;

  return (
    <div className="min-h-screen bg-background">
      <Scanlines />
      <CRTVignette />

      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="hover:text-accent">
              <AsciiBannerCompact />
            </Link>
            <span className="text-muted font-mono text-sm">{'//'} EU AI ACT COMPLIANCE</span>
          </div>
          <div className="flex items-center gap-3">
            <BackendSelector />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 max-w-4xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-mono mb-2">EU AI ACT COMPLIANCE</h1>
          <p className="text-muted text-sm font-mono">
            Enforcement begins August 2, 2026. Generate compliance artifacts as a byproduct of
            decision vetting.
          </p>
        </div>

        <PanelErrorBoundary panelName="Risk Classification">
          {/* Step 1: Risk Classification */}
          <section className="card p-6 mb-6">
            <h2 className="text-lg font-mono mb-1">1. CLASSIFY AI USE CASE</h2>
            <p className="text-xs text-muted font-mono mb-4">
              Describe your AI system to determine its risk category under the EU AI Act
            </p>

            {/* Sample use cases */}
            <div className="flex flex-wrap gap-2 mb-4">
              {SAMPLE_USE_CASES.map((sample) => (
                <button
                  key={sample.label}
                  onClick={() => setUseCase(sample.description)}
                  className="px-3 py-1 text-xs font-mono border border-border rounded hover:border-accent hover:text-accent transition-colors"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            <textarea
              value={useCase}
              onChange={(e) => setUseCase(e.target.value)}
              placeholder="Describe your AI system's purpose and functionality..."
              rows={3}
              className="w-full bg-background border border-border rounded p-3 font-mono text-sm focus:border-accent focus:outline-none resize-none"
            />

            <button
              onClick={classify}
              disabled={!useCase.trim() || classifying}
              className="mt-3 px-4 py-2 text-sm font-mono bg-accent/10 text-accent border border-accent/40 rounded hover:bg-accent/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {classifying ? 'CLASSIFYING...' : 'CLASSIFY RISK LEVEL'}
            </button>

            {/* Classification Result */}
            {classification && riskStyle && (
              <div className={`mt-4 border rounded p-4 ${riskStyle.bg} ${riskStyle.border}`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-lg font-mono font-bold ${riskStyle.text}`}>
                    {classification.risk_level.toUpperCase()} RISK
                  </span>
                  <span className="text-xs font-mono text-muted">
                    Confidence: {(classification.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                {classification.annex_iii_categories.length > 0 && (
                  <div className="mb-2">
                    <span className="text-xs font-mono text-muted">Annex III Categories:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {classification.annex_iii_categories.map((cat, i) => (
                        <span key={i} className="px-2 py-0.5 text-xs font-mono bg-background/50 rounded">
                          {cat}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="mb-2">
                  <span className="text-xs font-mono text-muted">Applicable Articles:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {classification.applicable_articles.map((art, i) => (
                      <span key={i} className="px-2 py-0.5 text-xs font-mono bg-background/50 rounded">
                        {art}
                      </span>
                    ))}
                  </div>
                </div>

                {classification.matched_keywords.length > 0 && (
                  <div>
                    <span className="text-xs font-mono text-muted">Matched Keywords:</span>
                    <span className="text-xs font-mono ml-2">
                      {classification.matched_keywords.join(', ')}
                    </span>
                  </div>
                )}

                {classification.risk_level === 'high' && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <button
                      onClick={generateReport}
                      className="px-4 py-2 text-sm font-mono bg-accent/10 text-accent border border-accent/40 rounded hover:bg-accent/20 transition-colors"
                    >
                      GENERATE CONFORMITY ASSESSMENT
                    </button>
                  </div>
                )}
              </div>
            )}
          </section>
        </PanelErrorBoundary>

        {/* Step 2: Conformity Assessment */}
        {showAssessments && (
          <PanelErrorBoundary panelName="Conformity Assessment">
            <section className="card p-6 mb-6">
              <h2 className="text-lg font-mono mb-1">2. CONFORMITY ASSESSMENT</h2>
              <p className="text-xs text-muted font-mono mb-4">
                Article-by-article compliance status based on your decision receipt
              </p>

              <div className="space-y-3">
                {assessments.map((assessment) => (
                  <details key={assessment.article} className="border border-border rounded overflow-hidden">
                    <summary className="p-3 bg-surface/50 cursor-pointer hover:bg-surface transition-colors flex items-center justify-between">
                      <span className="font-mono text-sm">
                        {assessment.article}: {assessment.title}
                      </span>
                      <span className={`font-mono text-xs ${STATUS_COLORS[assessment.status]}`}>
                        {STATUS_ICONS[assessment.status]}
                      </span>
                    </summary>
                    <div className="p-3 border-t border-border text-sm">
                      {assessment.findings.length > 0 && (
                        <div className="mb-2">
                          <span className="text-xs font-mono text-muted">Findings:</span>
                          <ul className="mt-1 space-y-1">
                            {assessment.findings.map((f, i) => (
                              <li key={i} className="text-xs font-mono pl-3 border-l-2 border-acid-green/40">
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {assessment.recommendations.length > 0 && (
                        <div>
                          <span className="text-xs font-mono text-muted">Recommendations:</span>
                          <ul className="mt-1 space-y-1">
                            {assessment.recommendations.map((r, i) => (
                              <li key={i} className="text-xs font-mono pl-3 border-l-2 border-acid-yellow/40">
                                {r}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </details>
                ))}
              </div>

              <button
                onClick={generateBundle}
                disabled={generatingBundle}
                className="mt-4 px-4 py-2 text-sm font-mono bg-accent/10 text-accent border border-accent/40 rounded hover:bg-accent/20 disabled:opacity-50 transition-colors"
              >
                {generatingBundle ? 'GENERATING...' : 'GENERATE FULL ARTIFACT BUNDLE'}
              </button>
            </section>
          </PanelErrorBoundary>
        )}

        {/* Step 3: Artifact Bundle */}
        {bundle && (
          <PanelErrorBoundary panelName="Compliance Bundle">
            <section className="card p-6 mb-6">
              <h2 className="text-lg font-mono mb-1">3. COMPLIANCE ARTIFACT BUNDLE</h2>
              <div className="flex items-center gap-4 mb-4">
                <span className="text-xs font-mono text-muted">
                  Bundle: {bundle.bundle_id}
                </span>
                <span className="text-xs font-mono text-muted">
                  Generated: {new Date(bundle.generated_at).toLocaleString()}
                </span>
              </div>
              <div className="text-xs font-mono text-muted mb-4 break-all">
                Integrity: {bundle.integrity_hash}
              </div>

              {/* Article tabs */}
              <div className="flex border-b border-border mb-4">
                {(['12', '13', '14'] as const).map((art) => (
                  <button
                    key={art}
                    onClick={() => setActiveArticle(art)}
                    className={`px-4 py-2 text-sm font-mono border-b-2 transition-colors ${
                      activeArticle === art
                        ? 'border-accent text-accent'
                        : 'border-transparent text-muted hover:text-text'
                    }`}
                  >
                    Article {art}
                  </button>
                ))}
              </div>

              {/* Article content */}
              <div className="bg-background border border-border rounded p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap overflow-auto max-h-96">
                  {JSON.stringify(
                    activeArticle === '12'
                      ? bundle.article_12
                      : activeArticle === '13'
                        ? bundle.article_13
                        : bundle.article_14,
                    null,
                    2
                  )}
                </pre>
              </div>

              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${bundle.bundle_id}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="px-3 py-1.5 text-xs font-mono bg-surface border border-border rounded hover:border-accent transition-colors"
                >
                  DOWNLOAD JSON
                </button>
              </div>
            </section>
          </PanelErrorBoundary>
        )}

        {/* Info section */}
        <section className="card p-6 border-accent/20">
          <h3 className="text-sm font-mono mb-3 text-accent">HOW IT WORKS</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-muted">
            <div>
              <div className="text-text mb-1">1. Run a Debate</div>
              Submit any question to Aragora&apos;s multi-agent debate engine. Agents propose, critique,
              and vote.
            </div>
            <div>
              <div className="text-text mb-1">2. Get a Receipt</div>
              Every debate produces a cryptographic decision receipt with consensus proofs and dissent
              trails.
            </div>
            <div>
              <div className="text-text mb-1">3. Generate Artifacts</div>
              The receipt automatically produces Article 12/13/14 compliance artifacts for regulators.
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-border bg-surface/50 py-4 mt-8">
        <div className="container mx-auto px-4 flex items-center justify-between text-xs text-muted font-mono">
          <span>EU AI Act enforcement: August 2, 2026</span>
          <div className="flex items-center gap-4">
            <Link href="/audit" className="hover:text-accent">
              AUDIT
            </Link>
            <Link href="/receipts" className="hover:text-accent">
              RECEIPTS
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
