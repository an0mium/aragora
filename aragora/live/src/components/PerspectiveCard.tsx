'use client';

// Role-to-color mapping for the 6 landing page analyst roles
const ROLE_COLORS: Record<string, { border: string; text: string; glow: string }> = {
  'strategic analyst': { border: 'var(--acid-cyan)', text: 'var(--acid-cyan)', glow: 'rgba(0, 255, 255, 0.08)' },
  "devil's advocate": { border: 'var(--crimson)', text: 'var(--crimson)', glow: 'rgba(255, 0, 64, 0.08)' },
  'implementation expert': { border: 'var(--acid-green)', text: 'var(--acid-green)', glow: 'rgba(57, 255, 20, 0.08)' },
  'industry analyst': { border: 'var(--purple)', text: 'var(--purple)', glow: 'rgba(191, 0, 255, 0.08)' },
  'risk assessor': { border: 'var(--gold, #ffd700)', text: 'var(--gold, #ffd700)', glow: 'rgba(255, 215, 0, 0.08)' },
  'synthesizer': { border: 'var(--acid-magenta)', text: 'var(--acid-magenta)', glow: 'rgba(255, 0, 255, 0.08)' },
};

function getRoleStyle(agentName: string) {
  const lower = agentName.toLowerCase();
  for (const [key, style] of Object.entries(ROLE_COLORS)) {
    if (lower.includes(key)) return style;
  }
  // Fallback: try matching legacy agent names
  if (lower.includes('analyst') || lower.includes('supportive')) return ROLE_COLORS['strategic analyst'];
  if (lower.includes('critic') || lower.includes('critical')) return ROLE_COLORS["devil's advocate"];
  if (lower.includes('balanced') || lower.includes('moderator')) return ROLE_COLORS['implementation expert'];
  if (lower.includes('contrarian')) return ROLE_COLORS['risk assessor'];
  if (lower.includes('synthesizer')) return ROLE_COLORS['synthesizer'];
  return { border: 'var(--acid-cyan)', text: 'var(--acid-cyan)', glow: 'rgba(0, 255, 255, 0.08)' };
}

// Extract a short role label from the agent name
function getRoleLabel(agentName: string): string {
  const lower = agentName.toLowerCase();
  for (const key of Object.keys(ROLE_COLORS)) {
    if (lower.includes(key)) return key.split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
  }
  // Fallback labels for legacy agent names
  if (lower.includes('analyst') || lower.includes('supportive')) return 'Strategic Analyst';
  if (lower.includes('critic') || lower.includes('critical')) return "Devil's Advocate";
  if (lower.includes('balanced') || lower.includes('moderator')) return 'Implementation Expert';
  if (lower.includes('contrarian')) return 'Risk Assessor';
  if (lower.includes('synthesizer')) return 'Synthesizer';
  return agentName;
}

interface PerspectiveCardProps {
  agentName: string;
  content: string;
  isFullWidth?: boolean;
}

export function PerspectiveCard({ agentName, content, isFullWidth }: PerspectiveCardProps) {
  const style = getRoleStyle(agentName);
  const label = getRoleLabel(agentName);

  return (
    <div
      className={`border border-[var(--border)] bg-[var(--surface)] p-4 transition-all duration-200 hover:shadow-lg ${
        isFullWidth ? 'col-span-full' : ''
      }`}
      style={{
        borderLeftWidth: '3px',
        borderLeftColor: style.border,
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 20px ${style.glow}`;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = 'none';
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ backgroundColor: style.border }}
        />
        <span className="text-xs font-bold font-mono uppercase tracking-wider" style={{ color: style.text }}>
          {label}
        </span>
        <span className="text-[10px] font-mono text-[var(--text-muted)] opacity-60">
          {agentName !== label.toLowerCase() ? agentName : ''}
        </span>
      </div>
      <p className="text-sm text-[var(--text)] whitespace-pre-wrap leading-relaxed font-mono">
        {content}
      </p>
    </div>
  );
}
