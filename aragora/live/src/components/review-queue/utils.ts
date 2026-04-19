import type { ReviewQueueItem, ReviewQueueVerdict } from './types';

export function verdictLabel(verdict: ReviewQueueVerdict | undefined): string {
  switch (verdict) {
    case 'approve_candidate':
      return 'APPROVE';
    case 'needs_human_attention':
      return 'ATTENTION';
    case 'repair_first':
      return 'REPAIR';
    default:
      return 'UNKNOWN';
  }
}

export function verdictTone(verdict: ReviewQueueVerdict | undefined): string {
  switch (verdict) {
    case 'approve_candidate':
      return 'border-emerald-400/40 bg-emerald-400/10 text-emerald-300';
    case 'needs_human_attention':
      return 'border-[var(--acid-yellow)]/40 bg-[var(--acid-yellow)]/10 text-[var(--acid-yellow)]';
    case 'repair_first':
      return 'border-acid-red/40 bg-acid-red/10 text-acid-red';
    default:
      return 'border-border bg-bg/40 text-text-muted';
  }
}

export function laneTone(lane: string): string {
  switch (lane) {
    case 'repairable':
      return 'text-acid-red';
    case 'needs_attention':
      return 'text-[var(--acid-yellow)]';
    case 'ready_now':
      return 'text-emerald-300';
    default:
      return 'text-text-muted';
  }
}

export function formatRelativeAge(value?: string | null): string {
  if (!value) return 'unknown age';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'unknown age';
  const deltaMs = Date.now() - date.getTime();
  const deltaHours = Math.max(0, deltaMs / (1000 * 60 * 60));
  if (deltaHours < 1) return '<1h old';
  if (deltaHours < 24) return `${Math.floor(deltaHours)}h old`;
  return `${Math.floor(deltaHours / 24)}d old`;
}

export function riskRank(item: ReviewQueueItem): number {
  if (item.machine_recommendation === 'repair_first') return 0;
  if (item.lane === 'repairable') return 1;
  if (item.machine_recommendation === 'needs_human_attention') return 2;
  if (item.lane === 'needs_attention') return 3;
  if (item.brief?.verdict === 'needs_human_attention') return 4;
  if (item.brief?.verdict === 'approve_candidate') return 5;
  return 6;
}

export function subsystemKey(item: ReviewQueueItem): string {
  return item.touched_subsystems[0] || 'zzz';
}
