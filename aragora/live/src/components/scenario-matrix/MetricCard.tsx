'use client';

export interface MetricCardProps {
  label: string;
  value: string | number;
  color?: string;
}

export function MetricCard({
  label,
  value,
  color = 'text-acid-green',
}: MetricCardProps) {
  return (
    <div className="bg-bg/50 border border-acid-green/20 p-3 text-center">
      <div className="text-xs font-mono text-text-muted mb-1">{label}</div>
      <div className={`text-lg font-mono ${color}`}>{value}</div>
    </div>
  );
}

export default MetricCard;
