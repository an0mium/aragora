'use client';

interface Fact {
  id: string;
  content: string;
  confidence: number;
  verified: boolean;
  source?: string;
  created_at: string;
}

interface FactsBrowserProps {
  facts: Fact[];
  onPageChange: (page: number) => void;
  totalPages: number;
  currentPage: number;
  loading?: boolean;
}

function confidenceColor(c: number): string {
  if (c > 0.8) return 'text-acid-green border-acid-green/30 bg-acid-green/10';
  if (c > 0.5) return 'text-acid-yellow border-acid-yellow/30 bg-acid-yellow/10';
  return 'text-crimson border-crimson/30 bg-crimson/10';
}

export function FactsBrowser({
  facts,
  onPageChange,
  totalPages,
  currentPage,
  loading = false,
}: FactsBrowserProps) {
  if (loading) {
    return (
      <div className="card p-4">
        <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} KNOWLEDGE FACTS</h3>
        <div className="animate-pulse space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-surface rounded" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card p-4">
      <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} KNOWLEDGE FACTS</h3>

      {facts.length === 0 ? (
        <p className="text-text-muted text-sm font-mono text-center py-8">No data available</p>
      ) : (
        <>
          {/* Facts list */}
          <div className="space-y-2 mb-4">
            {facts.map((fact) => (
              <div
                key={fact.id}
                className="border border-acid-green/10 rounded p-3 hover:bg-acid-green/5 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-text font-mono text-sm flex-1 leading-relaxed">
                    {fact.content}
                  </p>
                  <div className="flex items-center gap-2 shrink-0">
                    {/* Verified indicator */}
                    {fact.verified && (
                      <span
                        className="text-acid-green text-xs font-mono border border-acid-green/30 px-1.5 py-0.5 rounded"
                        title="Verified"
                      >
                        V
                      </span>
                    )}
                    {/* Confidence badge */}
                    <span
                      className={`text-xs font-mono px-1.5 py-0.5 rounded border ${confidenceColor(fact.confidence)}`}
                    >
                      {(fact.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3 mt-2">
                  {fact.source && (
                    <span className="text-text-muted text-xs font-mono">
                      src: {fact.source}
                    </span>
                  )}
                  <span className="text-text-muted text-xs font-mono">
                    {new Date(fact.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 pt-3 border-t border-acid-green/10">
              <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage <= 0}
                className="px-3 py-1 text-xs font-mono border border-acid-green/20 rounded disabled:opacity-30 disabled:cursor-not-allowed hover:bg-acid-green/10 text-text-muted hover:text-acid-green transition-colors"
              >
                {'<'} PREV
              </button>
              <span className="text-text-muted text-xs font-mono px-3">
                {currentPage + 1} / {totalPages}
              </span>
              <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage >= totalPages - 1}
                className="px-3 py-1 text-xs font-mono border border-acid-green/20 rounded disabled:opacity-30 disabled:cursor-not-allowed hover:bg-acid-green/10 text-text-muted hover:text-acid-green transition-colors"
              >
                NEXT {'>'}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default FactsBrowser;
