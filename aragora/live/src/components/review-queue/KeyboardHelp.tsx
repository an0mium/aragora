interface KeyboardHelpProps {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS = [
  ['j / k', 'Move down or up the queue'],
  ['Enter / Space', 'Expand or collapse the selected card'],
  ['a', 'Approve the selected PR'],
  ['r', 'Open request-changes composer'],
  ['d', 'Defer the selected PR for 4 hours'],
  ['o', 'Open the GitHub diff in a new tab'],
  ['?', 'Toggle this help overlay'],
];

export function KeyboardHelp({ open, onClose }: KeyboardHelpProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 px-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="w-full max-w-xl rounded-2xl border border-[var(--accent)]/25 bg-[linear-gradient(180deg,rgba(10,14,19,0.96),rgba(16,22,28,0.96))] p-6 shadow-[0_30px_80px_rgba(0,0,0,0.42)]"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="review-queue-shortcuts-title"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-theme-data uppercase tracking-[0.26em] text-[var(--accent)]">
              Keyboard
            </p>
            <h2 id="review-queue-shortcuts-title" className="mt-2 text-xl font-theme-data text-text">
              Settlement Shortcuts
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-border px-3 py-1 text-xs font-theme-data text-text-muted hover:border-[var(--accent)]/30 hover:text-text"
          >
            Close
          </button>
        </div>

        <div className="mt-5 grid gap-2">
          {SHORTCUTS.map(([keys, description]) => (
            <div
              key={keys}
              className="grid grid-cols-[8rem_1fr] gap-3 rounded-xl border border-[var(--accent)]/10 bg-bg/45 px-4 py-3"
            >
              <div className="text-sm font-theme-data text-[var(--acid-cyan)]">{keys}</div>
              <div className="text-sm font-theme-data text-text-muted">{description}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
