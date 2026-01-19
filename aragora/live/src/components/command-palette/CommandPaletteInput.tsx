'use client';

import { useRef, useEffect } from 'react';

interface CommandPaletteInputProps {
  value: string;
  onChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isSearching: boolean;
  resultCount: number;
  selectedIndex: number;
}

/**
 * CommandPaletteInput
 *
 * Search input for the command palette with ARIA attributes
 * for accessibility.
 */
export function CommandPaletteInput({
  value,
  onChange,
  onKeyDown,
  isSearching,
  resultCount,
  selectedIndex,
}: CommandPaletteInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus on mount
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <div className="flex items-center gap-2 px-4 py-3 border-b border-acid-green/20 bg-bg/50">
      {/* Search icon / prompt */}
      <span className="text-acid-green font-mono text-sm flex-shrink-0">
        {isSearching ? '...' : '>'}
      </span>

      {/* Input */}
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-expanded="true"
        aria-controls="command-palette-results"
        aria-activedescendant={
          resultCount > 0 ? `command-palette-item-${selectedIndex}` : undefined
        }
        aria-autocomplete="list"
        aria-label="Search commands and navigation"
        placeholder="Search or type a command..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        className="flex-1 bg-transparent border-none outline-none text-text font-mono text-sm placeholder:text-text-muted"
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck="false"
      />

      {/* Keyboard hint */}
      <kbd className="px-2 py-0.5 text-xs font-mono text-text-muted border border-acid-green/30 rounded flex-shrink-0">
        esc
      </kbd>
    </div>
  );
}

export default CommandPaletteInput;
