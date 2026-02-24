import { useState, useEffect, useRef, useCallback } from 'react';
import { apiFetch } from '@/lib/api';

interface BrainDumpPreview {
  themes: string[];
  ideaCount: number;
  urgencySignals: string[];
  isLoading: boolean;
}

const DEBOUNCE_MS = 500;
const MIN_TEXT_LENGTH = 20;

/**
 * Debounced brain dump preview hook. Sends text to backend for
 * theme detection and idea count estimation after a delay.
 */
export function useBrainDumpPreview(text: string): BrainDumpPreview {
  const [preview, setPreview] = useState<BrainDumpPreview>({
    themes: [],
    ideaCount: 0,
    urgencySignals: [],
    isLoading: false,
  });
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchPreview = useCallback(async (input: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPreview((prev) => ({ ...prev, isLoading: true }));
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data: any = await apiFetch('/api/v1/canvas/pipeline/from-braindump', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input, preview_only: true }),
        signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setPreview({
          themes: data.themes || [],
          ideaCount: data.ideas_parsed || data.idea_count || 0,
          urgencySignals: data.urgency_signals || [],
          isLoading: false,
        });
      }
    } catch {
      if (!controller.signal.aborted) {
        setPreview((prev) => ({ ...prev, isLoading: false }));
      }
    }
  }, []);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);

    if (text.trim().length < MIN_TEXT_LENGTH) {
      setPreview({ themes: [], ideaCount: 0, urgencySignals: [], isLoading: false });
      return;
    }

    timerRef.current = setTimeout(() => {
      fetchPreview(text);
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [text, fetchPreview]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return preview;
}
