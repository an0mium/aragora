'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface VoiceInputProps {
  debateId: string;
  onTranscript?: (text: string, isFinal: boolean) => void;
  onError?: (error: string) => void;
  apiBase?: string;
  disabled?: boolean;
  /** Optional callback to send transcript as debate suggestion */
  sendSuggestion?: (suggestion: string) => void;
  /** Auto-submit final transcripts as suggestions (requires sendSuggestion) */
  autoSubmitSuggestion?: boolean;
}

type VoiceStatus = 'idle' | 'connecting' | 'recording' | 'processing' | 'error';

interface TranscriptSegment {
  text: string;
  timestamp: number;
  isFinal: boolean;
}

export function VoiceInput({
  debateId,
  onTranscript,
  onError,
  apiBase = '',
  disabled = false,
  sendSuggestion,
  autoSubmitSuggestion = false,
}: VoiceInputProps) {
  const [status, setStatus] = useState<VoiceStatus>('idle');
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [currentText, setCurrentText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const startTimeRef = useRef<number>(0);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const statusRef = useRef<VoiceStatus>(status);
  const stopRecordingRef = useRef<() => void>(() => {});
  const handleErrorRef = useRef<(message: string) => void>(() => {});
  const handleServerMessageRef = useRef<(data: Record<string, unknown>) => void>(() => {});
  const startAudioCaptureRef = useRef<() => void>(() => {});
  const startDurationTimerRef = useRef<() => void>(() => {});

  // Keep statusRef in sync
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // Build WebSocket URL
  const getWsUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = apiBase ? new URL(apiBase).host : window.location.host;
    return `${protocol}//${host}/ws/voice/${debateId}`;
  }, [apiBase, debateId]);

  // Start recording
  const startRecording = useCallback(async () => {
    if (disabled || status === 'recording') return;

    setError(null);
    setStatus('connecting');

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      // Connect to WebSocket
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        // Send config
        ws.send(JSON.stringify({
          type: 'config',
          format: 'pcm',
          sample_rate: 16000,
          channels: 1,
          bits_per_sample: 16,
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleServerMessageRef.current(data);
        } catch {
          console.error('Failed to parse WebSocket message:', event.data);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        handleErrorRef.current('Connection error');
      };

      ws.onclose = () => {
        if (statusRef.current === 'recording') {
          stopRecordingRef.current();
        }
      };

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to access microphone';
      handleErrorRef.current(message);
    }
  }, [disabled, status, getWsUrl]);

  // Handle server messages
  const handleServerMessage = useCallback((data: Record<string, unknown>) => {
    switch (data.type) {
      case 'ready':
        setSessionId(data.session_id as string);
        setStatus('recording');
        startTimeRef.current = Date.now();
        startAudioCaptureRef.current();
        startDurationTimerRef.current();
        break;

      case 'transcript':
        const text = data.text as string;
        const isFinal = data.is_final as boolean;

        if (isFinal) {
          setTranscript((prev) => [...prev, {
            text,
            timestamp: Date.now(),
            isFinal: true,
          }]);
          setCurrentText('');

          // Auto-submit as debate suggestion if enabled
          if (autoSubmitSuggestion && sendSuggestion && text.trim()) {
            sendSuggestion(text.trim());
          }
        } else {
          setCurrentText(text);
        }

        onTranscript?.(text, isFinal);
        break;

      case 'error':
        handleErrorRef.current(data.message as string || 'Unknown error');
        break;

      case 'warning':
        // eslint-disable-next-line no-console
        console.warn('Voice warning:', data.message);
        break;
    }
  }, [onTranscript, autoSubmitSuggestion, sendSuggestion]);

  // Handle errors
  const handleError = useCallback((message: string) => {
    setError(message);
    setStatus('error');
    onError?.(message);
    stopRecordingRef.current();
  }, [onError]);

  // Start audio capture using Web Audio API
  const startAudioCapture = useCallback(() => {
    if (!mediaStreamRef.current) return;

    try {
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(mediaStreamRef.current);

      // Use ScriptProcessorNode for audio processing
      // Note: This is deprecated but widely supported. AudioWorklet is the modern alternative.
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) return;

        const inputData = event.inputBuffer.getChannelData(0);

        // Convert Float32 samples to Int16
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Send as binary
        wsRef.current.send(int16Data.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    } catch (err) {
      console.error('Audio capture error:', err);
      handleErrorRef.current('Failed to capture audio');
    }
  }, []);

  // Start duration timer
  const startDurationTimer = useCallback(() => {
    durationIntervalRef.current = setInterval(() => {
      setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  }, []);

  // Stop recording
  const stopRecording = useCallback(() => {
    // Stop duration timer
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }

    // Stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'end' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus('idle');
    setSessionId(null);
  }, []);

  // Keep refs in sync for use in callbacks
  stopRecordingRef.current = stopRecording;
  handleErrorRef.current = handleError;
  handleServerMessageRef.current = handleServerMessage;
  startAudioCaptureRef.current = startAudioCapture;
  startDurationTimerRef.current = startDurationTimer;

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRecordingRef.current();
    };
  }, []);

  // Format duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get full transcript text
  const getFullTranscript = () => {
    return transcript.map((s) => s.text).join(' ');
  };

  return (
    <div className="panel" style={{ padding: 0 }}>
      <div className="p-4 border-b border-border">
        <h3 className="panel-title-sm flex items-center gap-2">
          <span>Voice Input</span>
          {status === 'recording' && (
            <span className="flex items-center gap-1 text-crimson text-xs">
              <span className="w-2 h-2 bg-crimson rounded-full animate-pulse" />
              {formatDuration(duration)}
            </span>
          )}
        </h3>
      </div>

      <div className="p-4 space-y-4">
        {/* Control buttons */}
        <div className="flex items-center gap-3">
          {status === 'idle' || status === 'error' ? (
            <button
              onClick={startRecording}
              disabled={disabled}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
                ${disabled
                  ? 'bg-surface text-text-muted cursor-not-allowed'
                  : 'bg-accent hover:bg-accent/80 text-white'
                }
              `}
              aria-label="Start recording"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                />
              </svg>
              Start Recording
            </button>
          ) : status === 'connecting' ? (
            <button
              disabled
              className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-surface text-text-muted"
            >
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Connecting...
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-crimson hover:bg-crimson/80 text-white transition-all"
              aria-label="Stop recording"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5.25 7.5A2.25 2.25 0 017.5 5.25h9a2.25 2.25 0 012.25 2.25v9a2.25 2.25 0 01-2.25 2.25h-9a2.25 2.25 0 01-2.25-2.25v-9z"
                />
              </svg>
              Stop Recording
            </button>
          )}

          {transcript.length > 0 && (
            <button
              onClick={() => {
                setTranscript([]);
                setCurrentText('');
              }}
              className="text-sm text-text-muted hover:text-text transition-colors"
            >
              Clear
            </button>
          )}
        </div>

        {/* Error message */}
        {error && (
          <div className="bg-crimson/10 border border-crimson/30 rounded p-3 text-sm text-crimson">
            {error}
          </div>
        )}

        {/* Real-time transcript display */}
        {(transcript.length > 0 || currentText) && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs text-text-muted uppercase tracking-wider">
                Transcript
              </div>
              {autoSubmitSuggestion && sendSuggestion && (
                <div className="text-xs text-acid-green">
                  Auto-submitting to debate
                </div>
              )}
            </div>
            <div className="bg-surface border border-border rounded p-3 max-h-48 overflow-y-auto">
              <p className="text-sm leading-relaxed">
                {getFullTranscript()}
                {currentText && (
                  <span className="text-text-muted animate-pulse">
                    {' '}{currentText}
                  </span>
                )}
              </p>
            </div>
            <div className="flex items-center justify-between">
              <div className="text-xs text-text-muted">
                {transcript.length} segment{transcript.length !== 1 ? 's' : ''} transcribed
              </div>
              {/* Manual submit button (when auto-submit is disabled) */}
              {!autoSubmitSuggestion && sendSuggestion && getFullTranscript().trim() && (
                <button
                  onClick={() => {
                    const fullText = getFullTranscript().trim();
                    if (fullText) {
                      sendSuggestion(fullText);
                      setTranscript([]);
                      setCurrentText('');
                    }
                  }}
                  className="text-xs px-3 py-1 bg-accent hover:bg-accent/80 text-white rounded transition-colors"
                >
                  Submit to Debate
                </button>
              )}
            </div>
          </div>
        )}

        {/* Instructions */}
        {status === 'idle' && transcript.length === 0 && (
          <div className="text-sm text-text-muted">
            <p>Click &quot;Start Recording&quot; to speak your argument.</p>
            <p className="mt-1 text-xs">Your speech will be transcribed in real-time and can be added to the debate.</p>
          </div>
        )}
      </div>
    </div>
  );
}
