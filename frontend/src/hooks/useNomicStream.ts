import { useCallback, useRef, useEffect, useState } from 'react';

interface AudienceMessage {
  type: 'vote' | 'suggestion';
  content: string;
  client_ts: number;
}

interface StreamEvent {
  type: string;
  data: any;
  timestamp: number;
  round: number;
  agent: string;
  loop_id?: string;
}

interface UseNomicStreamOptions {
  url?: string;
  onEvent?: (event: StreamEvent) => void;
  onError?: (error: Error) => void;
}

export function useNomicStream(options: UseNomicStreamOptions = {}) {
  const { url = 'ws://localhost:8765', onEvent, onError } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [activeLoops, setActiveLoops] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'loop_list') {
          setActiveLoops(data.data.loops);
        } else if (onEvent) {
          onEvent(data);
        }
      } catch (err) {
        onError?.(new Error('Failed to parse WebSocket message'));
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      onError?.(new Error('WebSocket error'));
    };
  }, [url, onEvent, onError]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const sendMessage = useCallback((message: AudienceMessage, loopId?: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const payload = {
      type: 'client_message',
      loop_id: loopId || '',
      payload: message,
    };

    wsRef.current.send(JSON.stringify(payload));
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    isConnected,
    activeLoops,
    sendMessage,
    connect,
    disconnect,
  };
}