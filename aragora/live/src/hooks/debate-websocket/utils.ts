/**
 * Utility functions for Debate WebSocket hook
 */

/**
 * Create a composite key for streaming messages.
 * Combines agent name with task ID for distinguishing concurrent outputs.
 */
export function makeStreamingKey(agent: string, taskId: string): string {
  return taskId ? `${agent}:${taskId}` : agent;
}

/**
 * Calculate exponential backoff delay for reconnection attempts.
 * Uses exponential backoff with jitter to prevent thundering herd.
 */
export function calculateReconnectDelay(attempt: number, maxDelay: number): number {
  const baseDelay = Math.min(1000 * Math.pow(2, attempt), maxDelay);
  // Add 0-20% jitter to prevent synchronized reconnection attempts
  const jitter = baseDelay * Math.random() * 0.2;
  return baseDelay + jitter;
}

/**
 * Check if an error is a network-related error that should trigger reconnection.
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes('network') ||
      message.includes('connection') ||
      message.includes('timeout') ||
      message.includes('socket')
    );
  }
  return false;
}
