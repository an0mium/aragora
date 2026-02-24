import { test, expect } from './fixtures';

test.describe('Oracle Stream Recovery', () => {
  test('surfaces first-token stall and allows reset recovery', async ({ page, aragoraPage }) => {
    await page.addInitScript(() => {
      const nativeSetTimeout = window.setTimeout.bind(window);
      window.setTimeout = ((handler: TimerHandler, timeout?: number, ...args: unknown[]) => {
        const requested = typeof timeout === 'number' ? timeout : 0;
        const accelerated = requested >= 10000 ? 40 : requested;
        return nativeSetTimeout(handler, accelerated, ...args);
      }) as typeof window.setTimeout;

      class MockOracleWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        onopen: ((event: Event) => void) | null = null;
        onclose: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
        readyState = MockOracleWebSocket.CONNECTING;
        binaryType = 'arraybuffer';
        readonly url: string;

        constructor(url: string) {
          this.url = url;
          nativeSetTimeout(() => {
            this.readyState = MockOracleWebSocket.OPEN;
            this.onopen?.(new Event('open'));
            this.emit({ type: 'connected' });
          }, 0);
        }

        send(payload: string): void {
          let data: { type?: string } | null = null;
          try {
            data = JSON.parse(payload) as { type?: string };
          } catch {
            data = null;
          }
          if (!data) return;
          if (data.type === 'ask') {
            // Intentionally never emit token to force first-token stall.
            this.emit({ type: 'reflex_start' });
          } else if (data.type === 'stop') {
            this.emit({ type: 'pong' });
          }
        }

        close(): void {
          if (this.readyState === MockOracleWebSocket.CLOSED) return;
          this.readyState = MockOracleWebSocket.CLOSED;
          this.onclose?.(new Event('close'));
        }

        emit(payload: unknown): void {
          this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
        }
      }

      Object.defineProperty(window, 'WebSocket', {
        configurable: true,
        writable: true,
        value: MockOracleWebSocket,
      });
    });

    await page.goto('/oracle', { waitUntil: 'domcontentloaded' });
    await aragoraPage.dismissAllOverlays();

    await page.locator('textarea').first().fill('Show me where this strategy fails.');
    await page.getByRole('button', { name: 'SPEAK' }).click();

    const stallBadge = page.getByText('Stall: no first token');
    const resetButton = page.getByRole('button', { name: 'Reset Stream' });

    await expect(stallBadge).toBeVisible({ timeout: 5000 });
    await expect(resetButton).toBeVisible({ timeout: 5000 });

    await resetButton.click();

    await expect(stallBadge).toBeHidden({ timeout: 5000 });
    await expect(resetButton).toBeHidden({ timeout: 5000 });
  });
});
