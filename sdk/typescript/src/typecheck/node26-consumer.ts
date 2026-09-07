/// <reference types="node" />

import { createClient, streamDebate, type AragoraConfig } from '../index';
import type { ReadableStream as NodeReadableStream } from 'node:stream/web';

const config: AragoraConfig = {
  baseUrl: 'https://api.example.com',
  apiKey: 'test-key',
  headers: { 'X-Consumer': 'node26' },
};

const client = createClient(config);

const requestInit: RequestInit = {
  method: 'POST',
  headers: new Headers({ 'Content-Type': 'application/json' }),
  body: JSON.stringify({ task: 'node 26 consumer typecheck' }),
  signal: new AbortController().signal,
};

const request = new Request(new URL('/api/v1/debates', config.baseUrl), requestInit);
const nativeFetch: typeof globalThis.fetch = fetch;

function headersToRecord(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}

async function checkNodeFetchConsumer(): Promise<void> {
  const response: Response = await nativeFetch(request);
  await client.request<{ ok: boolean }>('GET', '/api/health', {
    headers: headersToRecord(response.headers),
  });
}

async function checkStreamingConsumer(
  stream: NodeReadableStream<Uint8Array>
): Promise<void> {
  const reader = stream.getReader();
  await reader.read();
}

async function checkSdkStreamConsumer(): Promise<void> {
  for await (const event of streamDebate(config, { debateId: 'debate-node-26' })) {
    const eventType: string = event.type;
    if (eventType === 'debate_end') {
      break;
    }
  }
}

void checkNodeFetchConsumer;
void checkStreamingConsumer;
void checkSdkStreamConsumer;
