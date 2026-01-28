/**
 * Queue Namespace API
 *
 * Provides access to background job queues, worker status, and DLQ management.
 */

export interface QueueJob {
  job_id: string;
  type?: string;
  status?: string;
  payload?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  attempts?: number;
  error?: string | null;
}

export interface QueueStats {
  pending?: number;
  processing?: number;
  completed?: number;
  failed?: number;
  dead_letter?: number;
  queue_depth?: number;
}

export interface QueueWorker {
  worker_id: string;
  status: string;
  last_seen?: string;
  current_job?: string | null;
}

interface QueueClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown; json?: Record<string, unknown> }
  ): Promise<T>;
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
}

export class QueueAPI {
  constructor(private client: QueueClientInterface) {}

  async listJobs(params?: { status?: string; type?: string; limit?: number; offset?: number }): Promise<{ jobs: QueueJob[]; total?: number }> {
    return this.client.request('GET', '/api/v1/queue/jobs', { params: params as Record<string, unknown> });
  }

  async enqueueJob(body: Record<string, unknown>): Promise<{ job_id: string }> {
    return this.client.post('/api/v1/queue/jobs', body);
  }

  async getJob(jobId: string): Promise<QueueJob> {
    return this.client.get(`/api/v1/queue/jobs/${jobId}`);
  }

  async deleteJob(jobId: string): Promise<{ deleted?: boolean }> {
    return this.client.delete(`/api/v1/queue/jobs/${jobId}`);
  }

  async retryJob(jobId: string): Promise<{ requeued?: boolean }> {
    return this.client.post(`/api/v1/queue/jobs/${jobId}/retry`);
  }

  async getStats(): Promise<QueueStats> {
    return this.client.get('/api/v1/queue/stats');
  }

  async listWorkers(): Promise<{ workers: QueueWorker[] }> {
    return this.client.get('/api/v1/queue/workers');
  }

  async listDLQ(params?: { limit?: number; offset?: number }): Promise<{ jobs: QueueJob[]; total?: number }> {
    return this.client.request('GET', '/api/v1/queue/dlq', { params: params as Record<string, unknown> });
  }

  async requeueDLQ(body?: { job_ids?: string[] }): Promise<{ requeued?: number; failed?: number }> {
    return this.client.post('/api/v1/queue/dlq/requeue', body);
  }

  async cleanup(params?: { max_age_days?: number }): Promise<{ cleaned?: number }> {
    return this.client.request('POST', '/api/v1/queue/cleanup', { params: params as Record<string, unknown> });
  }

  async listStale(params?: { stale_after_seconds?: number }): Promise<{ jobs: QueueJob[]; total?: number }> {
    return this.client.request('GET', '/api/v1/queue/stale', { params: params as Record<string, unknown> });
  }
}
