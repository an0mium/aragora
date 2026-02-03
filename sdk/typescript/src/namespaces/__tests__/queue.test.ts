/**
 * Queue Namespace Tests
 *
 * Comprehensive tests for the queue namespace API including:
 * - Job listing and management
 * - Job enqueueing
 * - Worker status
 * - Dead letter queue (DLQ)
 * - Cleanup operations
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { QueueAPI } from '../queue';

interface MockClient {
  request: Mock;
  get: Mock;
  post: Mock;
  delete: Mock;
}

describe('QueueAPI Namespace', () => {
  let api: QueueAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
      get: vi.fn(),
      post: vi.fn(),
      delete: vi.fn(),
    };
    api = new QueueAPI(mockClient as any);
  });

  // ===========================================================================
  // Job Listing
  // ===========================================================================

  describe('Job Listing', () => {
    it('should list all jobs', async () => {
      const mockResponse = {
        jobs: [
          {
            job_id: 'job_1',
            type: 'debate_process',
            status: 'pending',
            created_at: '2024-01-20T10:00:00Z',
            attempts: 0,
          },
          {
            job_id: 'job_2',
            type: 'email_send',
            status: 'processing',
            created_at: '2024-01-20T10:01:00Z',
            attempts: 1,
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listJobs();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/jobs', {
        params: undefined,
      });
      expect(result.jobs).toHaveLength(2);
      expect(result.total).toBe(2);
    });

    it('should list jobs with filters', async () => {
      const mockResponse = {
        jobs: [
          { job_id: 'job_1', type: 'debate_process', status: 'failed' },
        ],
        total: 1,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.listJobs({
        status: 'failed',
        type: 'debate_process',
        limit: 50,
        offset: 10,
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/jobs', {
        params: { status: 'failed', type: 'debate_process', limit: 50, offset: 10 },
      });
    });
  });

  // ===========================================================================
  // Job Enqueueing
  // ===========================================================================

  describe('Job Enqueueing', () => {
    it('should enqueue a job', async () => {
      const mockResponse = { job_id: 'job_new_123' };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.enqueueJob({
        type: 'debate_process',
        payload: { debate_id: 'd_123', priority: 'high' },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/queue/jobs', {
        type: 'debate_process',
        payload: { debate_id: 'd_123', priority: 'high' },
      });
      expect(result.job_id).toBe('job_new_123');
    });

    it('should enqueue job with delay', async () => {
      const mockResponse = { job_id: 'job_delayed' };
      mockClient.post.mockResolvedValue(mockResponse);

      await api.enqueueJob({
        type: 'scheduled_report',
        delay_seconds: 3600,
        payload: { report_type: 'weekly' },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/queue/jobs', {
        type: 'scheduled_report',
        delay_seconds: 3600,
        payload: { report_type: 'weekly' },
      });
    });
  });

  // ===========================================================================
  // Individual Job Operations
  // ===========================================================================

  describe('Individual Job Operations', () => {
    it('should get job by ID', async () => {
      const mockJob = {
        job_id: 'job_123',
        type: 'debate_process',
        status: 'completed',
        payload: { debate_id: 'd_123' },
        created_at: '2024-01-20T10:00:00Z',
        updated_at: '2024-01-20T10:05:00Z',
        attempts: 1,
        error: null,
      };
      mockClient.get.mockResolvedValue(mockJob);

      const result = await api.getJob('job_123');

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/queue/jobs/job_123');
      expect(result.status).toBe('completed');
      expect(result.attempts).toBe(1);
    });

    it('should get failed job with error', async () => {
      const mockJob = {
        job_id: 'job_failed',
        type: 'email_send',
        status: 'failed',
        attempts: 3,
        error: 'SMTP connection timeout after 30s',
      };
      mockClient.get.mockResolvedValue(mockJob);

      const result = await api.getJob('job_failed');

      expect(result.status).toBe('failed');
      expect(result.error).toBe('SMTP connection timeout after 30s');
    });

    it('should delete job', async () => {
      const mockResponse = { deleted: true };
      mockClient.delete.mockResolvedValue(mockResponse);

      const result = await api.deleteJob('job_123');

      expect(mockClient.delete).toHaveBeenCalledWith('/api/v1/queue/jobs/job_123');
      expect(result.deleted).toBe(true);
    });

    it('should retry job', async () => {
      const mockResponse = { requeued: true };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.retryJob('job_failed');

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/queue/jobs/job_failed/retry');
      expect(result.requeued).toBe(true);
    });
  });

  // ===========================================================================
  // Queue Statistics
  // ===========================================================================

  describe('Queue Statistics', () => {
    it('should get queue stats', async () => {
      const mockStats = {
        pending: 150,
        processing: 25,
        completed: 10000,
        failed: 50,
        dead_letter: 10,
        queue_depth: 175,
      };
      mockClient.get.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/queue/stats');
      expect(result.pending).toBe(150);
      expect(result.processing).toBe(25);
      expect(result.dead_letter).toBe(10);
    });

    it('should show empty queue stats', async () => {
      const mockStats = {
        pending: 0,
        processing: 0,
        completed: 5000,
        failed: 0,
        dead_letter: 0,
        queue_depth: 0,
      };
      mockClient.get.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(result.queue_depth).toBe(0);
    });
  });

  // ===========================================================================
  // Workers
  // ===========================================================================

  describe('Workers', () => {
    it('should list workers', async () => {
      const mockResponse = {
        workers: [
          {
            worker_id: 'worker_1',
            status: 'idle',
            last_seen: '2024-01-20T10:05:00Z',
            current_job: null,
          },
          {
            worker_id: 'worker_2',
            status: 'busy',
            last_seen: '2024-01-20T10:05:01Z',
            current_job: 'job_123',
          },
          {
            worker_id: 'worker_3',
            status: 'idle',
            last_seen: '2024-01-20T10:05:00Z',
            current_job: null,
          },
        ],
      };
      mockClient.get.mockResolvedValue(mockResponse);

      const result = await api.listWorkers();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/queue/workers');
      expect(result.workers).toHaveLength(3);
      expect(result.workers[1].current_job).toBe('job_123');
    });
  });

  // ===========================================================================
  // Dead Letter Queue
  // ===========================================================================

  describe('Dead Letter Queue', () => {
    it('should list DLQ jobs', async () => {
      const mockResponse = {
        jobs: [
          {
            job_id: 'dlq_1',
            type: 'webhook_delivery',
            status: 'dead',
            attempts: 5,
            error: 'Max retries exceeded',
          },
          {
            job_id: 'dlq_2',
            type: 'email_send',
            status: 'dead',
            attempts: 3,
            error: 'Invalid recipient',
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listDLQ();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/dlq', {
        params: undefined,
      });
      expect(result.jobs).toHaveLength(2);
      expect(result.total).toBe(2);
    });

    it('should list DLQ with pagination', async () => {
      const mockResponse = { jobs: [], total: 50 };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.listDLQ({ limit: 10, offset: 20 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/dlq', {
        params: { limit: 10, offset: 20 },
      });
    });

    it('should requeue all DLQ jobs', async () => {
      const mockResponse = { requeued: 10, failed: 0 };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.requeueDLQ();

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/queue/dlq/requeue', undefined);
      expect(result.requeued).toBe(10);
    });

    it('should requeue specific DLQ jobs', async () => {
      const mockResponse = { requeued: 2, failed: 1 };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.requeueDLQ({ job_ids: ['dlq_1', 'dlq_2', 'dlq_3'] });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/queue/dlq/requeue', {
        job_ids: ['dlq_1', 'dlq_2', 'dlq_3'],
      });
      expect(result.requeued).toBe(2);
      expect(result.failed).toBe(1);
    });
  });

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  describe('Cleanup', () => {
    it('should cleanup old jobs', async () => {
      const mockResponse = { cleaned: 500 };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.cleanup();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/queue/cleanup', {
        params: undefined,
      });
      expect(result.cleaned).toBe(500);
    });

    it('should cleanup with custom max age', async () => {
      const mockResponse = { cleaned: 1000 };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.cleanup({ max_age_days: 7 });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/queue/cleanup', {
        params: { max_age_days: 7 },
      });
      expect(result.cleaned).toBe(1000);
    });
  });

  // ===========================================================================
  // Stale Jobs
  // ===========================================================================

  describe('Stale Jobs', () => {
    it('should list stale jobs', async () => {
      const mockResponse = {
        jobs: [
          {
            job_id: 'stale_1',
            type: 'debate_process',
            status: 'processing',
            updated_at: '2024-01-20T08:00:00Z',
          },
        ],
        total: 1,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listStale();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/stale', {
        params: undefined,
      });
      expect(result.jobs).toHaveLength(1);
    });

    it('should list stale jobs with custom threshold', async () => {
      const mockResponse = { jobs: [], total: 0 };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.listStale({ stale_after_seconds: 1800 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/queue/stale', {
        params: { stale_after_seconds: 1800 },
      });
    });
  });
});
