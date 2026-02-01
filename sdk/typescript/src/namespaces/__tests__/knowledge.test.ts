/**
 * Knowledge Namespace Tests
 *
 * Comprehensive tests for the Knowledge namespace API including:
 * - Core knowledge operations (search, add, update, delete)
 * - Natural language query
 * - Fact CRUD operations
 * - Knowledge Mound operations (nodes, relationships, queries)
 * - Validation and refresh
 * - Sources and embeddings
 * - Graph traversal
 * - Visibility and access control
 * - Sharing operations
 * - Federation
 * - Global knowledge
 * - Deduplication
 * - Pruning
 * - Culture
 * - Dashboard and metrics
 * - Contradictions
 * - Governance
 * - Analytics
 * - Extraction
 * - Confidence decay
 * - Export
 * - Auto-curation
 * - Synchronization
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createClient, AragoraClient } from '../../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Knowledge Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-api-key',
      retryEnabled: false,
    });
  });

  // ===========================================================================
  // Core Knowledge Operations
  // ===========================================================================

  describe('Core Knowledge Operations', () => {
    it('should search knowledge entries', async () => {
      const mockResults = {
        results: [
          { id: 'k1', content: 'Machine learning basics', score: 0.95, source: 'docs' },
          { id: 'k2', content: 'Deep learning overview', score: 0.88, source: 'research' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResults)),
      });

      const result = await client.knowledge.search('machine learning');

      expect(result.results).toHaveLength(2);
      expect(result.results[0].score).toBe(0.95);
    });

    it('should search knowledge with options', async () => {
      const mockResults = { results: [{ id: 'k1', content: 'Result', score: 0.9 }] };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResults)),
      });

      const result = await client.knowledge.search('AI', {
        type: 'fact',
        tags: ['ml'],
        min_confidence: 0.8,
        limit: 10,
      });

      expect(result.results).toHaveLength(1);
    });

    it('should add a knowledge entry', async () => {
      const mockResponse = {
        id: 'entry-123',
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.add({
        content: 'Neural networks are computational models',
        source: 'research-paper',
        tags: ['AI', 'ML'],
        importance: 0.95,
      });

      expect(result.id).toBe('entry-123');
    });

    it('should get a knowledge entry by ID', async () => {
      const mockEntry = {
        id: 'entry-123',
        content: 'Test knowledge entry',
        source: 'manual',
        confidence: 0.9,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockEntry)),
      });

      const result = await client.knowledge.get('entry-123');

      expect(result.id).toBe('entry-123');
      expect(result.content).toBe('Test knowledge entry');
    });

    it('should update a knowledge entry', async () => {
      const mockUpdated = {
        id: 'entry-123',
        content: 'Updated content',
        importance: 0.98,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockUpdated)),
      });

      const result = await client.knowledge.update('entry-123', {
        content: 'Updated content',
        importance: 0.98,
      });

      expect(result.content).toBe('Updated content');
    });

    it('should delete a knowledge entry', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ deleted: true })),
      });

      const result = await client.knowledge.delete('entry-123');

      expect(result.deleted).toBe(true);
    });

    it('should get knowledge stats', async () => {
      const mockStats = {
        total_entries: 5000,
        by_visibility: { private: 2000, team: 2500, global: 500 },
        by_source: { manual: 1000, debate: 3000, import: 1000 },
        storage_bytes: 10485760,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStats)),
      });

      const result = await client.knowledge.stats();

      expect(result.total_entries).toBe(5000);
      expect(result.by_visibility.private).toBe(2000);
    });

    it('should bulk import knowledge entries', async () => {
      const mockResponse = {
        imported: 10,
        failed: 1,
        errors: [{ index: 5, error: 'Invalid format' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.bulkImport([
        { content: 'Fact 1' },
        { content: 'Fact 2' },
      ]);

      expect(result.imported).toBe(10);
      expect(result.failed).toBe(1);
    });

    it('should bulk export knowledge entries', async () => {
      const mockExport = {
        entries: [{ id: 'k1', content: 'Entry 1' }],
        total: 100,
        format: 'json',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.knowledge.bulkExport({ format: 'json' });

      expect(result.format).toBe('json');
      expect(result.total).toBe(100);
    });
  });

  // ===========================================================================
  // Natural Language Query
  // ===========================================================================

  describe('Natural Language Query', () => {
    it('should run a natural language query', async () => {
      const mockResponse = {
        answer: 'Machine learning is a subset of AI...',
        sources: [{ id: 'k1', title: 'ML Basics' }],
        confidence: 0.92,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.query('What is machine learning?');

      expect(result.answer).toContain('Machine learning');
      expect(result.confidence).toBe(0.92);
    });
  });

  // ===========================================================================
  // Fact CRUD Operations
  // ===========================================================================

  describe('Fact CRUD Operations', () => {
    it('should list facts with filtering', async () => {
      const mockFacts = {
        facts: [
          { id: 'f1', statement: 'Fact 1', confidence: 0.9 },
          { id: 'f2', statement: 'Fact 2', confidence: 0.85 },
        ],
        total: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockFacts)),
      });

      const result = await client.knowledge.listFacts({
        min_confidence: 0.8,
        limit: 50,
      });

      expect(result.facts).toHaveLength(2);
      expect(result.total).toBe(100);
    });

    it('should get a specific fact', async () => {
      const mockFact = {
        fact: { id: 'fact-123', statement: 'Test fact', confidence: 0.95 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockFact)),
      });

      const result = await client.knowledge.getFact('fact-123');

      expect(result.fact).toBeDefined();
    });

    it('should create a new fact', async () => {
      const mockResponse = {
        id: 'fact-456',
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.createFact({
        statement: 'TypeScript is a typed superset of JavaScript',
        confidence: 0.99,
        topics: ['programming', 'typescript'],
      });

      expect(result.id).toBe('fact-456');
    });

    it('should update an existing fact', async () => {
      const mockUpdated = {
        fact: { id: 'fact-123', confidence: 0.98 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockUpdated)),
      });

      const result = await client.knowledge.updateFact('fact-123', {
        confidence: 0.98,
        validation_status: 'verified',
      });

      expect(result.fact).toBeDefined();
    });

    it('should delete a fact', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ deleted: true })),
      });

      const result = await client.knowledge.deleteFact('fact-123');

      expect(result.deleted).toBe(true);
    });

    it('should verify a fact', async () => {
      const mockVerification = {
        verified: true,
        confidence: 0.95,
        evidence: [{ id: 'e1', content: 'Supporting evidence' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVerification)),
      });

      const result = await client.knowledge.verifyFact('fact-123');

      expect(result.verified).toBe(true);
      expect(result.confidence).toBe(0.95);
    });

    it('should get fact contradictions', async () => {
      const mockContradictions = {
        contradictions: [
          { id: 'c1', conflicting_fact_id: 'fact-456', reason: 'Opposing claim' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockContradictions)),
      });

      const result = await client.knowledge.getFactContradictions('fact-123');

      expect(result.contradictions).toHaveLength(1);
    });

    it('should list fact relations', async () => {
      const mockRelations = {
        relations: [
          { id: 'r1', target_fact_id: 'fact-456', relation_type: 'supports' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRelations)),
      });

      const result = await client.knowledge.listFactRelations('fact-123');

      expect(result.relations).toHaveLength(1);
    });

    it('should add a fact relation', async () => {
      const mockResponse = { relation_id: 'rel-789' };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.addFactRelation('fact-123', 'fact-456', 'supports');

      expect(result.relation_id).toBe('rel-789');
    });

    it('should add relation between facts with options', async () => {
      const mockResponse = { relation_id: 'rel-999' };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.addRelationBetweenFacts('fact-1', 'fact-2', 'elaborates', {
        confidence: 0.85,
        metadata: { source: 'debate' },
      });

      expect(result.relation_id).toBe('rel-999');
    });
  });

  // ===========================================================================
  // Knowledge Mound Operations
  // ===========================================================================

  describe('Knowledge Mound Operations', () => {
    it('should query the Knowledge Mound', async () => {
      const mockResult = {
        nodes: [
          { id: 'n1', content: 'AI basics', node_type: 'concept', confidence: 0.9 },
        ],
        relationships: [
          { id: 'r1', source_id: 'n1', target_id: 'n2', relationship_type: 'related_to' },
        ],
        total: 1,
        query_time_ms: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.knowledge.queryMound('AI', { depth: 2 });

      expect(result.nodes).toHaveLength(1);
      expect(result.query_time_ms).toBe(50);
    });

    it('should list Knowledge Mound nodes', async () => {
      const mockNodes = {
        nodes: [
          { id: 'n1', content: 'Node 1', node_type: 'fact', confidence: 0.95 },
          { id: 'n2', content: 'Node 2', node_type: 'concept', confidence: 0.88 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockNodes)),
      });

      const result = await client.knowledge.listNodes({ limit: 50, type: 'fact' });

      expect(result.nodes).toHaveLength(2);
    });

    it('should create a Knowledge Mound node', async () => {
      const mockResponse = {
        id: 'node-123',
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.createNode({
        content: 'New knowledge node',
        node_type: 'fact',
        confidence: 0.9,
        tags: ['test'],
        visibility: 'team',
      });

      expect(result.id).toBe('node-123');
    });

    it('should get a specific node', async () => {
      const mockNode = {
        node: { id: 'node-123', content: 'Test node', node_type: 'fact' },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockNode)),
      });

      const result = await client.knowledge.getNode('node-123');

      expect(result.node).toBeDefined();
    });

    it('should get node relationships', async () => {
      const mockRelationships = {
        relationships: [
          { id: 'r1', source_id: 'node-123', target_id: 'node-456', relationship_type: 'supports' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRelationships)),
      });

      const result = await client.knowledge.getNodeRelationships('node-123');

      expect(result.relationships).toHaveLength(1);
    });

    it('should list Knowledge Mound relationships', async () => {
      const mockRelationships = {
        relationships: [
          { id: 'r1', source_id: 'n1', target_id: 'n2', relationship_type: 'supports', strength: 0.9 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRelationships)),
      });

      const result = await client.knowledge.listRelationships({ limit: 100 });

      expect(result.relationships).toHaveLength(1);
    });

    it('should create a Knowledge Mound relationship', async () => {
      const mockResponse = {
        id: 'rel-123',
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.knowledge.createRelationship({
        source_id: 'node-1',
        target_id: 'node-2',
        relationship_type: 'supports',
        strength: 0.85,
        confidence: 0.9,
      });

      expect(result.id).toBe('rel-123');
    });

    it('should get Knowledge Mound stats', async () => {
      const mockStats = {
        total_nodes: 1000,
        total_relationships: 2500,
        nodes_by_type: { fact: 500, concept: 300, claim: 200 },
        relationships_by_type: { supports: 1000, contradicts: 200 },
        avg_confidence: 0.85,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStats)),
      });

      const result = await client.knowledge.moundStats();

      expect(result.total_nodes).toBe(1000);
      expect(result.avg_confidence).toBe(0.85);
    });

    it('should get stale knowledge', async () => {
      const mockStale = {
        items: [{ id: 'old-1', last_validated: '2023-01-01' }],
        total: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStale)),
      });

      const result = await client.knowledge.getStale({ max_age_days: 30 });

      expect(result.total).toBe(50);
    });

    it('should revalidate knowledge', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ updated: true })),
      });

      const result = await client.knowledge.revalidate('node-123', {
        valid: true,
        new_confidence: 0.95,
        notes: 'Verified with new data',
      });

      expect(result.updated).toBe(true);
    });

    it('should get knowledge lineage', async () => {
      const mockLineage = {
        nodes: [{ id: 'n1' }, { id: 'n2' }],
        relationships: [{ id: 'r1', source_id: 'n1', target_id: 'n2' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockLineage)),
      });

      const result = await client.knowledge.getLineage('node-123', {
        direction: 'ancestors',
        max_depth: 3,
      });

      expect(result.nodes).toHaveLength(2);
    });

    it('should get related knowledge', async () => {
      const mockRelated = {
        nodes: [{ id: 'related-1' }],
        relationships: [{ id: 'r1', relationship_type: 'supports' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRelated)),
      });

      const result = await client.knowledge.getRelated('node-123', {
        relationship_types: ['supports'],
        limit: 10,
      });

      expect(result.nodes).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Validation and Refresh
  // ===========================================================================

  describe('Validation and Refresh', () => {
    it('should validate knowledge entries', async () => {
      const mockValidation = {
        validated: 100,
        invalid: 5,
        results: [{ node_id: 'n1', status: 'valid' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockValidation)),
      });

      const result = await client.knowledge.validate({ workspace_id: 'ws-123' });

      expect(result.validated).toBe(100);
      expect(result.invalid).toBe(5);
    });

    it('should refresh knowledge from sources', async () => {
      const mockRefresh = {
        refreshed: 50,
        updated: 20,
        errors: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRefresh)),
      });

      const result = await client.knowledge.refresh({ source_types: ['api', 'document'] });

      expect(result.refreshed).toBe(50);
      expect(result.updated).toBe(20);
    });
  });

  // ===========================================================================
  // Sources and Embeddings
  // ===========================================================================

  describe('Sources and Embeddings', () => {
    it('should get sources for a knowledge entry', async () => {
      const mockSources = {
        sources: [
          { id: 's1', type: 'document', url: 'https://example.com/doc' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSources)),
      });

      const result = await client.knowledge.getSources('entry-123');

      expect(result.sources).toHaveLength(1);
    });

    it('should get embeddings for a knowledge entry', async () => {
      const mockEmbeddings = {
        embeddings: [0.1, 0.2, 0.3, 0.4],
        model: 'text-embedding-3-small',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockEmbeddings)),
      });

      const result = await client.knowledge.getEmbeddings('entry-123');

      expect(result.embeddings).toHaveLength(4);
      expect(result.model).toBe('text-embedding-3-small');
    });

    it('should generate embeddings for text', async () => {
      const mockGenerated = {
        embeddings: [0.5, 0.6, 0.7],
        model: 'text-embedding-3-large',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockGenerated)),
      });

      const result = await client.knowledge.generateEmbeddings('Sample text', {
        model: 'text-embedding-3-large',
      });

      expect(result.embeddings).toHaveLength(3);
    });
  });

  // ===========================================================================
  // Graph Traversal
  // ===========================================================================

  describe('Graph Traversal', () => {
    it('should traverse the knowledge graph', async () => {
      const mockGraph = {
        nodes: [{ id: 'n1' }, { id: 'n2' }],
        relationships: [{ id: 'r1', source_id: 'n1', target_id: 'n2' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockGraph)),
      });

      const result = await client.knowledge.traverseGraph('node-123', {
        depth: 3,
        direction: 'outgoing',
      });

      expect(result.nodes).toHaveLength(2);
    });

    it('should get node lineage', async () => {
      const mockLineage = {
        nodes: [{ id: 'ancestor-1' }],
        path: [{ from: 'node-123', to: 'ancestor-1' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockLineage)),
      });

      const result = await client.knowledge.getNodeLineage('node-123', { max_depth: 5 });

      expect(result.nodes).toHaveLength(1);
    });

    it('should get node neighbors', async () => {
      const mockNeighbors = {
        nodes: [{ id: 'neighbor-1' }],
        relationships: [{ id: 'r1' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockNeighbors)),
      });

      const result = await client.knowledge.getNodeNeighbors('node-123', { limit: 10 });

      expect(result.nodes).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Visibility and Access Control
  // ===========================================================================

  describe('Visibility and Access Control', () => {
    it('should get node visibility', async () => {
      const mockVisibility = {
        visibility: 'team',
        access_grants: [{ grantee_id: 'user-1', permission: 'read' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVisibility)),
      });

      const result = await client.knowledge.getVisibility('node-123');

      expect(result.visibility).toBe('team');
    });

    it('should set node visibility', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ updated: true })),
      });

      const result = await client.knowledge.setVisibility('node-123', 'public');

      expect(result.updated).toBe(true);
    });

    it('should list access grants', async () => {
      const mockGrants = {
        grants: [{ id: 'g1', grantee_id: 'user-1', permission: 'write' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockGrants)),
      });

      const result = await client.knowledge.listAccessGrants('node-123');

      expect(result.grants).toHaveLength(1);
    });

    it('should grant access to a node', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ grant_id: 'grant-456' })),
      });

      const result = await client.knowledge.grantAccess('node-123', {
        grantee_id: 'user-789',
        grantee_type: 'user',
        permission: 'read',
      });

      expect(result.grant_id).toBe('grant-456');
    });

    it('should revoke access from a node', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ revoked: true })),
      });

      const result = await client.knowledge.revokeAccess('node-123', 'grant-456');

      expect(result.revoked).toBe(true);
    });
  });

  // ===========================================================================
  // Sharing Operations
  // ===========================================================================

  describe('Sharing Operations', () => {
    it('should share a knowledge item', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ share_id: 'share-123' })),
      });

      const result = await client.knowledge.share('item-456', {
        target_id: 'workspace-789',
        target_type: 'workspace',
        permission: 'read',
      });

      expect(result.share_id).toBe('share-123');
    });

    it('should get items shared with me', async () => {
      const mockShared = {
        items: [{ id: 'i1', content: 'Shared item' }],
        total: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockShared)),
      });

      const result = await client.knowledge.getSharedWithMe({ limit: 50 });

      expect(result.items).toHaveLength(1);
      expect(result.total).toBe(10);
    });

    it('should get my shares', async () => {
      const mockShares = {
        shares: [{ id: 's1', item_id: 'item-1', target_id: 'ws-1' }],
        total: 5,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockShares)),
      });

      const result = await client.knowledge.getMyShares();

      expect(result.shares).toHaveLength(1);
    });

    it('should revoke a share', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ revoked: true })),
      });

      const result = await client.knowledge.revokeShare('share-123');

      expect(result.revoked).toBe(true);
    });

    it('should update a share', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ updated: true })),
      });

      const result = await client.knowledge.updateShare('share-123', {
        permission: 'write',
        expires_at: '2025-01-01T00:00:00Z',
      });

      expect(result.updated).toBe(true);
    });
  });

  // ===========================================================================
  // Federation
  // ===========================================================================

  describe('Federation', () => {
    it('should register a federated region', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ region_id: 'region-123' })),
      });

      const result = await client.knowledge.registerRegion({
        name: 'EU Region',
        endpoint: 'https://eu.aragora.ai',
        api_key: 'eu-api-key',
      });

      expect(result.region_id).toBe('region-123');
    });

    it('should list federated regions', async () => {
      const mockRegions = {
        regions: [
          { id: 'r1', name: 'US Region', endpoint: 'https://us.aragora.ai' },
          { id: 'r2', name: 'EU Region', endpoint: 'https://eu.aragora.ai' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRegions)),
      });

      const result = await client.knowledge.listRegions();

      expect(result.regions).toHaveLength(2);
    });

    it('should unregister a region', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ unregistered: true })),
      });

      const result = await client.knowledge.unregisterRegion('region-123');

      expect(result.unregistered).toBe(true);
    });

    it('should push sync to a region', async () => {
      const mockSync = { synced: 100, failed: 2 };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSync)),
      });

      const result = await client.knowledge.syncPush('region-123', { scope: 'workspace' });

      expect(result.synced).toBe(100);
    });

    it('should pull sync from a region', async () => {
      const mockPull = { received: 50, merged: 45 };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPull)),
      });

      const result = await client.knowledge.syncPull('region-123', { limit: 100 });

      expect(result.received).toBe(50);
      expect(result.merged).toBe(45);
    });

    it('should sync with all regions', async () => {
      const mockResults = {
        results: [
          { region_id: 'r1', synced: 50 },
          { region_id: 'r2', synced: 30 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResults)),
      });

      const result = await client.knowledge.syncAll({ direction: 'both' });

      expect(result.results).toHaveLength(2);
    });

    it('should get federation status', async () => {
      const mockStatus = {
        regions: [{ id: 'r1', status: 'healthy' }],
        last_sync: '2024-01-01T00:00:00Z',
        health: 'healthy',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.knowledge.getFederationStatus();

      expect(result.health).toBe('healthy');
    });
  });

  // ===========================================================================
  // Global Knowledge
  // ===========================================================================

  describe('Global Knowledge', () => {
    it('should store a global fact', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'global-123' })),
      });

      const result = await client.knowledge.storeGlobalFact({
        content: 'Universal fact',
        source: 'verified-source',
        confidence: 0.99,
        tags: ['universal'],
      });

      expect(result.id).toBe('global-123');
    });

    it('should query global knowledge', async () => {
      const mockResults = {
        results: [{ id: 'g1', content: 'Global fact' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResults)),
      });

      const result = await client.knowledge.queryGlobal('universal truth', { limit: 10 });

      expect(result.results).toHaveLength(1);
    });

    it('should promote to global level', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ promoted: true, global_id: 'global-456' })),
      });

      const result = await client.knowledge.promoteToGlobal('node-123');

      expect(result.promoted).toBe(true);
      expect(result.global_id).toBe('global-456');
    });

    it('should get system facts', async () => {
      const mockFacts = {
        facts: [{ id: 'sf1', content: 'System fact' }],
        total: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockFacts)),
      });

      const result = await client.knowledge.getSystemFacts({ limit: 50 });

      expect(result.facts).toHaveLength(1);
      expect(result.total).toBe(100);
    });
  });

  // ===========================================================================
  // Deduplication
  // ===========================================================================

  describe('Deduplication', () => {
    it('should get duplicate clusters', async () => {
      const mockClusters = {
        clusters: [
          { cluster_id: 'c1', nodes: ['n1', 'n2'], similarity: 0.95 },
        ],
        total: 5,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockClusters)),
      });

      const result = await client.knowledge.getDuplicateClusters({ threshold: 0.9 });

      expect(result.clusters).toHaveLength(1);
    });

    it('should get dedup report', async () => {
      const mockReport = {
        total_nodes: 1000,
        duplicate_clusters: 50,
        potential_savings: 200,
        recommendations: [{ action: 'merge', cluster_id: 'c1' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.knowledge.getDedupReport();

      expect(result.duplicate_clusters).toBe(50);
    });

    it('should merge duplicate cluster', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ merged_id: 'merged-123', removed_count: 3 })),
      });

      const result = await client.knowledge.mergeDuplicateCluster('cluster-456', {
        strategy: 'highest_confidence',
      });

      expect(result.merged_id).toBe('merged-123');
      expect(result.removed_count).toBe(3);
    });

    it('should auto-merge exact duplicates', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ merged: 10, clusters_processed: 15 })),
      });

      const result = await client.knowledge.autoMergeExactDuplicates({ dry_run: false });

      expect(result.merged).toBe(10);
    });
  });

  // ===========================================================================
  // Pruning
  // ===========================================================================

  describe('Pruning', () => {
    it('should get prunable items', async () => {
      const mockItems = {
        items: [{ id: 'old-1', reason: 'stale' }],
        total: 25,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockItems)),
      });

      const result = await client.knowledge.getPrunableItems({ max_age_days: 90 });

      expect(result.items).toHaveLength(1);
      expect(result.total).toBe(25);
    });

    it('should execute prune', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ pruned: 10, archived: 8, deleted: 2 })),
      });

      const result = await client.knowledge.executePrune(['node-1', 'node-2'], { action: 'archive' });

      expect(result.pruned).toBe(10);
      expect(result.archived).toBe(8);
    });

    it('should run auto-prune', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ pruned: 50, archived: 45 })),
      });

      const result = await client.knowledge.autoPrune({ policy: 'conservative' });

      expect(result.pruned).toBe(50);
    });

    it('should get prune history', async () => {
      const mockHistory = {
        events: [{ id: 'e1', timestamp: '2024-01-01', nodes_pruned: 10 }],
        total: 5,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHistory)),
      });

      const result = await client.knowledge.getPruneHistory();

      expect(result.events).toHaveLength(1);
    });

    it('should restore pruned item', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ restored: true })),
      });

      const result = await client.knowledge.restorePrunedItem('node-123');

      expect(result.restored).toBe(true);
    });
  });

  // ===========================================================================
  // Culture
  // ===========================================================================

  describe('Culture', () => {
    it('should get organization culture', async () => {
      const mockCulture = {
        principles: [{ id: 'p1', content: 'Quality first' }],
        values: [{ id: 'v1', content: 'Transparency' }],
        policies: [{ id: 'pol1', content: 'Data privacy' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCulture)),
      });

      const result = await client.knowledge.getCulture();

      expect(result.principles).toHaveLength(1);
      expect(result.values).toHaveLength(1);
    });

    it('should add a culture document', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'doc-123' })),
      });

      const result = await client.knowledge.addCultureDocument({
        type: 'principle',
        content: 'Customer satisfaction is paramount',
        source: 'company-handbook',
      });

      expect(result.id).toBe('doc-123');
    });

    it('should promote to culture', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ promoted: true })),
      });

      const result = await client.knowledge.promoteToCulture('node-123', { type: 'value' });

      expect(result.promoted).toBe(true);
    });
  });

  // ===========================================================================
  // Dashboard and Metrics
  // ===========================================================================

  describe('Dashboard and Metrics', () => {
    it('should get dashboard health', async () => {
      const mockHealth = {
        status: 'healthy',
        score: 95,
        recommendations: [{ priority: 'low', message: 'Consider deduplication' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHealth)),
      });

      const result = await client.knowledge.getDashboardHealth();

      expect(result.status).toBe('healthy');
      expect(result.score).toBe(95);
    });

    it('should get dashboard metrics', async () => {
      const mockMetrics = {
        queries: { total: 10000, avg_latency_ms: 50 },
        storage: { used_bytes: 1048576 },
        performance: { cache_hit_rate: 0.85 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMetrics)),
      });

      const result = await client.knowledge.getDashboardMetrics();

      expect((result.queries as { total: number }).total).toBe(10000);
    });

    it('should reset dashboard metrics', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ reset: true })),
      });

      const result = await client.knowledge.resetDashboardMetrics();

      expect(result.reset).toBe(true);
    });

    it('should get dashboard adapters', async () => {
      const mockAdapters = {
        adapters: [
          { name: 'postgres', status: 'healthy' },
          { name: 'redis', status: 'healthy' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAdapters)),
      });

      const result = await client.knowledge.getDashboardAdapters();

      expect(result.adapters).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Contradictions
  // ===========================================================================

  describe('Contradictions', () => {
    it('should detect contradictions', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ detected: 5, scan_id: 'scan-123' })),
      });

      const result = await client.knowledge.detectContradictions({ scope: 'workspace' });

      expect(result.detected).toBe(5);
      expect(result.scan_id).toBe('scan-123');
    });

    it('should list contradictions', async () => {
      const mockContradictions = {
        contradictions: [
          { id: 'c1', node_a_id: 'n1', node_b_id: 'n2', status: 'unresolved' },
        ],
        total: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockContradictions)),
      });

      const result = await client.knowledge.listContradictions({ status: 'unresolved' });

      expect(result.contradictions).toHaveLength(1);
      expect(result.total).toBe(10);
    });

    it('should resolve a contradiction', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ resolved: true })),
      });

      const result = await client.knowledge.resolveContradiction('contradiction-123', {
        strategy: 'keep_first',
        notes: 'First claim has better evidence',
      });

      expect(result.resolved).toBe(true);
    });

    it('should get contradiction stats', async () => {
      const mockStats = {
        total: 50,
        unresolved: 10,
        resolved: 40,
        by_type: { factual: 30, temporal: 20 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStats)),
      });

      const result = await client.knowledge.getContradictionStats();

      expect(result.total).toBe(50);
      expect(result.unresolved).toBe(10);
    });
  });

  // ===========================================================================
  // Governance
  // ===========================================================================

  describe('Governance', () => {
    it('should create a governance role', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ role_id: 'role-123' })),
      });

      const result = await client.knowledge.createGovernanceRole({
        name: 'Knowledge Curator',
        permissions: ['read', 'write', 'validate'],
        description: 'Can curate and validate knowledge',
      });

      expect(result.role_id).toBe('role-123');
    });

    it('should assign a governance role', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ assigned: true })),
      });

      const result = await client.knowledge.assignGovernanceRole('user-123', 'role-456');

      expect(result.assigned).toBe(true);
    });

    it('should revoke a governance role', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ revoked: true })),
      });

      const result = await client.knowledge.revokeGovernanceRole('user-123', 'role-456');

      expect(result.revoked).toBe(true);
    });

    it('should get user governance permissions', async () => {
      const mockPermissions = {
        permissions: ['read', 'write', 'validate'],
        roles: [{ id: 'r1', name: 'Curator' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPermissions)),
      });

      const result = await client.knowledge.getUserGovernancePermissions('user-123');

      expect(result.permissions).toContain('read');
      expect(result.roles).toHaveLength(1);
    });

    it('should check governance permission', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ allowed: true })),
      });

      const result = await client.knowledge.checkGovernancePermission('user-123', 'validate');

      expect(result.allowed).toBe(true);
    });

    it('should query governance audit', async () => {
      const mockAudit = {
        events: [{ id: 'e1', action: 'create', timestamp: '2024-01-01' }],
        total: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAudit)),
      });

      const result = await client.knowledge.queryGovernanceAudit({ action: 'create' });

      expect(result.events).toHaveLength(1);
    });

    it('should get user governance activity', async () => {
      const mockActivity = {
        activities: [{ id: 'a1', action: 'validate_node' }],
        total: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockActivity)),
      });

      const result = await client.knowledge.getUserGovernanceActivity('user-123');

      expect(result.activities).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Analytics
  // ===========================================================================

  describe('Analytics', () => {
    it('should analyze coverage', async () => {
      const mockCoverage = {
        coverage: [{ topic: 'AI', coverage_percent: 85 }],
        gaps: [{ topic: 'Quantum Computing', priority: 'high' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCoverage)),
      });

      const result = await client.knowledge.analyzeCoverage({ topics: ['AI', 'ML'] });

      expect(result.coverage).toHaveLength(1);
      expect(result.gaps).toHaveLength(1);
    });

    it('should analyze usage', async () => {
      const mockUsage = {
        patterns: [{ pattern: 'frequent_queries', count: 500 }],
        trends: [{ direction: 'up', metric: 'queries' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockUsage)),
      });

      const result = await client.knowledge.analyzeUsage({ period: 'week' });

      expect(result.patterns).toHaveLength(1);
    });

    it('should record usage event', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ recorded: true })),
      });

      const result = await client.knowledge.recordUsageEvent({
        node_id: 'node-123',
        event_type: 'view',
        metadata: { user_id: 'user-1' },
      });

      expect(result.recorded).toBe(true);
    });

    it('should capture quality snapshot', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          snapshot_id: 'snap-123',
          metrics: { avg_confidence: 0.88 },
        })),
      });

      const result = await client.knowledge.captureQualitySnapshot();

      expect(result.snapshot_id).toBe('snap-123');
    });

    it('should get quality trend', async () => {
      const mockTrend = {
        trend: [
          { date: '2024-01-01', avg_confidence: 0.85 },
          { date: '2024-01-02', avg_confidence: 0.87 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTrend)),
      });

      const result = await client.knowledge.getQualityTrend({ period: 'week' });

      expect(result.trend).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Extraction
  // ===========================================================================

  describe('Extraction', () => {
    it('should extract from debate', async () => {
      const mockExtraction = {
        extracted: 15,
        claims: [{ id: 'c1', content: 'Extracted claim' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExtraction)),
      });

      const result = await client.knowledge.extractFromDebate('debate-123', {
        confidence_threshold: 0.8,
        auto_promote: false,
      });

      expect(result.extracted).toBe(15);
      expect(result.claims).toHaveLength(1);
    });

    it('should promote extracted claims', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ promoted: 5 })),
      });

      const result = await client.knowledge.promoteExtracted(['claim-1', 'claim-2'], {
        target_tier: 'verified',
      });

      expect(result.promoted).toBe(5);
    });
  });

  // ===========================================================================
  // Confidence Decay
  // ===========================================================================

  describe('Confidence Decay', () => {
    it('should apply confidence decay', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ affected: 100, average_decay: 0.02 })),
      });

      const result = await client.knowledge.applyConfidenceDecay({
        scope: 'workspace',
        decay_rate: 0.01,
      });

      expect(result.affected).toBe(100);
      expect(result.average_decay).toBe(0.02);
    });

    it('should record confidence event', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ new_confidence: 0.92 })),
      });

      const result = await client.knowledge.recordConfidenceEvent('node-123', {
        type: 'validation',
        impact: 0.05,
        notes: 'Confirmed by expert',
      });

      expect(result.new_confidence).toBe(0.92);
    });

    it('should get confidence history', async () => {
      const mockHistory = {
        events: [
          { timestamp: '2024-01-01', type: 'decay', confidence_change: -0.01 },
        ],
        total: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHistory)),
      });

      const result = await client.knowledge.getConfidenceHistory('node-123');

      expect(result.events).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Export
  // ===========================================================================

  describe('Export', () => {
    it('should export as D3 JSON', async () => {
      const mockExport = {
        nodes: [{ id: 'n1', label: 'Node 1' }],
        links: [{ source: 'n1', target: 'n2' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.knowledge.exportD3({ scope: 'workspace', depth: 3 });

      expect(result.nodes).toHaveLength(1);
      expect(result.links).toHaveLength(1);
    });

    it('should export as GraphML', async () => {
      const mockXml = '<?xml version="1.0"?><graphml>...</graphml>';

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockXml)),
      });

      const result = await client.knowledge.exportGraphML({ scope: 'all' });

      expect(result).toContain('graphml');
    });

    it('should index a repository', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ indexed: 500, job_id: 'job-123' })),
      });

      const result = await client.knowledge.indexRepository('https://github.com/org/repo', {
        branch: 'main',
        paths: ['docs/', 'src/'],
      });

      expect(result.indexed).toBe(500);
      expect(result.job_id).toBe('job-123');
    });
  });

  // ===========================================================================
  // Auto-Curation
  // ===========================================================================

  describe('Auto-Curation', () => {
    it('should get curation policy', async () => {
      const mockPolicy = {
        policy: {
          auto_promote: true,
          auto_archive_days: 90,
          quality_threshold: 0.8,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPolicy)),
      });

      const result = await client.knowledge.getCurationPolicy();

      expect(result.policy).toBeDefined();
    });

    it('should set curation policy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ updated: true })),
      });

      const result = await client.knowledge.setCurationPolicy({
        auto_promote: true,
        auto_archive_days: 60,
        quality_threshold: 0.85,
      });

      expect(result.updated).toBe(true);
    });

    it('should get curation status', async () => {
      const mockStatus = {
        last_run: '2024-01-01T00:00:00Z',
        pending_actions: 15,
        health: 'healthy',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.knowledge.getCurationStatus();

      expect(result.health).toBe('healthy');
      expect(result.pending_actions).toBe(15);
    });

    it('should run curation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          actions_taken: 20,
          promoted: 10,
          archived: 10,
        })),
      });

      const result = await client.knowledge.runCuration({ dry_run: false });

      expect(result.actions_taken).toBe(20);
    });

    it('should get curation history', async () => {
      const mockHistory = {
        runs: [{ id: 'r1', timestamp: '2024-01-01', actions_taken: 50 }],
        total: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHistory)),
      });

      const result = await client.knowledge.getCurationHistory();

      expect(result.runs).toHaveLength(1);
    });

    it('should get quality scores', async () => {
      const mockScores = {
        scores: { tier_1: 0.95, tier_2: 0.85, tier_3: 0.7 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockScores)),
      });

      const result = await client.knowledge.getQualityScores();

      expect(result.scores).toBeDefined();
    });

    it('should get tier distribution', async () => {
      const mockDistribution = {
        distribution: { verified: 500, unverified: 200, archived: 100 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDistribution)),
      });

      const result = await client.knowledge.getTierDistribution();

      expect(result.distribution).toBeDefined();
    });
  });

  // ===========================================================================
  // Synchronization
  // ===========================================================================

  describe('Synchronization', () => {
    it('should sync from continuum memory', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ synced: 50 })),
      });

      const result = await client.knowledge.syncFromContinuum({ limit: 100 });

      expect(result.synced).toBe(50);
    });

    it('should sync from consensus memory', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ synced: 30 })),
      });

      const result = await client.knowledge.syncFromConsensus({ since: '2024-01-01' });

      expect(result.synced).toBe(30);
    });

    it('should sync from fact store', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ synced: 100 })),
      });

      const result = await client.knowledge.syncFromFacts({ limit: 200 });

      expect(result.synced).toBe(100);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle 404 for non-existent knowledge entry', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Knowledge entry not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.knowledge.get('nonexistent'))
        .rejects.toThrow('Knowledge entry not found');
    });

    it('should handle validation errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Content is required',
          code: 'MISSING_FIELD',
          field: 'content',
        }),
      });

      await expect(client.knowledge.add({ content: '' }))
        .rejects.toThrow('Content is required');
    });

    it('should handle permission errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Access denied',
          code: 'FORBIDDEN',
        }),
      });

      await expect(client.knowledge.storeGlobalFact({
        content: 'Test',
        source: 'test',
        confidence: 0.9,
      })).rejects.toThrow('Access denied');
    });

    it('should handle server errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          error: 'Internal server error',
          code: 'INTERNAL_ERROR',
        }),
      });

      await expect(client.knowledge.search('test'))
        .rejects.toThrow('Internal server error');
    });
  });
});
