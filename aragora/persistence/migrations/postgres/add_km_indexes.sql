-- PostgreSQL Performance Indexes for Knowledge Mound Tables
-- Run with: psql -d aragora < add_km_indexes.sql
-- Uses CONCURRENTLY to avoid table locks in production

-- =============================================================================
-- Knowledge Nodes (core metadata)
-- =============================================================================

-- Workspace scoping (most common filter)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_workspace
    ON knowledge_nodes(workspace_id);

-- Workspace + type (common combined filter)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_workspace_type
    ON knowledge_nodes(workspace_id, node_type);

-- Workspace + visibility (access control queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_workspace_visibility
    ON knowledge_nodes(workspace_id, visibility);

-- Created timestamp for time-ordered queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_created
    ON knowledge_nodes(created_at DESC);

-- Workspace + created (paginated listing)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_workspace_created
    ON knowledge_nodes(workspace_id, created_at DESC);

-- Content hash for deduplication checks (fast lookups)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_content_hash
    ON knowledge_nodes(content_hash);

-- Confidence filtering (high-confidence knowledge)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_confidence
    ON knowledge_nodes(confidence DESC) WHERE confidence > 0.7;

-- Validation status for revalidation queues
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_validation
    ON knowledge_nodes(validation_status) WHERE validation_status != 'verified';

-- Revalidation requests (background job queue)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_revalidation
    ON knowledge_nodes(workspace_id, updated_at) WHERE revalidation_requested = TRUE;

-- =============================================================================
-- Provenance Chains (audit and lineage tracking)
-- =============================================================================

-- Node lookup (primary access pattern)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provenance_chains_node
    ON provenance_chains(node_id);

-- Source lookup (find all derived content)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provenance_chains_source
    ON provenance_chains(source_type, source_id);

-- Debate provenance (debate audit trails)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provenance_chains_debate
    ON provenance_chains(debate_id) WHERE debate_id IS NOT NULL;

-- User provenance (user contribution tracking)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provenance_chains_user
    ON provenance_chains(user_id) WHERE user_id IS NOT NULL;

-- Parent chain traversal (lineage queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provenance_chains_parent
    ON provenance_chains(parent_provenance_id) WHERE parent_provenance_id IS NOT NULL;

-- =============================================================================
-- Knowledge Relationships (graph traversal)
-- =============================================================================

-- Outgoing edges (traverse from node)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relationships_from
    ON knowledge_relationships(from_node_id);

-- Incoming edges (traverse to node)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relationships_to
    ON knowledge_relationships(to_node_id);

-- Bidirectional lookup (common in graph queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relationships_nodes
    ON knowledge_relationships(from_node_id, to_node_id);

-- Relationship type filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relationships_type
    ON knowledge_relationships(relationship_type);

-- Strong relationships (confidence > threshold)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relationships_strong
    ON knowledge_relationships(from_node_id, strength DESC) WHERE strength > 0.5;

-- =============================================================================
-- Node Topics (categorization and semantic search)
-- =============================================================================

-- Topic lookup (find all nodes with topic)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_topics_topic
    ON node_topics(topic);

-- =============================================================================
-- Culture Patterns (organizational learning)
-- =============================================================================

-- Workspace patterns (common query)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_culture_patterns_workspace
    ON culture_patterns(workspace_id);

-- Workspace + pattern type (specific pattern lookup)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_culture_patterns_workspace_type
    ON culture_patterns(workspace_id, pattern_type);

-- Pattern key lookup (deduplication)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_culture_patterns_key
    ON culture_patterns(workspace_id, pattern_key);

-- Confidence filtering (high-confidence patterns)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_culture_patterns_confidence
    ON culture_patterns(confidence DESC) WHERE confidence > 0.7;

-- =============================================================================
-- Staleness Checks (revalidation queue)
-- =============================================================================

-- Node staleness history
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_staleness_checks_node
    ON staleness_checks(node_id, checked_at DESC);

-- Pending revalidations (job queue)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_staleness_checks_pending
    ON staleness_checks(node_id) WHERE revalidation_requested = TRUE AND revalidation_completed_at IS NULL;

-- =============================================================================
-- Access Grants (permission checks - hot path)
-- =============================================================================

-- Item permissions lookup (primary access pattern)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_grants_item
    ON access_grants(item_id);

-- Grantee permissions (user's accessible items)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_grants_grantee
    ON access_grants(grantee_type, grantee_id);

-- Combined lookup (item + grantee, most common)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_grants_item_grantee
    ON access_grants(item_id, grantee_type, grantee_id);

-- Expiration cleanup (background job)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_grants_expires
    ON access_grants(expires_at) WHERE expires_at IS NOT NULL;

-- =============================================================================
-- Federation Nodes (distributed KM)
-- =============================================================================

-- Node status monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_nodes_status
    ON federation_nodes(status);

-- Heartbeat monitoring (find stale nodes)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_nodes_heartbeat
    ON federation_nodes(last_heartbeat DESC) WHERE status = 'active';

-- Region filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_nodes_region
    ON federation_nodes(region) WHERE region IS NOT NULL;

-- =============================================================================
-- Federation Sync State (replication tracking)
-- =============================================================================

-- Source node syncs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_sync_source
    ON federation_sync_state(source_node_id);

-- Target node syncs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_sync_target
    ON federation_sync_state(target_node_id);

-- Pending syncs (job queue)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_sync_pending
    ON federation_sync_state(next_sync_at) WHERE sync_status = 'pending';

-- Failed syncs (retry queue)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_sync_failed
    ON federation_sync_state(source_node_id) WHERE sync_status = 'failed';

-- =============================================================================
-- Federation Item Ownership (distributed ownership)
-- =============================================================================

-- Owner lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_ownership_owner
    ON federation_item_ownership(owner_node_id);

-- Authoritative items (primary copies)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_federation_ownership_authoritative
    ON federation_item_ownership(owner_node_id) WHERE is_authoritative = TRUE;

-- =============================================================================
-- Distributed Locks (coordination)
-- =============================================================================

-- Owner cleanup (release stale locks)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_distributed_locks_owner
    ON distributed_locks(owner_node_id);

-- Expired locks (cleanup job)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_distributed_locks_expires
    ON distributed_locks(expires_at) WHERE expires_at < NOW();

-- =============================================================================
-- Archived Nodes (historical data)
-- =============================================================================

-- Original node lookup (restore queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archived_nodes_original
    ON archived_nodes(original_id);

-- Workspace archives
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_archived_nodes_workspace
    ON archived_nodes(workspace_id, archived_at DESC);

-- =============================================================================
-- Post-creation: Analyze tables for query planner
-- =============================================================================
-- Run ANALYZE on all tables after index creation:
-- ANALYZE knowledge_nodes;
-- ANALYZE provenance_chains;
-- ANALYZE knowledge_relationships;
-- ANALYZE node_topics;
-- ANALYZE culture_patterns;
-- ANALYZE staleness_checks;
-- ANALYZE access_grants;
-- ANALYZE federation_nodes;
-- ANALYZE federation_sync_state;
-- ANALYZE federation_item_ownership;
-- ANALYZE distributed_locks;
-- ANALYZE archived_nodes;
