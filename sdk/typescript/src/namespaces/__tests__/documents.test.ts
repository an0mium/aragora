/**
 * Documents Namespace Tests
 *
 * Comprehensive tests for the documents namespace API including:
 * - Document listing
 * - Supported formats
 * - Document upload
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { DocumentsAPI } from '../documents';

interface MockClient {
  get: Mock;
  post: Mock;
}

describe('DocumentsAPI Namespace', () => {
  let api: DocumentsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
    };
    api = new DocumentsAPI(mockClient as any);
  });

  // ===========================================================================
  // Document Listing
  // ===========================================================================

  describe('Document Listing', () => {
    it('should list all documents', async () => {
      const mockDocuments = {
        documents: [
          {
            id: 'doc_1',
            name: 'report.pdf',
            format: 'pdf',
            size_bytes: 1024000,
            mime_type: 'application/pdf',
            uploaded_at: '2024-01-20T10:00:00Z',
            updated_at: '2024-01-20T10:00:00Z',
          },
          {
            id: 'doc_2',
            name: 'data.csv',
            format: 'csv',
            size_bytes: 50000,
            mime_type: 'text/csv',
            uploaded_at: '2024-01-19T15:00:00Z',
            updated_at: '2024-01-19T15:00:00Z',
          },
        ],
        total: 2,
      };
      mockClient.get.mockResolvedValue(mockDocuments);

      const result = await api.list();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/documents');
      expect(result.documents).toHaveLength(2);
      expect(result.total).toBe(2);
    });

    it('should list documents by debate', async () => {
      const mockDocuments = {
        documents: [{ id: 'doc_1', name: 'evidence.pdf', debate_id: 'd_123' }],
        total: 1,
      };
      mockClient.get.mockResolvedValue(mockDocuments);

      await api.list({ debate_id: 'd_123' });

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/documents?debate_id=d_123');
    });

    it('should list documents with pagination', async () => {
      const mockDocuments = { documents: [], total: 100 };
      mockClient.get.mockResolvedValue(mockDocuments);

      await api.list({ limit: 10, offset: 20 });

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/documents?limit=10&offset=20');
    });

    it('should list documents with all filters', async () => {
      const mockDocuments = { documents: [], total: 0 };
      mockClient.get.mockResolvedValue(mockDocuments);

      await api.list({ debate_id: 'd_123', limit: 5, offset: 10 });

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/documents?debate_id=d_123&limit=5&offset=10');
    });
  });

  // ===========================================================================
  // Document Formats
  // ===========================================================================

  describe('Document Formats', () => {
    it('should get supported formats', async () => {
      const mockFormats = {
        formats: [
          {
            extension: 'pdf',
            mime_types: ['application/pdf'],
            description: 'Portable Document Format',
            max_size_mb: 50,
          },
          {
            extension: 'docx',
            mime_types: ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            description: 'Microsoft Word Document',
            max_size_mb: 25,
          },
          {
            extension: 'csv',
            mime_types: ['text/csv'],
            description: 'Comma-Separated Values',
            max_size_mb: 100,
          },
          {
            extension: 'txt',
            mime_types: ['text/plain'],
            description: 'Plain Text',
            max_size_mb: 10,
          },
        ],
      };
      mockClient.get.mockResolvedValue(mockFormats);

      const result = await api.getFormats();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/documents/formats');
      expect(result.formats).toHaveLength(4);
      expect(result.formats[0].extension).toBe('pdf');
      expect(result.formats[0].max_size_mb).toBe(50);
    });
  });

  // ===========================================================================
  // Document Upload
  // ===========================================================================

  describe('Document Upload', () => {
    it('should upload document', async () => {
      const mockResult = {
        document: {
          id: 'doc_new',
          name: 'report.pdf',
          format: 'pdf',
          size_bytes: 1024000,
          mime_type: 'application/pdf',
          uploaded_at: '2024-01-20T10:00:00Z',
          updated_at: '2024-01-20T10:00:00Z',
        },
        extracted_text: 'This is the extracted content from the PDF...',
        page_count: 10,
      };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.upload({
        name: 'report.pdf',
        content: 'base64encodedcontent...',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/documents/upload', {
        name: 'report.pdf',
        content: 'base64encodedcontent...',
      });
      expect(result.document.id).toBe('doc_new');
      expect(result.page_count).toBe(10);
    });

    it('should upload document with format hint', async () => {
      const mockResult = {
        document: { id: 'doc_new2', name: 'data.csv', format: 'csv' },
      };
      mockClient.post.mockResolvedValue(mockResult);

      await api.upload({
        name: 'data.csv',
        content: 'base64content',
        format: 'csv',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/documents/upload', {
        name: 'data.csv',
        content: 'base64content',
        format: 'csv',
      });
    });

    it('should upload document with debate association', async () => {
      const mockResult = {
        document: { id: 'doc_new3', debate_id: 'd_123' },
      };
      mockClient.post.mockResolvedValue(mockResult);

      await api.upload({
        name: 'evidence.pdf',
        content: 'base64content',
        debate_id: 'd_123',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/documents/upload', {
        name: 'evidence.pdf',
        content: 'base64content',
        debate_id: 'd_123',
      });
    });

    it('should upload document with metadata', async () => {
      const mockResult = {
        document: { id: 'doc_new4', metadata: { category: 'legal', confidential: true } },
      };
      mockClient.post.mockResolvedValue(mockResult);

      await api.upload({
        name: 'contract.pdf',
        content: 'base64content',
        metadata: { category: 'legal', confidential: true },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/documents/upload', {
        name: 'contract.pdf',
        content: 'base64content',
        metadata: { category: 'legal', confidential: true },
      });
    });

    it('should upload document with all options', async () => {
      const mockResult = { document: { id: 'doc_full' } };
      mockClient.post.mockResolvedValue(mockResult);

      await api.upload({
        name: 'full-example.docx',
        content: 'base64content',
        format: 'docx',
        debate_id: 'd_456',
        metadata: { source: 'email', reviewed: false },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/documents/upload', {
        name: 'full-example.docx',
        content: 'base64content',
        format: 'docx',
        debate_id: 'd_456',
        metadata: { source: 'email', reviewed: false },
      });
    });
  });
});
