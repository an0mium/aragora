/**
 * Documents Namespace API
 *
 * Provides access to document management for debates and knowledge.
 * Supports various document formats and upload capabilities.
 */

/**
 * Document metadata
 */
export interface Document {
  id: string;
  name: string;
  format: string;
  size_bytes: number;
  mime_type: string;
  uploaded_at: string;
  updated_at: string;
  debate_id?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Supported document format
 */
export interface DocumentFormat {
  extension: string;
  mime_types: string[];
  description: string;
  max_size_mb: number;
}

/**
 * Upload result
 */
export interface UploadResult {
  document: Document;
  extracted_text?: string;
  page_count?: number;
}

/**
 * Internal client interface
 */
interface DocumentsClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
}

/**
 * Documents API namespace.
 *
 * Provides methods for document management:
 * - List documents
 * - Get supported formats
 * - Upload documents
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List documents
 * const docs = await client.documents.list();
 *
 * // Get supported formats
 * const formats = await client.documents.getFormats();
 *
 * // Upload a document
 * const result = await client.documents.upload({
 *   name: 'report.pdf',
 *   content: base64Data,
 *   debate_id: 'debate-123'
 * });
 * ```
 */
export class DocumentsAPI {
  constructor(private client: DocumentsClientInterface) {}

  /**
   * List documents.
   */
  async list(options?: { debate_id?: string; limit?: number; offset?: number }): Promise<{ documents: Document[]; total: number }> {
    const params = new URLSearchParams();
    if (options?.debate_id) params.set('debate_id', options.debate_id);
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());
    const query = params.toString() ? `?${params.toString()}` : '';
    return this.client.get(`/api/v1/documents${query}`);
  }

  /**
   * Get supported document formats.
   */
  async getFormats(): Promise<{ formats: DocumentFormat[] }> {
    return this.client.get('/api/v1/documents/formats');
  }

  /**
   * Upload a document.
   */
  async upload(body: {
    name: string;
    content: string;
    format?: string;
    debate_id?: string;
    metadata?: Record<string, unknown>;
  }): Promise<UploadResult> {
    return this.client.post('/api/v1/documents/upload', body);
  }
}
