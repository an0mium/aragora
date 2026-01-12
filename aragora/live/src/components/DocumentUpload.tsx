'use client';

import { useState, useRef, useCallback } from 'react';

interface UploadedDocument {
  id: string;
  filename: string;
  word_count: number;
  page_count?: number;
  preview: string;
}

interface DocumentUploadProps {
  onDocumentsChange?: (docIds: string[]) => void;
  apiBase?: string;
}

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

const ACCEPTED_TYPES = {
  'application/pdf': '.pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
  'text/plain': '.txt',
  'text/markdown': '.md',
};

const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.markdown'];

export function DocumentUpload({ onDocumentsChange, apiBase = '' }: DocumentUploadProps) {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFile = useCallback(async (file: File) => {
    setStatus('uploading');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiBase}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      const newDoc: UploadedDocument = {
        id: data.document.id,
        filename: data.document.filename,
        word_count: data.document.word_count,
        page_count: data.document.page_count,
        preview: data.document.preview,
      };

      setDocuments((prev) => {
        const updated = [...prev, newDoc];
        onDocumentsChange?.(updated.map((d) => d.id));
        return updated;
      });

      setStatus('success');
      setTimeout(() => setStatus('idle'), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setStatus('error');
    }
  }, [apiBase, onDocumentsChange]);

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];

    // Validate file extension
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Supported: ${ACCEPTED_EXTENSIONS.join(', ')}`);
      setStatus('error');
      return;
    }

    // Validate MIME type matches extension (security check)
    const expectedMimes = Object.entries(ACCEPTED_TYPES)
      .filter(([, extension]) => extension === ext || (ext === '.markdown' && extension === '.md'))
      .map(([mime]) => mime);

    if (expectedMimes.length > 0 && file.type && !expectedMimes.includes(file.type)) {
      // Allow empty MIME type (some browsers don't set it)
      if (file.type !== '') {
        setError(`File MIME type (${file.type}) doesn't match extension (${ext}). Possible file spoofing.`);
        setStatus('error');
        return;
      }
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large. Maximum size is 10MB.');
      setStatus('error');
      return;
    }

    uploadFile(file);
  }, [uploadFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  }, [handleFileSelect]);

  const removeDocument = (docId: string) => {
    setDocuments((prev) => {
      const updated = prev.filter((d) => d.id !== docId);
      onDocumentsChange?.(updated.map((d) => d.id));
      return updated;
    });
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return '📄';
      case 'docx':
        return '📝';
      case 'txt':
        return '📃';
      case 'md':
      case 'markdown':
        return '📋';
      default:
        return '📎';
    }
  };

  return (
    <div className="panel" style={{ padding: 0 }}>
      <div className="p-4 border-b border-border">
        <h3 className="panel-title-sm">
          Documents
        </h3>
      </div>

      <div className="p-4 space-y-4">
        {/* Drop zone */}
        <div
          role="button"
          tabIndex={status === 'uploading' ? -1 : 0}
          aria-label="Upload a document. Click or press Enter to select a file, or drag and drop."
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          className={`
            border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all
            focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-surface
            ${isDragging
              ? 'border-accent bg-accent/10'
              : 'border-border hover:border-accent/50 hover:bg-surface'
            }
            ${status === 'uploading' ? 'opacity-50 pointer-events-none' : ''}
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(',')}
            onChange={(e) => handleFileSelect(e.target.files)}
            className="hidden"
          />

          {status === 'uploading' ? (
            <div className="flex items-center justify-center gap-2 text-text-muted">
              <svg
                className="animate-spin h-5 w-5"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Uploading...</span>
            </div>
          ) : (
            <>
              <div className="text-2xl mb-2">📎</div>
              <div className="text-sm text-text-muted">
                Drop files here or click to upload
              </div>
              <div className="text-xs text-text-muted mt-1">
                PDF, DOCX, TXT, MD (max 10MB)
              </div>
            </>
          )}
        </div>

        {/* Error message */}
        {status === 'error' && error && (
          <div className="bg-crimson/10 border border-crimson/30 rounded p-2 text-sm text-crimson">
            {error}
          </div>
        )}

        {/* Success message */}
        {status === 'success' && (
          <div className="bg-success/10 border border-success/30 rounded p-2 text-sm text-success">
            Document uploaded successfully
          </div>
        )}

        {/* Uploaded documents list */}
        {documents.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-text-muted uppercase tracking-wider">
              Attached ({documents.length})
            </div>
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="bg-surface border border-border rounded p-2 flex items-start gap-2"
              >
                <span className="text-lg flex-shrink-0">{getFileIcon(doc.filename)}</span>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">{doc.filename}</div>
                  <div className="text-xs text-text-muted">
                    {doc.word_count.toLocaleString()} words
                    {doc.page_count && doc.page_count > 1 && ` | ${doc.page_count} pages`}
                  </div>
                  {doc.preview && (
                    <div className="text-xs text-text-muted mt-1 line-clamp-2">
                      {doc.preview.slice(0, 100)}...
                    </div>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeDocument(doc.id);
                  }}
                  className="text-text-muted hover:text-crimson transition-colors p-1"
                  title="Remove document"
                  aria-label={`Remove document: ${doc.filename}`}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={1.5}
                    stroke="currentColor"
                    className="w-4 h-4"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
