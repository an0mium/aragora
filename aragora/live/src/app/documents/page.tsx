'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { FolderUploadDialog } from '@/components/FolderUploadDialog';
import { useAuth } from '@/context/AuthContext';

interface Document {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  created_at: string;
}

interface BatchJob {
  id: string;
  document_count: number;
  status: string;
  progress: number;
  created_at: string;
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    completed: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    processing: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    pending: 'bg-acid-blue/20 text-acid-blue border-acid-blue/40',
    failed: 'bg-acid-red/20 text-acid-red border-acid-red/40',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[status] || colors.pending}`}>
      {status.toUpperCase()}
    </span>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export default function DocumentsPage() {
  const { config: backendConfig } = useBackend();
  const { user, isAuthenticated } = useAuth();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [folderDialogOpen, setFolderDialogOpen] = useState(false);

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.url}/api/documents`, {
        headers: { 'Authorization': `Bearer ${user?.token || ''}` },
      });
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      setError('Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.url, user?.token]);

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 10000);
    return () => clearInterval(interval);
  }, [fetchDocuments]);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) await uploadFiles(files);
  };

  const uploadFiles = async (files: File[]) => {
    setUploading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));
      await fetch(`${backendConfig.url}/api/documents/batch`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${user?.token || ''}` },
        body: formData,
      });
      await fetchDocuments();
    } finally {
      setUploading(false);
    }
  };

  const stats = {
    total: documents.length,
    completed: documents.filter((d) => d.status === 'completed').length,
    processing: documents.filter((d) => d.status === 'processing').length,
    totalChunks: documents.reduce((sum, d) => sum + d.chunk_count, 0),
    totalSize: documents.reduce((sum, d) => sum + d.size_bytes, 0),
  };

  const filteredDocuments = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background">
      <Scanlines />
      <CRTVignette />

      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="hover:text-accent"><AsciiBannerCompact /></Link>
            <span className="text-muted font-mono text-sm">// DOCUMENTS</span>
          </div>
          <div className="flex items-center gap-3">
            <BackendSelector />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="card p-4"><div className="text-xs text-muted font-mono mb-1">TOTAL</div><div className="text-2xl font-bold text-accent">{stats.total}</div></div>
          <div className="card p-4"><div className="text-xs text-muted font-mono mb-1">PROCESSED</div><div className="text-2xl font-bold text-acid-green">{stats.completed}</div></div>
          <div className="card p-4"><div className="text-xs text-muted font-mono mb-1">PROCESSING</div><div className="text-2xl font-bold text-acid-yellow">{stats.processing}</div></div>
          <div className="card p-4"><div className="text-xs text-muted font-mono mb-1">CHUNKS</div><div className="text-2xl font-bold">{stats.totalChunks.toLocaleString()}</div></div>
          <div className="card p-4"><div className="text-xs text-muted font-mono mb-1">SIZE</div><div className="text-2xl font-bold">{formatFileSize(stats.totalSize)}</div></div>
        </div>

        <div
          className={`card p-8 mb-6 border-2 border-dashed ${dragActive ? 'border-accent bg-accent/10' : 'border-border'}`}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
        >
          <div className="text-center">
            <div className="text-4xl mb-3">📁</div>
            <div className="text-lg font-mono mb-2">{uploading ? 'UPLOADING...' : 'DROP FILES HERE'}</div>
            <div className="flex items-center justify-center gap-3">
              <label className="btn btn-primary cursor-pointer">
                <input type="file" multiple className="hidden" onChange={(e) => e.target.files && uploadFiles(Array.from(e.target.files))} />
                SELECT FILES
              </label>
              <button
                onClick={() => setFolderDialogOpen(true)}
                className="btn btn-secondary"
              >
                📂 UPLOAD FOLDER
              </button>
            </div>
          </div>
        </div>

        <FolderUploadDialog
          isOpen={folderDialogOpen}
          onClose={() => setFolderDialogOpen(false)}
          onComplete={() => {
            setFolderDialogOpen(false);
            fetchDocuments();
          }}
          apiBase={backendConfig.url}
          authToken={user?.token}
        />

        <div className="flex items-center justify-between mb-4">
          <input type="text" placeholder="Search..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="input w-64" />
          {selectedDocs.size > 0 && (
            <Link href={`/audit/new?documents=${Array.from(selectedDocs).join(',')}`} className="btn btn-primary">🔍 START AUDIT ({selectedDocs.size})</Link>
          )}
        </div>

        <PanelErrorBoundary title="Documents">
          <div className="card overflow-hidden">
            {loading ? (
              <div className="p-8 text-center animate-pulse text-muted font-mono">LOADING...</div>
            ) : filteredDocuments.length === 0 ? (
              <div className="p-8 text-center"><div className="text-4xl mb-3">📭</div><div className="text-muted font-mono">NO DOCUMENTS</div></div>
            ) : (
              <table className="w-full">
                <thead className="bg-surface border-b border-border">
                  <tr>
                    <th className="p-3 text-left"><input type="checkbox" onChange={(e) => setSelectedDocs(e.target.checked ? new Set(filteredDocuments.map(d => d.id)) : new Set())} /></th>
                    <th className="p-3 text-left font-mono text-xs text-muted">FILE</th>
                    <th className="p-3 text-left font-mono text-xs text-muted">STATUS</th>
                    <th className="p-3 text-left font-mono text-xs text-muted">SIZE</th>
                    <th className="p-3 text-left font-mono text-xs text-muted">CHUNKS</th>
                    <th className="p-3 text-left font-mono text-xs text-muted">UPLOADED</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDocuments.map((doc) => (
                    <tr key={doc.id} className="border-b border-border hover:bg-surface/50">
                      <td className="p-3"><input type="checkbox" checked={selectedDocs.has(doc.id)} onChange={(e) => { const n = new Set(selectedDocs); e.target.checked ? n.add(doc.id) : n.delete(doc.id); setSelectedDocs(n); }} /></td>
                      <td className="p-3 font-mono text-sm">{doc.filename}</td>
                      <td className="p-3"><StatusBadge status={doc.status} /></td>
                      <td className="p-3 font-mono text-sm">{formatFileSize(doc.size_bytes)}</td>
                      <td className="p-3 font-mono text-sm">{doc.chunk_count}</td>
                      <td className="p-3 font-mono text-sm text-muted">{formatDate(doc.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </PanelErrorBoundary>
      </main>

      <footer className="border-t border-border bg-surface/50 py-4 mt-8">
        <div className="container mx-auto px-4 flex items-center justify-between text-xs text-muted font-mono">
          <span>ARAGORA DOCUMENTS</span>
          <Link href="/audit" className="hover:text-accent">AUDIT DASHBOARD →</Link>
        </div>
      </footer>
    </div>
  );
}
