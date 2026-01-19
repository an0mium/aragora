'use client';

import { useState, useRef, useCallback, DragEvent } from 'react';

export interface UploadFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  progress: number;
  error?: string;
}

interface FileUploaderProps {
  onUpload: (files: File[]) => Promise<void>;
  accept?: string[];
  maxSize?: number; // bytes
  maxFiles?: number;
  multiple?: boolean;
  folderUpload?: boolean;
  disabled?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const DEFAULT_ACCEPT = [
  '.pdf', '.txt', '.md', '.doc', '.docx', '.xls', '.xlsx',
  '.ppt', '.pptx', '.csv', '.json', '.xml', '.html', '.htm',
  '.rtf', '.odt', '.ods', '.odp', '.epub'
];

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

/**
 * Drag-and-drop file uploader component
 *
 * Features:
 * - Drag and drop support
 * - Click to browse
 * - Multiple file selection
 * - Folder upload support
 * - File type validation
 * - Size validation
 * - Progress tracking
 */
export function FileUploader({
  onUpload,
  accept = DEFAULT_ACCEPT,
  maxSize = 100 * 1024 * 1024, // 100MB default
  maxFiles = 50,
  multiple = true,
  folderUpload = false,
  disabled = false,
  className = '',
  children,
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCountRef = useRef(0);

  const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  const validateFile = useCallback((file: File): string | null => {
    // Check file size
    if (file.size > maxSize) {
      return `File too large (max ${formatFileSize(maxSize)})`;
    }

    // Check file type
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    const mimeMatch = accept.some(a =>
      a.startsWith('.') ? ext === a.toLowerCase() : file.type.startsWith(a)
    );
    if (!mimeMatch && accept.length > 0) {
      return `File type not supported`;
    }

    return null;
  }, [accept, maxSize]);

  const processFiles = useCallback(async (fileList: FileList | File[]) => {
    const newFiles: UploadFile[] = [];
    const filesArray = Array.from(fileList);

    // Limit number of files
    const filesToProcess = filesArray.slice(0, maxFiles - files.length);

    for (const file of filesToProcess) {
      const error = validateFile(file);
      newFiles.push({
        file,
        id: generateId(),
        status: error ? 'error' : 'pending',
        progress: 0,
        error: error || undefined,
      });
    }

    setFiles(prev => [...prev, ...newFiles]);

    // Auto-upload valid files
    const validFiles = newFiles.filter(f => f.status === 'pending').map(f => f.file);
    if (validFiles.length > 0) {
      setIsUploading(true);
      try {
        await onUpload(validFiles);
        // Mark as completed
        setFiles(prev =>
          prev.map(f =>
            validFiles.includes(f.file)
              ? { ...f, status: 'completed', progress: 100 }
              : f
          )
        );
      } catch (err) {
        // Mark as error
        setFiles(prev =>
          prev.map(f =>
            validFiles.includes(f.file)
              ? {
                  ...f,
                  status: 'error',
                  error: err instanceof Error ? err.message : 'Upload failed',
                }
              : f
          )
        );
      } finally {
        setIsUploading(false);
      }
    }
  }, [files.length, maxFiles, onUpload, validateFile]);

  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current++;
    if (e.dataTransfer.types.includes('Files')) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current--;
    if (dragCountRef.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCountRef.current = 0;
    setIsDragging(false);

    if (disabled || isUploading) return;

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      processFiles(droppedFiles);
    }
  }, [disabled, isUploading, processFiles]);

  const handleClick = useCallback(() => {
    if (!disabled && !isUploading) {
      fileInputRef.current?.click();
    }
  }, [disabled, isUploading]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      processFiles(selectedFiles);
    }
    // Reset input to allow selecting same file again
    e.target.value = '';
  }, [processFiles]);

  const removeFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  }, []);

  const clearFiles = useCallback(() => {
    setFiles([]);
  }, []);

  const acceptString = accept.join(',');

  return (
    <div className={className}>
      {/* Drop zone */}
      <div
        onClick={handleClick}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-6 text-center
          transition-all duration-200 cursor-pointer
          ${isDragging
            ? 'border-acid-green bg-acid-green/10'
            : 'border-acid-green/30 hover:border-acid-green/50 hover:bg-acid-green/5'
          }
          ${disabled || isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptString}
          multiple={multiple}
          {...(folderUpload ? { webkitdirectory: '', directory: '' } : {})}
          onChange={handleFileInput}
          className="hidden"
          disabled={disabled || isUploading}
        />

        {children || (
          <>
            <div className="text-3xl mb-2 text-acid-green/70">
              {isDragging ? '>' : '+'}
            </div>
            <div className="font-mono text-sm text-text">
              {isDragging
                ? 'DROP FILES HERE'
                : isUploading
                ? 'UPLOADING...'
                : 'DRAG & DROP OR CLICK TO UPLOAD'}
            </div>
            <div className="text-xs text-text-muted mt-1">
              Max {formatFileSize(maxSize)} per file
              {maxFiles > 1 && ` • Up to ${maxFiles} files`}
            </div>
          </>
        )}
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted font-mono">
              {files.length} file{files.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={clearFiles}
              className="text-xs text-acid-green/70 hover:text-acid-green font-mono"
            >
              Clear all
            </button>
          </div>

          {files.map(f => (
            <FileItem
              key={f.id}
              file={f}
              onRemove={() => removeFile(f.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Individual file item in the upload list
 */
function FileItem({
  file,
  onRemove,
}: {
  file: UploadFile;
  onRemove: () => void;
}) {
  const statusColors = {
    pending: 'text-text-muted',
    uploading: 'text-acid-cyan',
    completed: 'text-acid-green',
    error: 'text-crimson',
  };

  const statusIcons = {
    pending: '...',
    uploading: '>>',
    completed: 'OK',
    error: '!!',
  };

  return (
    <div className="flex items-center gap-3 p-2 bg-surface border border-acid-green/20 rounded text-sm font-mono">
      <span className={`w-6 text-center ${statusColors[file.status]}`}>
        {statusIcons[file.status]}
      </span>

      <div className="flex-1 min-w-0">
        <div className="truncate text-text">{file.file.name}</div>
        <div className="text-xs text-text-muted">
          {formatFileSize(file.file.size)}
          {file.error && (
            <span className="text-crimson ml-2">{file.error}</span>
          )}
        </div>
      </div>

      {file.status === 'uploading' && (
        <div className="w-16">
          <div className="h-1 bg-acid-green/20 rounded overflow-hidden">
            <div
              className="h-full bg-acid-green transition-all"
              style={{ width: `${file.progress}%` }}
            />
          </div>
        </div>
      )}

      <button
        onClick={onRemove}
        className="text-text-muted hover:text-crimson transition-colors"
        title="Remove"
      >
        x
      </button>
    </div>
  );
}

/**
 * Compact file upload button (no drop zone)
 */
export function FileUploadButton({
  onUpload,
  accept = DEFAULT_ACCEPT,
  maxSize = 100 * 1024 * 1024,
  multiple = false,
  disabled = false,
  className = '',
  children = 'Upload',
}: Omit<FileUploaderProps, 'folderUpload' | 'maxFiles'> & {
  children?: React.ReactNode;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleClick = () => {
    if (!disabled && !isUploading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      setIsUploading(true);
      try {
        await onUpload(Array.from(selectedFiles));
      } finally {
        setIsUploading(false);
      }
    }
    e.target.value = '';
  };

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept.join(',')}
        multiple={multiple}
        onChange={handleFileInput}
        className="hidden"
        disabled={disabled || isUploading}
      />
      <button
        onClick={handleClick}
        disabled={disabled || isUploading}
        className={`
          px-3 py-1.5 font-mono text-sm
          border border-acid-green/30
          text-acid-green hover:bg-acid-green/10
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-colors
          ${className}
        `}
      >
        {isUploading ? 'Uploading...' : children}
      </button>
    </>
  );
}
