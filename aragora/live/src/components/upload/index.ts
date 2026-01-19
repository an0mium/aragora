/**
 * Upload Components - Unified media upload system.
 *
 * Provides smart file upload with:
 * - Multi-file drag-and-drop with folder support
 * - Smart file type detection and processing
 * - Cloud storage integration (Google Drive, OneDrive, Dropbox)
 * - YouTube/URL import
 * - Recording capabilities
 */

export {
  UnifiedMediaUploader,
  type FileCategory,
  type ProcessingAction,
  type UploadItem,
  type UnifiedMediaUploaderProps,
} from './UnifiedMediaUploader';

export {
  CloudStoragePicker,
  type CloudProvider,
  type CloudFile,
  type CloudStoragePickerProps,
} from './CloudStoragePicker';
