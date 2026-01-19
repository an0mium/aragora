/**
 * Global type extensions for the Aragora project
 */

// Extend File interface to include webkitRelativePath (used for directory uploads)
interface File {
  /**
   * Returns the path of the file relative to the directory selected by the user
   * in a directory picker (webkitdirectory).
   * @see https://developer.mozilla.org/en-US/docs/Web/API/File/webkitRelativePath
   */
  readonly webkitRelativePath: string;
}

// Input element extension for directory upload attributes
declare namespace React {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}
