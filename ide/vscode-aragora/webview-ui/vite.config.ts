import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        entryFileNames: 'main.js',
        assetFileNames: 'main.[ext]',
      },
    },
    sourcemap: true,
    // Reduce chunk size for webview
    chunkSizeWarningLimit: 500,
  },
  // Ensure assets are relative for webview
  base: './',
});
