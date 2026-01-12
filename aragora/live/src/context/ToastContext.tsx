'use client';

import React, { createContext, useContext, ReactNode } from 'react';
import { useToast, type ToastType } from '@/hooks/useToast';
import { ToastContainer } from '@/components/ToastContainer';

interface ToastContextType {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
  showError: (message: string, duration?: number) => void;
  showSuccess: (message: string, duration?: number) => void;
  clearToasts: () => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const { toasts, showToast, showError, showSuccess, removeToast, clearToasts } = useToast();

  return (
    <ToastContext.Provider value={{ showToast, showError, showSuccess, clearToasts }}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  );
}

export function useToastContext(): ToastContextType {
  const context = useContext(ToastContext);
  if (context === undefined) {
    throw new Error('useToastContext must be used within a ToastProvider');
  }
  return context;
}
