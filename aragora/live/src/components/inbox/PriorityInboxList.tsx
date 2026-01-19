'use client';

import { useState, useEffect } from 'react';

interface Email {
  id: string;
  subject: string;
  sender: string;
  snippet: string;
  date: string;
  priority_score: number;
  labels: string[];
}

interface PriorityInboxListProps {
  apiBase: string;
  userId: string;
  authToken?: string;
}

export function PriorityInboxList({
  apiBase,
  userId,
  authToken,
}: PriorityInboxListProps) {
  const [emails, setEmails] = useState<Email[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEmails = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiBase}/api/inbox/priority?user_id=${userId}`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (!response.ok) throw new Error('Failed to fetch emails');
        const data = await response.json();
        setEmails(data.emails || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchEmails();
  }, [apiBase, userId, authToken]);

  if (isLoading) {
    return (
      <div className="border border-acid-green/30 bg-surface/50 p-4 rounded">
        <h3 className="text-acid-green font-mono text-sm mb-4">Priority Inbox</h3>
        <div className="text-center py-8 text-text-muted font-mono text-sm">
          Loading emails...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="border border-acid-green/30 bg-surface/50 p-4 rounded">
        <h3 className="text-acid-green font-mono text-sm mb-4">Priority Inbox</h3>
        <div className="text-center py-8 text-red-400 font-mono text-sm">
          {error}
        </div>
      </div>
    );
  }

  if (emails.length === 0) {
    return (
      <div className="border border-acid-green/30 bg-surface/50 p-4 rounded">
        <h3 className="text-acid-green font-mono text-sm mb-4">Priority Inbox</h3>
        <div className="text-center py-8 text-text-muted font-mono text-sm">
          No priority emails found
        </div>
      </div>
    );
  }

  return (
    <div className="border border-acid-green/30 bg-surface/50 p-4 rounded">
      <h3 className="text-acid-green font-mono text-sm mb-4">Priority Inbox</h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {emails.map((email) => (
          <div
            key={email.id}
            className="border border-acid-green/20 bg-bg/30 p-3 rounded cursor-pointer hover:bg-bg/50 transition-colors"
          >
            <div className="flex justify-between items-start mb-1">
              <span className="text-text font-mono text-sm truncate flex-1">
                {email.subject}
              </span>
              <span className="text-acid-green text-xs ml-2">
                {email.priority_score}%
              </span>
            </div>
            <div className="flex justify-between text-text-muted text-xs">
              <span>{email.sender}</span>
              <span>{email.date}</span>
            </div>
            <p className="text-text-muted text-xs mt-1 truncate">{email.snippet}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
