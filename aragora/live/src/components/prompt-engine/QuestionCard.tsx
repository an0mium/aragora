'use client';

import type { ClarifyingQuestion } from '@/hooks/usePromptEngine';

interface QuestionCardProps {
  question: ClarifyingQuestion;
  selectedAnswer: string | undefined;
  onAnswer: (questionId: string, answer: string) => void;
}

export function QuestionCard({ question, selectedAnswer, onAnswer }: QuestionCardProps) {
  return (
    <div className="border border-[var(--border)] p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-mono text-sm text-[var(--text)] font-bold">
          {question.question}
        </h3>
        <span className={`font-mono text-xs px-2 py-0.5 ${
          question.impact === 'high'
            ? 'text-[var(--crimson)] border border-[var(--crimson)]/30'
            : question.impact === 'medium'
            ? 'text-[var(--acid-yellow)] border border-[var(--acid-yellow)]/30'
            : 'text-[var(--text-muted)] border border-[var(--border)]'
        }`}>
          {question.impact}
        </span>
      </div>
      <p className="font-mono text-xs text-[var(--text-muted)]">
        {question.why_it_matters}
      </p>
      <div className="space-y-2">
        {question.options.map(option => (
          <button
            key={option.label}
            onClick={() => onAnswer(question.id, option.label)}
            className={`w-full text-left p-3 border font-mono text-xs ${
              selectedAnswer === option.label
                ? 'border-[var(--acid-green)] bg-[var(--acid-green)]/10 text-[var(--acid-green)]'
                : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--text-muted)]'
            }`}
          >
            <div className="font-bold">{option.label}</div>
            <div className="mt-1 text-[var(--text-muted)]">{option.description}</div>
            <div className="mt-1 text-[var(--text-muted)]/70 italic">Tradeoff: {option.tradeoff}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
