'use client';

import type { ClarifyingQuestion } from '@/hooks/usePromptEngine';
import { QuestionCard } from './QuestionCard';

interface RefinementFlowProps {
  questions: ClarifyingQuestion[];
  answers: Record<string, string>;
  onAnswer: (questionId: string, answer: string) => void;
}

export function RefinementFlow({ questions, answers, onAnswer }: RefinementFlowProps) {
  const answered = Object.keys(answers).length;
  const total = questions.length;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-sm text-[var(--acid-green)] font-bold">
          Clarifying Questions
        </h2>
        <span className="font-mono text-xs text-[var(--text-muted)]">
          {answered}/{total} answered
        </span>
      </div>
      <div className="w-full bg-[var(--border)] h-1">
        <div
          className="bg-[var(--acid-green)] h-1 transition-all"
          style={{ width: `${total > 0 ? (answered / total) * 100 : 0}%` }}
        />
      </div>
      <div className="space-y-4">
        {questions.map(q => (
          <QuestionCard
            key={q.id}
            question={q}
            selectedAnswer={answers[q.id]}
            onAnswer={onAnswer}
          />
        ))}
      </div>
    </div>
  );
}
