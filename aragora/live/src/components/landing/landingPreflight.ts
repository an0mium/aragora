export interface LandingPreparedDebateOption {
  id: string;
  label: string;
  description: string;
  originalQuestion: string;
  interpretedQuestion: string;
  debatePrompt: string;
  agents: number;
  rounds: number;
  recommended?: boolean;
}

export interface LandingDebatePreflight {
  title: string;
  prompt: string;
  warning?: string;
  options: LandingPreparedDebateOption[];
}

type LandingPreflightResult =
  | { type: 'ready'; option: LandingPreparedDebateOption }
  | { type: 'confirm'; preflight: LandingDebatePreflight };

const DEFAULT_PROFILE = {
  agents: 3,
  rounds: 2,
};

const FAST_PROFILE = {
  agents: 2,
  rounds: 1,
};

function normalizePrompt(prompt: string): string {
  return prompt
    .replace(/\s+/g, ' ')
    .replace(/\s+\?/g, '?')
    .trim();
}

function extractLeadQuestion(prompt: string): string | null {
  const normalized = prompt.trim();
  if (!normalized) return null;

  const firstQuestion = normalized.match(/(.{12,240}?\?)/);
  if (firstQuestion?.[1]) {
    return normalizePrompt(firstQuestion[1]);
  }

  const firstLine = normalized
    .split(/\n+/)
    .map((line) => line.trim())
    .find(Boolean);

  if (!firstLine) return null;

  if (firstLine.length <= 240) {
    return normalizePrompt(firstLine);
  }

  return normalizePrompt(firstLine.slice(0, 220));
}

function buildOriginalOption(
  prompt: string,
  profile: { agents: number; rounds: number },
): LandingPreparedDebateOption {
  return {
    id: 'original',
    label: 'Use original wording',
    description: 'Debate the full prompt exactly as written.',
    originalQuestion: prompt,
    interpretedQuestion: prompt,
    debatePrompt: prompt,
    agents: profile.agents,
    rounds: profile.rounds,
  };
}

function buildPracticalFoodOption(
  prompt: string,
  lowerPrompt: string,
  profile: { agents: number; rounds: number },
): LandingPreparedDebateOption | null {
  const mentionsMicrowave = /\bmicrowave|reheat|heated|warming|warm(ed|ing)?\b/.test(lowerPrompt);
  const mentionsChicken = /\bchicken|nugget|nuggets|tender|tenders|poultry\b/.test(lowerPrompt);
  const mentionsKid = /\b4 year old|kid|child|toddler|son|daughter|hungry\b/.test(lowerPrompt);
  const mentionsEdgeCases = /\balive|dead|ethical|ethic|moral|killing|factory|grinding\b/.test(
    lowerPrompt,
  );

  if (!mentionsMicrowave || !mentionsChicken || !mentionsEdgeCases) {
    return null;
  }

  const practicalPrompt = lowerPrompt.includes('nugget')
    ? 'I am deciding whether reheating pre-cooked chicken nuggets in a microwave for a young child is safe and appropriate right now. Answer the practical food-safety question first, then briefly note any ethical concern only if it materially changes the decision.'
    : 'I am deciding whether cooking or reheating chicken in a microwave is safe and appropriate for eating right now. Answer the practical food-safety question first, then briefly separate any ethical issue from the cooking advice.';

  return {
    id: 'practical-food',
    label: 'Practical food-safety first',
    description: mentionsKid
      ? 'Focus on the parent/feeding question before the philosophical one.'
      : 'Focus on the practical cooking answer before abstract ethics.',
    originalQuestion: prompt,
    interpretedQuestion: practicalPrompt,
    debatePrompt: practicalPrompt,
    agents: profile.agents,
    rounds: profile.rounds,
    recommended: true,
  };
}

function buildLeadQuestionOption(
  prompt: string,
  leadQuestion: string,
  profile: { agents: number; rounds: number },
): LandingPreparedDebateOption | null {
  const normalizedLead = normalizePrompt(leadQuestion);
  const normalizedPrompt = normalizePrompt(prompt);

  if (!normalizedLead || normalizedLead === normalizedPrompt) {
    return null;
  }

  return {
    id: 'lead-question',
    label: 'Debate the lead question only',
    description: 'Ignore the pasted transcript and focus on the first concrete question.',
    originalQuestion: prompt,
    interpretedQuestion: normalizedLead,
    debatePrompt: normalizedLead,
    agents: profile.agents,
    rounds: profile.rounds,
    recommended: true,
  };
}

function dedupeOptions(
  options: LandingPreparedDebateOption[],
): LandingPreparedDebateOption[] {
  const seen = new Set<string>();
  return options.filter((option) => {
    const key = option.debatePrompt.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function prepareLandingDebate(prompt: string): LandingPreflightResult {
  const normalizedPrompt = normalizePrompt(prompt);
  const lowerPrompt = normalizedPrompt.toLowerCase();
  const isLongPrompt =
    normalizedPrompt.length > 480
    || (prompt.match(/\n/g)?.length ?? 0) >= 3;
  const looksLikePastedContext = /\bhere is what|codex said|claude said|skip to main content|worked for|consensus reached|decision receipt\b/.test(
    lowerPrompt,
  );
  const shouldUseFastProfile = isLongPrompt || looksLikePastedContext;
  const profile = shouldUseFastProfile ? FAST_PROFILE : DEFAULT_PROFILE;

  const originalOption = buildOriginalOption(normalizedPrompt, profile);
  const leadQuestion = extractLeadQuestion(prompt);
  const leadQuestionOption =
    shouldUseFastProfile && leadQuestion
      ? buildLeadQuestionOption(normalizedPrompt, leadQuestion, profile)
      : null;
  const practicalFoodOption = buildPracticalFoodOption(normalizedPrompt, lowerPrompt, profile);

  const options = dedupeOptions(
    [practicalFoodOption, leadQuestionOption, originalOption].filter(
      (option): option is LandingPreparedDebateOption => option !== null,
    ),
  );

  if (options.length === 1) {
    return { type: 'ready', option: options[0] };
  }

  const hasPracticalFoodFork = Boolean(practicalFoodOption);
  const title = hasPracticalFoodFork
    ? 'Choose which version of the question to debate'
    : 'Choose the scope for this landing-page debate';
  const preflightPrompt = hasPracticalFoodFork
    ? 'Your wording mixes a practical cooking question with ethical edge cases. Pick the version Aragora should answer first.'
    : 'The landing preview works best with one focused question. Pick the version you want Aragora to debate.';
  const warning = shouldUseFastProfile
    ? 'Long pasted prompts are slower and more likely to time out here. The focused option uses a faster landing profile.'
    : undefined;

  return {
    type: 'confirm',
    preflight: {
      title,
      prompt: preflightPrompt,
      warning,
      options,
    },
  };
}
