# Aragora Quickstart Video Script

**Duration:** 2 minutes
**Target:** Developers evaluating Aragora

---

## INTRO (0:00 - 0:15)

**[Screen: Aragora logo animation]**

> "What if your AI decisions could be stress-tested by multiple AI agents before you ship?"
>
> "I'm going to show you how Aragora helps you make better AI-powered decisions in under 2 minutes."

---

## THE PROBLEM (0:15 - 0:30)

**[Screen: Split view - single AI response vs multiple responses]**

> "When you ask one AI a question, you get one perspective. But for important decisions - architecture choices, code reviews, security assessments - one perspective isn't enough."
>
> "Aragora runs multi-agent debates where Claude, GPT-4, and Gemini argue, critique, and reach consensus on your questions."

---

## DEMO: FIRST DEBATE (0:30 - 1:00)

**[Screen: Terminal / Code editor]**

> "Let's run a debate. I'll install the SDK..."

```bash
npm install @aragora/sdk
```

> "...and create a simple debate:"

```typescript
import { createClient } from '@aragora/sdk';

const client = createClient({ apiKey: 'your-key' });

const debate = await client.debates.create({
  task: 'Should we use microservices or a monolith for our new product?',
  agents: ['claude', 'gpt-4', 'gemini'],
  protocol: { rounds: 3 }
});

console.log(debate.consensus.final_answer);
```

**[Screen: Show real debate output]**

> "Three AI agents just debated this question, challenged each other's assumptions, and reached consensus with 87% confidence."

---

## KEY FEATURES (1:00 - 1:30)

**[Screen: Quick cuts between features]**

> "But that's just the beginning."

**[Show: Gauntlet interface]**
> "Use Gauntlet to stress-test your specs before implementation."

**[Show: Analytics dashboard]**
> "Track agent performance with ELO rankings and detailed analytics."

**[Show: Workflow builder]**
> "Build automated workflows that combine debates, approvals, and integrations."

**[Show: VS Code extension]**
> "Or use our VS Code extension to run debates without leaving your editor."

---

## CALL TO ACTION (1:30 - 1:50)

**[Screen: aragora.ai website]**

> "Aragora integrates with LangChain, LlamaIndex, AutoGen, and runs anywhere - your laptop, your cloud, or our managed service."
>
> "Get started free at aragora.ai. Your first 100 debates are on us."

---

## OUTRO (1:50 - 2:00)

**[Screen: Logo + tagline]**

> "Aragora. Stress-test your AI decisions."

**[Show: QR code to docs.aragora.ai]**

---

## B-ROLL SHOTS NEEDED

1. Terminal showing `npm install` command
2. Code editor with SDK code
3. Debate results in terminal
4. Gauntlet interface showing spec analysis
5. Analytics dashboard with charts
6. Workflow builder canvas
7. VS Code extension in action
8. Website homepage

## VOICEOVER NOTES

- Tone: Confident, technical but accessible
- Speed: Moderate, let code breathe
- Energy: Building momentum toward CTA

## GRAPHICS

- Aragora logo (animated intro)
- Feature icons for quick cuts
- Code syntax highlighting
- Confidence score visualization
- QR code for docs

---

## SOCIAL CLIPS

### Clip 1: "The Problem" (15s)
> "When you ask one AI a question, you get one perspective. For important decisions, one perspective isn't enough. That's why we built Aragora."

### Clip 2: "First Debate" (30s)
> [Show code + output]
> "Three AI agents just debated whether to use microservices, challenged each other, and reached consensus with 87% confidence. Three lines of code."

### Clip 3: "Integration" (15s)
> "Aragora works with LangChain, LlamaIndex, AutoGen, or standalone. Your laptop or your cloud. Get started free."
