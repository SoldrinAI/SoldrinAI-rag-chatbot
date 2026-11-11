// Lightweight settings and helpers. Kept free of the web-llm import so the page loads fast;
// the heavy engine lives in llm.ts and is only fetched when the user starts the model.

export interface ModelOption {
  id: string;          // WebLLM prebuilt model id
  label: string;
  downloadMB: number;  // approximate, shown to the user before downloading
  memoryMB: number;    // approximate GPU memory needed while running
  note: string;
}

/** 4-bit instruction-tuned models. Weights are fetched from the Hugging Face CDN, smallest first. */
export const MODELS: ModelOption[] = [
  { id: 'SmolLM2-360M-Instruct-q4f16_1-MLC', label: 'SmolLM2 360M', downloadMB: 230, memoryMB: 380,
    note: 'Fastest and smallest. Expect weak, sometimes wrong answers.' },
  { id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 0.5B', downloadMB: 300, memoryMB: 950,
    note: 'Small and quick. Fine for simple lookups, often garbles details.' },
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 1B', downloadMB: 700, memoryMB: 880,
    note: 'A good balance of size and fluency.' },
  { id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 1.5B', downloadMB: 950, memoryMB: 1630,
    note: 'Noticeably more accurate with tables and exact values.' },
  { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 3B', downloadMB: 1800, memoryMB: 2500,
    note: 'Best quality here, but a big download and needs a capable GPU.' },
];
export const DEFAULT_MODEL_ID = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';

export function modelById(id: string): ModelOption {
  return MODELS.find((m) => m.id === id) ?? MODELS[1];
}

export const sizeLabel = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

/** Small context on purpose: few retrieved chunks, short history, short answers. */
export const TOP_K = 3;
export const MAX_CONTEXT_CHARS = 500;
export const HISTORY_TURNS = 1;
export const MAX_ANSWER_TOKENS = 256;

const REMOVE_FLAG = 'soldrinai-rag:remove-model-on-close';

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

export function webGpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/** "Remove the model when I leave" is best effort: browsers can't await work during unload,
 *  so we remember the choice and finish the deletion at the start of the next visit. */
export function setRemoveOnClose(on: boolean) {
  try {
    if (on) localStorage.setItem(REMOVE_FLAG, '1');
    else localStorage.removeItem(REMOVE_FLAG);
  } catch { /* storage blocked: ignore */ }
}

export function removeOnCloseEnabled(): boolean {
  try {
    return localStorage.getItem(REMOVE_FLAG) === '1';
  } catch {
    return false;
  }
}

const REFUSAL = /\b(could ?n[o']t find|cannot find|can't find|no information|not (?:found|mentioned|provided))\b/i;

/**
 * Small models follow instructions placed next to the question much better than a long system
 * prompt, so the excerpts travel in the user turn. The excerpts were already ranked as the best
 * matches, so the prompt tells the model to use them and to treat loose requests ("I want red")
 * as a request to list the matching items, rather than refusing.
 */
export function buildMessages(question: string, excerpts: string[], history: Turn[]) {
  const context = excerpts.map((e, i) => `[${i + 1}] ${e.slice(0, MAX_CONTEXT_CHARS)}`).join('\n');
  // Past refusals make a small model refuse again, so they are not carried forward.
  const past = history.filter((t, i) => !(t.role === 'assistant' && REFUSAL.test(t.content)) &&
    !(t.role === 'user' && history[i + 1]?.role === 'assistant' && REFUSAL.test(history[i + 1].content)));
  return [
    {
      role: 'system' as const,
      content:
        'You are a helpful assistant that answers questions about the user\'s uploaded document. ' +
        'You are given the most relevant excerpts. Write a short natural-language answer in your own words; ' +
        'never copy an excerpt word for word. Read every excerpt, pick the ones that fit the question, ' +
        'and quote only the exact values (names, numbers, statuses) that appear in them. ' +
        'Mention the excerpt numbers you used, like [1]. If the user states a wish or a keyword ' +
        '(for example "I want red"), say which items match, one per line. ' +
        'Only if none of the excerpts are relevant, say the document does not seem to cover it.',
    },
    ...past.slice(-HISTORY_TURNS * 2),
    {
      role: 'user' as const,
      content: `Document excerpts:\n${context}\n\nQuestion: ${question}\n\nAnswer using the excerpts above.`,
    },
  ];
}
