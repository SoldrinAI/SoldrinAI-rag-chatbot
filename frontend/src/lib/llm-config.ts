// Lightweight settings and helpers. Kept free of the web-llm import so the page loads fast;
// the heavy engine lives in llm.ts and is only fetched when the user starts the model.

/** Small instruction-tuned model, 4-bit quantised. Weights are fetched from the Hugging Face CDN. */
export const MODEL_ID = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
export const MODEL_LABEL = 'Qwen2.5 0.5B Instruct';
export const MODEL_DOWNLOAD_MB = 300; // approximate, shown to the user before downloading

/** Small context on purpose: few retrieved chunks, short history, short answers. */
export const TOP_K = 3;
export const MAX_CONTEXT_CHARS = 500;
export const HISTORY_TURNS = 2;
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

export function buildMessages(question: string, excerpts: string[], history: Turn[]) {
  const context = excerpts.map((e, i) => `[${i + 1}] ${e.slice(0, MAX_CONTEXT_CHARS)}`).join('\n');
  return [
    {
      role: 'system' as const,
      content:
        'You answer questions using ONLY the numbered excerpts from the user\'s document. ' +
        'Cite the excerpt numbers you used, like [1]. If the excerpts do not contain the answer, ' +
        'say you could not find it in the document. Be concise.\n\nExcerpts:\n' + context,
    },
    ...history.slice(-HISTORY_TURNS * 2),
    { role: 'user' as const, content: question },
  ];
}
