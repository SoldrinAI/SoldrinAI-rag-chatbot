import { CreateWebWorkerMLCEngine, deleteModelAllInfoInCache, hasModelInCache, type MLCEngineInterface } from '@mlc-ai/web-llm';
import { MAX_ANSWER_TOKENS, MODEL_ID, buildMessages, removeOnCloseEnabled, type Turn } from './llm-config';

let engine: MLCEngineInterface | null = null;

export async function isModelCached(): Promise<boolean> {
  try {
    return await hasModelInCache(MODEL_ID);
  } catch {
    return false;
  }
}

export async function loadModel(onProgress: (fraction: number, text: string) => void): Promise<void> {
  if (engine) return;
  const worker = new Worker(new URL('../llm.worker.ts', import.meta.url), { type: 'module' });
  engine = await CreateWebWorkerMLCEngine(worker, MODEL_ID, {
    initProgressCallback: (r) => onProgress(r.progress, r.text),
  });
}

export async function unloadModel(): Promise<void> {
  try {
    await engine?.unload();
  } finally {
    engine = null;
  }
}

/** Remove the downloaded weights from this browser. */
export async function deleteModelData(): Promise<void> {
  await unloadModel();
  await deleteModelAllInfoInCache(MODEL_ID);
}

/** Call once at startup: finishes a "remove on close" deletion requested in a previous visit. */
export async function cleanupIfRequested(): Promise<void> {
  if (!removeOnCloseEnabled() || engine) return;
  try {
    await deleteModelAllInfoInCache(MODEL_ID);
  } catch { /* nothing cached */ }
}

export function installCloseHandler() {
  window.addEventListener('pagehide', () => {
    if (removeOnCloseEnabled()) void engine?.unload();
  });
}

export async function* answer(question: string, excerpts: string[], history: Turn[]): AsyncGenerator<string> {
  if (!engine) throw new Error('The model is not loaded yet.');
  const stream = await engine.chat.completions.create({
    messages: buildMessages(question, excerpts, history),
    stream: true,
    temperature: 0.2,
    max_tokens: MAX_ANSWER_TOKENS,
  });
  for await (const part of stream) {
    const delta = part.choices[0]?.delta?.content;
    if (delta) yield delta;
  }
}
