import { CreateWebWorkerMLCEngine, deleteModelAllInfoInCache, hasModelInCache, type MLCEngineInterface } from '@mlc-ai/web-llm';
import { MAX_ANSWER_TOKENS, MODELS, buildMessages, removeOnCloseEnabled, type Turn } from './llm-config';

let engine: MLCEngineInterface | null = null;
let loadedId: string | null = null;

export const loadedModelId = () => loadedId;

export async function cachedModelIds(): Promise<string[]> {
  const found: string[] = [];
  for (const m of MODELS) {
    try {
      if (await hasModelInCache(m.id)) found.push(m.id);
    } catch { /* treat as not cached */ }
  }
  return found;
}

export async function loadModel(modelId: string, onProgress: (fraction: number, text: string) => void): Promise<void> {
  if (engine && loadedId === modelId) return;
  await unloadModel();
  const worker = new Worker(new URL('../llm.worker.ts', import.meta.url), { type: 'module' });
  engine = await CreateWebWorkerMLCEngine(worker, modelId, {
    initProgressCallback: (r) => onProgress(r.progress, r.text),
  });
  loadedId = modelId;
}

export async function unloadModel(): Promise<void> {
  try {
    await engine?.unload();
  } finally {
    engine = null;
    loadedId = null;
  }
}

async function deleteAll(): Promise<void> {
  for (const m of MODELS) {
    try {
      await deleteModelAllInfoInCache(m.id);
    } catch { /* not cached */ }
  }
}

/** Remove every downloaded model from this browser. */
export async function deleteModelData(): Promise<void> {
  await unloadModel();
  await deleteAll();
}

/** Call once at startup: finishes a "remove on close" deletion requested in a previous visit. */
export async function cleanupIfRequested(): Promise<void> {
  if (!removeOnCloseEnabled() || engine) return;
  await deleteAll();
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
