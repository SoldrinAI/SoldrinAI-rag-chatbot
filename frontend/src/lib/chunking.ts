import { DIM } from './api';

export interface Chunk {
  text: string;
  vector: Float32Array;
}

export interface ChunkOptions {
  /** Start a new chunk when a sentence is less similar than this to the chunk so far. */
  threshold: number;
  /** Don't split on similarity before the chunk has at least this many characters. */
  minChars: number;
  /** Hard cap on chunk size (keeps the LLM context small). */
  maxChars: number;
}

export const DEFAULT_CHUNKING: ChunkOptions = { threshold: 0.35, minChars: 180, maxChars: 550 };

export function normalise(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Semantic chunking: walk the sentences in order and keep appending while each new
 * sentence is still similar to the running chunk centroid. A drop in similarity marks a
 * topic change, so a new chunk starts there. The chunk vector is the normalised mean of
 * its sentence vectors, so no second embedding pass is needed.
 */
export function semanticChunks(sentences: string[], vectors: Float32Array[], opts: ChunkOptions = DEFAULT_CHUNKING): Chunk[] {
  const chunks: Chunk[] = [];
  let texts: string[] = [];
  let sum = new Float32Array(DIM);
  let chars = 0;

  const flush = () => {
    if (!texts.length) return;
    chunks.push({ text: texts.join(' '), vector: normalise(sum) });
    texts = [];
    sum = new Float32Array(DIM);
    chars = 0;
  };

  sentences.forEach((sentence, i) => {
    const v = vectors[i];
    if (texts.length) {
      const similar = dot(normalise(sum), v);
      const topicShift = similar < opts.threshold && chars >= opts.minChars;
      if (topicShift || chars + sentence.length + 1 > opts.maxChars) flush();
    }
    texts.push(sentence);
    chars += sentence.length + 1;
    for (let d = 0; d < DIM; d++) sum[d] += v[d];
  });
  flush();
  return chunks;
}

/** Table rows: one row is one chunk. */
export function rowChunks(rows: string[], vectors: Float32Array[]): Chunk[] {
  return rows.map((text, i) => ({ text, vector: vectors[i] }));
}
