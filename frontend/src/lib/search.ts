import { dot, type Chunk } from './chunking';

export const VECTOR_WEIGHT = 0.65;
const TOKEN_RE = /[A-Za-z0-9_]+/g;

export interface Hit {
  index: number;
  score: number;
  vectorScore: number;
  bm25Score: number;
}

export function tokenize(text: string): string[] {
  return (text.match(TOKEN_RE) ?? []).map((t) => t.toLowerCase());
}

export function bm25(docs: string[][], query: string[]): number[] {
  const n = docs.length;
  const scores = new Array<number>(n).fill(0);
  if (!n || !query.length) return scores;
  const avg = docs.reduce((s, d) => s + d.length, 0) / n || 1;
  const df = new Map<string, number>();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
  docs.forEach((d, i) => {
    const counts = new Map<string, number>();
    for (const t of d) counts.set(t, (counts.get(t) ?? 0) + 1);
    for (const t of query) {
      const c = counts.get(t);
      if (!c) continue;
      const f = df.get(t)!;
      const idf = Math.log(1 + (n - f + 0.5) / (f + 0.5));
      scores[i] += (idf * c * 2.2) / (c + 1.2 * (0.25 + (0.75 * d.length) / avg));
    }
  });
  return scores;
}

function minMax(x: number[]): number[] {
  if (!x.length) return x;
  const lo = Math.min(...x);
  const hi = Math.max(...x);
  if (hi === lo) return x.map(() => (hi > 0 ? 1 : 0));
  return x.map((v) => (v - lo) / (hi - lo));
}

export interface Index {
  chunks: Chunk[];
  tokens: string[][];
}

export function buildIndex(chunks: Chunk[]): Index {
  return { chunks, tokens: chunks.map((c) => tokenize(c.text)) };
}

/** Hybrid ranking: 65% semantic (cosine) + 35% BM25 keyword, each min-max normalised. */
export function hybridSearch(index: Index, query: string, queryVector: Float32Array, limit: number): Hit[] {
  const vec = minMax(index.chunks.map((c) => Math.max(0, dot(c.vector, queryVector))));
  const lex = minMax(bm25(index.tokens, tokenize(query)));
  return index.chunks
    .map((_, i) => ({
      index: i,
      score: VECTOR_WEIGHT * vec[i] + (1 - VECTOR_WEIGHT) * lex[i],
      vectorScore: vec[i],
      bm25Score: lex[i],
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}
