import { describe, expect, it } from 'vitest';
import { DIM } from './api';
import { normalise, rowChunks, semanticChunks, type Chunk } from './chunking';
import { bm25, buildIndex, hybridSearch, tokenize } from './search';

function unit(axis: number, mix = 0): Float32Array {
  const v = new Float32Array(DIM);
  v[axis] = 1;
  v[(axis + 1) % DIM] = mix;
  return normalise(v);
}

describe('bm25', () => {
  it('ranks the document containing the rare query term first', () => {
    const docs = [tokenize('the cat sat on the mat'), tokenize('invoice 4821 is overdue'), tokenize('the dog ran')];
    const s = bm25(docs, tokenize('invoice 4821'));
    expect(s[1]).toBeGreaterThan(s[0]);
    expect(s[1]).toBeGreaterThan(s[2]);
  });
});

describe('semanticChunks', () => {
  it('splits where the topic changes and never exceeds the size cap', () => {
    const a = 'Dogs are loyal companions that enjoy long walks outside every day.';
    const b = 'Invoices are due within thirty days of the billing date for all clients.';
    const sentences = [a, a, a, b, b, b];
    const vectors = [unit(0), unit(0), unit(0), unit(5), unit(5), unit(5)];
    const chunks = semanticChunks(sentences, vectors, { threshold: 0.5, minChars: 50, maxChars: 400 });
    expect(chunks.length).toBe(2);
    expect(chunks[0].text.startsWith('Dogs')).toBe(true);
    expect(chunks[1].text.startsWith('Invoices')).toBe(true);
    chunks.forEach((c) => expect(c.text.length).toBeLessThanOrEqual(400));
  });
});

describe('hybridSearch', () => {
  const chunks: Chunk[] = rowChunks(
    ['Name: Ana; Role: Veterinarian', 'Name: Raj; Role: Mechanic', 'Name: Lena; Role: Chef'],
    [unit(0), unit(1), unit(2)],
  );
  const index = buildIndex(chunks);

  it('uses semantic similarity when there is no keyword match', () => {
    const hits = hybridSearch(index, 'someone who treats animals', unit(0, 0.1), 2);
    expect(hits[0].index).toBe(0);
    expect(hits).toHaveLength(2);
  });

  it('uses keywords to break ties on exact terms', () => {
    const hits = hybridSearch(index, 'Mechanic', normalise(new Float32Array(DIM).fill(0.01)), 1);
    expect(hits[0].index).toBe(1);
  });
});
