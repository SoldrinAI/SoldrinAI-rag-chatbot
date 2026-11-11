export interface Parsed {
  kind: 'prose' | 'table';
  name: string;
  units: string[];
  truncated: boolean;
  limits: string;
  columns?: string[];
}

export interface Limits {
  maxUploadBytes: number;
  extensions: string[];
}

export const DEFAULT_LIMITS: Limits = {
  maxUploadBytes: 4 * 1024 * 1024,
  extensions: ['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx'],
};

export const EMBED_BATCH = 64;
export const DIM = 384;

// API URLs are relative on purpose: the same build works at the site root and under a sub-path
// such as /rag-search/ (the page URL must end in a slash there).
async function failure(res: Response): Promise<Error> {
  let message = `Request failed (${res.status}).`;
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') message = body.detail;
    else if (Array.isArray(body.detail)) message = 'The request was not accepted. Try a smaller file.';
  } catch { /* keep the generic message */ }
  return new Error(message);
}

export async function parseFile(file: File): Promise<Parsed> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('api/parse', { method: 'POST', body: form });
  if (!res.ok) throw await failure(res);
  return res.json();
}

/** Decode base64 float16 (as produced by the API) into Float32 vectors of length DIM. */
export function decodeVectors(b64: string): Float32Array[] {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const n = bytes.length / 2 / DIM;
  const out: Float32Array[] = [];
  for (let v = 0; v < n; v++) {
    const vec = new Float32Array(DIM);
    for (let d = 0; d < DIM; d++) vec[d] = halfToFloat(view.getUint16((v * DIM + d) * 2, true));
    out.push(vec);
  }
  return out;
}

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export async function embedTexts(texts: string[], onProgress?: (done: number, total: number) => void): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const res = await fetch('api/embed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: texts.slice(i, i + EMBED_BATCH) }),
    });
    if (!res.ok) throw await failure(res);
    const { vectors } = await res.json();
    out.push(...decodeVectors(vectors));
    onProgress?.(Math.min(i + EMBED_BATCH, texts.length), texts.length);
  }
  return out;
}

export async function fetchLimits(): Promise<Limits> {
  try {
    const res = await fetch('api/limits');
    return res.ok ? await res.json() : DEFAULT_LIMITS;
  } catch {
    return DEFAULT_LIMITS;
  }
}
