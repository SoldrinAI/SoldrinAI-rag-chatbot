import { useEffect, useRef, useState, type DragEvent, type FormEvent } from 'react';
import {
  AlertCircle, Cpu, Download, FileText, LoaderCircle, RotateCcw, Send, ShieldCheck, Trash2, UploadCloud,
} from 'lucide-react';
import { DEFAULT_LIMITS, embedTexts, fetchLimits, parseFile, type Limits, type Parsed } from './lib/api';
import { rowChunks, semanticChunks } from './lib/chunking';
import { buildIndex, hybridSearch, type Index } from './lib/search';
import {
  MODEL_DOWNLOAD_MB, MODEL_LABEL, TOP_K, removeOnCloseEnabled, setRemoveOnClose, webGpuAvailable, type Turn,
} from './lib/llm-config';

// The model engine is large, so it is only fetched when the visitor actually needs it.
const engine = () => import('./lib/llm');

interface Source { text: string; score: number }
interface Message extends Turn { sources?: Source[] }
type ModelState = 'idle' | 'loading' | 'ready' | 'error';

const mb = (b: number) => `${(b / 1024 / 1024).toFixed(0)} MB`;

export default function App() {
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [doc, setDoc] = useState<Parsed | null>(null);
  const [index, setIndex] = useState<Index | null>(null);
  const [indexing, setIndexing] = useState<{ stage: string; pct: number } | null>(null);
  const [fileError, setFileError] = useState('');
  const [dragging, setDragging] = useState(false);

  const [modelState, setModelState] = useState<ModelState>('idle');
  const [modelCached, setModelCached] = useState(false);
  const [modelProgress, setModelProgress] = useState({ pct: 0, text: '' });
  const [modelError, setModelError] = useState('');
  const [removeOnClose, setRemoveOnCloseState] = useState(removeOnCloseEnabled());
  const gpu = webGpuAvailable();

  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchLimits().then(setLimits);
    if (!webGpuAvailable()) return;
    engine().then(async (m) => {
      m.installCloseHandler();
      await m.cleanupIfRequested();
      setModelCached(await m.isModelCached());
    }).catch(() => { /* engine unavailable: the download button still works */ });
  }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  async function handleFile(file: File | undefined) {
    if (!file) return;
    setFileError('');
    const ext = file.name.toLowerCase().split('.').pop() ?? '';
    if (!limits.extensions.includes(ext)) {
      setFileError(`Unsupported file type. Upload one of: ${limits.extensions.join(', ')}.`);
      return;
    }
    if (file.size > limits.maxUploadBytes) {
      setFileError(`File is too large (${mb(file.size)}). The limit is ${mb(limits.maxUploadBytes)}.`);
      return;
    }
    clearDocument();
    try {
      setIndexing({ stage: 'Reading file', pct: 5 });
      const parsed = await parseFile(file);
      const vectors = await embedTexts(parsed.units, (done, total) =>
        setIndexing({
          stage: parsed.kind === 'table' ? 'Embedding rows' : 'Embedding sentences',
          pct: 10 + Math.round((done / total) * 85),
        }));
      setIndexing({ stage: 'Building index', pct: 97 });
      const chunks = parsed.kind === 'table' ? rowChunks(parsed.units, vectors) : semanticChunks(parsed.units, vectors);
      setIndex(buildIndex(chunks));
      setDoc(parsed);
    } catch (e) {
      setFileError(e instanceof Error ? e.message : 'Something went wrong while reading the file.');
    } finally {
      setIndexing(null);
    }
  }

  function clearDocument() {
    setDoc(null);
    setIndex(null);
    setMessages([]);
    setFileError('');
  }

  async function startModel() {
    setModelState('loading');
    setModelError('');
    try {
      await (await engine()).loadModel((fraction, text) => setModelProgress({ pct: Math.round(fraction * 100), text }));
      setModelState('ready');
      setModelCached(true);
    } catch (e) {
      setModelError(e instanceof Error ? e.message : 'The model could not be loaded.');
      setModelState('error');
    }
  }

  async function removeModel() {
    await (await engine()).deleteModelData();
    setModelState('idle');
    setModelCached(false);
    setModelProgress({ pct: 0, text: '' });
  }

  function toggleRemoveOnClose(on: boolean) {
    setRemoveOnClose(on);
    setRemoveOnCloseState(on);
  }

  async function ask(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy || !index || modelState !== 'ready') return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    setQuestion('');
    setBusy(true);
    setMessages((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    try {
      const [qv] = await embedTexts([q]);
      const hits = hybridSearch(index, q, qv, TOP_K);
      const sources = hits.map((h) => ({ text: index.chunks[h.index].text, score: h.score }));
      const patch = (content: string, withSources = false) =>
        setMessages((m) => {
          const copy = [...m];
          copy[copy.length - 1] = { role: 'assistant', content, ...(withSources ? { sources } : {}) };
          return copy;
        });
      let text = '';
      for await (const piece of (await engine()).answer(q, sources.map((s) => s.text), history)) {
        text += piece;
        patch(text);
      }
      patch(text.trim() || 'I could not produce an answer.', true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Something went wrong.';
      setMessages((m) => [...m.slice(0, -1), { role: 'assistant', content: `Error: ${msg}` }]);
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    void handleFile(e.dataTransfer.files[0]);
  }

  return (
    <div className="page">
      <header className="top">
        <div className="brand"><span className="logo">S</span> SoldrinAI <span className="muted">RAG Chatbot</span></div>
        <p className="privacy"><ShieldCheck size={15} /> Nothing is stored. The AI model runs privately on your device.</p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>1. Your document</h2>
          {!doc && !indexing && (
            <div
              className={`drop ${dragging ? 'over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInput.current?.click()}
              role="button" tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter') fileInput.current?.click(); }}
            >
              <UploadCloud size={30} />
              <strong>Drop a file here or click to browse</strong>
              <span className="muted">PDF, Word, TXT, Markdown, Excel or CSV · up to {mb(limits.maxUploadBytes)}</span>
              <input ref={fileInput} type="file" hidden accept={limits.extensions.map((x) => `.${x}`).join(',')}
                onChange={(e) => { void handleFile(e.target.files?.[0]); e.target.value = ''; }} />
            </div>
          )}

          {indexing && (
            <div className="progress-box">
              <div className="row"><LoaderCircle className="spin" size={16} /> {indexing.stage}…</div>
              <div className="bar"><i style={{ width: `${indexing.pct}%` }} /></div>
            </div>
          )}

          {fileError && <p className="error"><AlertCircle size={15} /> {fileError}</p>}

          {doc && index && (
            <div className="doc">
              <div className="row"><FileText size={18} /> <strong>{doc.name}</strong></div>
              <p className="muted">
                {index.chunks.length} {doc.kind === 'table' ? 'rows (one chunk per row)' : 'semantic chunks'} indexed
                {doc.truncated && <> · only the {doc.limits} were used</>}
              </p>
              <button className="ghost" onClick={clearDocument}><RotateCcw size={14} /> Use a different file</button>
            </div>
          )}

          <ul className="notes muted">
            <li>Tables: each row becomes one searchable chunk.</li>
            <li>Documents are split into topic-based chunks using a small sentence-transformer.</li>
            <li>Large files are trimmed to keep things fast; the limits are shown after indexing.</li>
            <li>Your text is sent to SoldrinAI only to compute embeddings, and is never stored.</li>
          </ul>
        </section>

        <section className="card chat">
          <h2>2. Chat</h2>

          {!gpu && (
            <p className="error"><AlertCircle size={15} /> This browser doesn't support WebGPU, which the on-device model needs.
              Try desktop Chrome or Edge.</p>
          )}

          {gpu && modelState !== 'ready' && (
            <div className="model-card">
              <div className="row"><Cpu size={18} /> <strong>{MODEL_LABEL}</strong> <span className="muted">· runs in your browser</span></div>
              {modelState === 'loading' ? (
                <>
                  <div className="bar"><i style={{ width: `${modelProgress.pct}%` }} /></div>
                  <p className="muted small">{modelProgress.text || 'Starting…'}</p>
                </>
              ) : modelCached ? (
                <>
                  <p className="muted">This model is already saved in your browser, so nothing needs to be downloaded.</p>
                  <button onClick={startModel}>Load model</button>
                </>
              ) : (
                <>
                  <p>
                    <Download size={14} /> Starting the chatbot downloads about <strong>{MODEL_DOWNLOAD_MB} MB</strong> once.
                    There is nothing to install; it is saved in your browser and runs locally. You can delete it any time below.
                  </p>
                  <button onClick={startModel}>Download model &amp; start</button>
                </>
              )}
              {modelState === 'error' && <p className="error"><AlertCircle size={15} /> {modelError}</p>}
            </div>
          )}

          <div className="messages">
            {messages.length === 0 && (
              <p className="muted empty">
                {doc ? 'Ask a question about your document.' : 'Upload a document to begin.'}
              </p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="bubble">{m.content || <LoaderCircle className="spin" size={14} />}</div>
                {m.sources && (
                  <details className="sources">
                    <summary>Sources ({m.sources.length})</summary>
                    {m.sources.map((s, n) => (
                      <p key={n}><b>[{n + 1}]</b> {s.text.slice(0, 300)}{s.text.length > 300 ? '…' : ''}</p>
                    ))}
                  </details>
                )}
              </div>
            ))}
            <div ref={endRef} />
          </div>

          <form className="ask" onSubmit={ask}>
            <input
              value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={500}
              placeholder={doc ? (modelState === 'ready' ? 'Ask about your document…' : 'Start the model to chat') : 'Upload a document first'}
              disabled={!doc || modelState !== 'ready' || busy}
            />
            <button disabled={!doc || modelState !== 'ready' || busy || !question.trim()} aria-label="Send"><Send size={16} /></button>
          </form>

          <div className="privacy-tools">
            <label className="check">
              <input type="checkbox" checked={removeOnClose} onChange={(e) => toggleRemoveOnClose(e.target.checked)} />
              Remove the downloaded model when I close this tab
            </label>
            <button className="ghost danger" onClick={removeModel} disabled={!modelCached && modelState !== 'ready'}>
              <Trash2 size={14} /> Delete model data now
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
