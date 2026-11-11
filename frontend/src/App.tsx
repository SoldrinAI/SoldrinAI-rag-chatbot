import { useEffect, useRef, useState, type DragEvent, type FormEvent } from 'react';
import {
  AlertCircle, Check, Cpu, Download, FileText, LoaderCircle, RotateCcw, Send, ShieldCheck, Trash2, UploadCloud, X,
} from 'lucide-react';
import { DEFAULT_LIMITS, embedTexts, fetchLimits, parseFile, type Limits, type Parsed } from './lib/api';
import { rowChunks, semanticChunks } from './lib/chunking';
import { buildIndex, hybridSearch, type Index } from './lib/search';
import {
  DEFAULT_MODEL_ID, MODELS, TOP_K, modelById, sizeLabel, removeOnCloseEnabled, setRemoveOnClose, webGpuAvailable, type Turn,
} from './lib/llm-config';

// The model engine is large, so it is only fetched when the visitor actually needs it.
const engine = () => import('./lib/llm');

interface Source { text: string; score: number }
interface Message extends Turn { sources?: Source[] }
type ModelState = 'idle' | 'loading' | 'ready' | 'error';

const mb = (b: number) => `${(b / 1024 / 1024).toFixed(0)} MB`;

function Brand() {
  return (
    <div className="brand">
      <svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
        <path d="M16 2.5l11.7 6.75v13.5L16 29.5 4.3 22.75V9.25z" />
        <path d="M16 2.5v27M4.3 9.25l23.4 13.5M27.7 9.25L4.3 22.75" opacity=".55" />
        <circle cx="16" cy="16" r="3.2" fill="currentColor" stroke="none" />
        <g fill="#fff"><circle cx="16" cy="2.5" r="2" /><circle cx="27.7" cy="9.25" r="2" /><circle cx="27.7" cy="22.75" r="2" /><circle cx="16" cy="29.5" r="2" /><circle cx="4.3" cy="22.75" r="2" /><circle cx="4.3" cy="9.25" r="2" /></g>
      </svg>
      <span>Soldrin<b>AI</b> <em>RAG Chatbot</em></span>
    </div>
  );
}

export default function App() {
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [doc, setDoc] = useState<Parsed | null>(null);
  const [index, setIndex] = useState<Index | null>(null);
  const [indexing, setIndexing] = useState<{ stage: string; pct: number } | null>(null);
  const [fileError, setFileError] = useState('');
  const [dragging, setDragging] = useState(false);

  const [modelState, setModelState] = useState<ModelState>('idle');
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const [cachedIds, setCachedIds] = useState<string[]>([]);
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
      setCachedIds(await m.cachedModelIds());
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
      const m = await engine();
      await m.loadModel(modelId, (fraction, text) => setModelProgress({ pct: Math.round(fraction * 100), text }));
      setModelState('ready');
      setCachedIds(await m.cachedModelIds());
    } catch (e) {
      setModelError(e instanceof Error ? e.message : 'The model could not be loaded.');
      setModelState('error');
    }
  }

  async function removeModel() {
    await (await engine()).deleteModelData();
    setModelState('idle');
    setCachedIds([]);
    setModelProgress({ pct: 0, text: '' });
  }

  async function changeModel() {
    await (await engine()).unloadModel();
    setModelState('idle');
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

  const chosen = modelById(modelId);
  const step = !doc ? 0 : modelState !== 'ready' ? 1 : 2;
  const steps = ['Upload a document', 'Start the model', 'Chat'];

  return (
    <div className="shell">
      <header><Brand /><span className="status"><i /> Private session</span></header>
      <main>
        <div className="hero">
          <h1>Chat with your documents, <em>privately</em>.</h1>
          <p>Upload a file and ask questions in plain words. SoldrinAI finds the right passages, and a small AI model answers right in your browser.</p>
        </div>

        <ol className="stepper">
          {steps.map((label, i) => (
            <li key={label} className={i === step ? 'on' : i < step ? 'done' : ''}>
              <span>{i < step ? <Check size={12} /> : i + 1}</span>{label}
            </li>
          ))}
        </ol>

        <div className="privacy" role="note">
          <ShieldCheck size={20} />
          <div>
            <strong>Your files are never stored.</strong>
            <p>Text is used only to compute embeddings, then discarded. Answers are generated by a model running on your device.</p>
          </div>
        </div>

        <div className="split">
          <section className="card">
            <h2>Your document</h2>
            {!doc && !indexing && (
              <div
                className={`drop ${dragging ? 'drag' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInput.current?.click()}
                role="button" tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter') fileInput.current?.click(); }}
              >
                <UploadCloud size={30} />
                <span><b>Drop a file here</b> or click to browse</span>
                <span>PDF, Word, TXT, Markdown, Excel or CSV · up to {mb(limits.maxUploadBytes)}</span>
                <input ref={fileInput} type="file" hidden accept={limits.extensions.map((x) => `.${x}`).join(',')}
                  onChange={(e) => { void handleFile(e.target.files?.[0]); e.target.value = ''; }} />
              </div>
            )}

            {indexing && (
              <div className="stack">
                <span className="row"><LoaderCircle className="spin" size={16} /> {indexing.stage}…</span>
                <div className="bar"><i style={{ width: `${indexing.pct}%` }} /></div>
              </div>
            )}

            {fileError && <div className="err"><AlertCircle size={16} /><span>{fileError}</span><button onClick={() => setFileError('')} aria-label="Dismiss"><X size={14} /></button></div>}

            {doc && index && (
              <div className="stack">
                <div className="file"><FileText size={18} /><span>{doc.name}</span>
                  <button onClick={clearDocument}><RotateCcw size={14} /> Replace</button></div>
                <p className="muted">
                  {index.chunks.length} {doc.kind === 'table' ? 'rows (one chunk per row)' : 'semantic chunks'} indexed
                  {doc.truncated && <> · only the {doc.limits} were used</>}
                </p>
              </div>
            )}

            <ul className="notes muted">
              <li>Tables: each row becomes one searchable chunk.</li>
              <li>Documents are split into topic-based chunks by a small sentence-transformer.</li>
              <li>Large files are trimmed to keep things fast.</li>
            </ul>
          </section>

          <section className="card chat">
            <h2>Chat</h2>

            {!gpu && (
              <div className="err"><AlertCircle size={16} /><span>This browser doesn't support WebGPU, which the on-device model needs. Try desktop Chrome or Edge.</span></div>
            )}

            {gpu && modelState !== 'ready' && (
              <div className="model-card">
                <label className="field">
                  <span className="row"><Cpu size={16} /> <strong>Choose a model</strong> <span className="muted">· runs in your browser</span></span>
                  <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={modelState === 'loading'}>
                    {MODELS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label} · {sizeLabel(m.downloadMB)}{cachedIds.includes(m.id) ? ' · downloaded' : ''}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="muted small">{chosen.note} Needs about {sizeLabel(chosen.memoryMB)} of GPU memory.</p>
                <div className="warn">
                  <AlertCircle size={16} />
                  <span>Answer quality depends on the model. Smaller models are faster to load but may give wrong, incomplete or
                    garbled answers. If a result looks off, try a larger model.</span>
                </div>
                {modelState === 'loading' ? (
                  <>
                    <div className="bar"><i style={{ width: `${modelProgress.pct}%` }} /></div>
                    <p className="muted small">{modelProgress.text || 'Starting…'}</p>
                  </>
                ) : cachedIds.includes(modelId) ? (
                  <>
                    <p className="muted">This model is already saved in your browser, so nothing needs to be downloaded.</p>
                    <button className="btn" onClick={startModel}>Load {chosen.label}</button>
                  </>
                ) : (
                  <>
                    <p className="muted"><Download size={14} /> Starting downloads about <strong>{sizeLabel(chosen.downloadMB)}</strong> once.
                      Nothing to install: it is saved in your browser and runs locally. You can delete it any time below.</p>
                    <button className="btn" onClick={startModel}>Download {chosen.label} &amp; start</button>
                  </>
                )}
                {modelState === 'error' && <div className="err"><AlertCircle size={16} /><span>{modelError}</span></div>}
              </div>
            )}

            {gpu && modelState === 'ready' && (
              <div className="active-model">
                <span><Cpu size={14} /> {chosen.label} <span className="muted">· quality depends on the model</span></span>
                <button className="link" onClick={changeModel}><RotateCcw size={13} /> Change model</button>
              </div>
            )}

            <div className="messages">
              {messages.length === 0 && (
                <p className="muted empty">{doc ? 'Ask a question about your document.' : 'Upload a document to begin.'}</p>
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

            <form className="searchbar" onSubmit={ask}>
              <input
                value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={500}
                placeholder={doc ? (modelState === 'ready' ? 'Ask about your document…' : 'Start the model to chat') : 'Upload a document first'}
                disabled={!doc || modelState !== 'ready' || busy}
              />
              <button className="btn" disabled={!doc || modelState !== 'ready' || busy || !question.trim()} aria-label="Send"><Send size={16} /></button>
            </form>

            <div className="tools">
              <label className="check">
                <input type="checkbox" checked={removeOnClose} onChange={(e) => toggleRemoveOnClose(e.target.checked)} />
                Remove the downloaded model when I close this tab
              </label>
              <button className="link danger" onClick={removeModel} disabled={cachedIds.length === 0 && modelState !== 'ready'}>
                <Trash2 size={14} /> Delete model data now
              </button>
            </div>
          </section>
        </div>
      </main>
      <footer>
        <ShieldCheck size={14} /> <span>Your uploaded files are never stored. © SoldrinAI</span>
      </footer>
    </div>
  );
}
