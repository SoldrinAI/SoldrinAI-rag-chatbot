import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

// Runs the model off the main thread so the page stays responsive while it generates.
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg: MessageEvent) => handler.onmessage(msg);
