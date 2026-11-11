"""Embedding engine (no torch: ONNX Runtime keeps the serverless bundle small).

Model: sentence-transformers/all-MiniLM-L6-v2, int8-quantised ONNX (~23 MB, 384-d).
Retrieval (BM25 + cosine) runs in the browser; this module only produces vectors.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_NAME = "all-MiniLM-L6-v2 (ONNX int8)"
MODEL_DIR = Path(__file__).parent / "model"
MAX_TOKENS = 128
DIM = 384


@lru_cache(maxsize=1)
def _load():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(MAX_TOKENS)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    return tok, sess, {i.name for i in sess.get_inputs()}


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """Return L2-normalised sentence embeddings, shape (n, 384), float32."""
    tok, sess, names = _load()
    out = []
    for start in range(0, len(texts), batch):
        encs = tok.encode_batch(texts[start:start + batch])
        width = max(len(e.ids) for e in encs) or 1
        ids = np.zeros((len(encs), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = sess.run(None, {k: v for k, v in feed.items() if k in names})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
        out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None))
    return np.vstack(out).astype(np.float32) if out else np.zeros((0, DIM), np.float32)


def encode_vectors(v: np.ndarray) -> str:
    """float16 + base64 keeps responses ~4x smaller than JSON floats."""
    return base64.b64encode(v.astype(np.float16).tobytes()).decode()
