"""Turn an uploaded file into small "units" ready for embedding.

* Prose (PDF, DOCX, TXT/MD): split into sentences. The browser groups neighbouring
  sentences into semantic chunks using their embeddings.
* Tables (CSV, XLSX): one row = one unit (= one chunk), rendered as "Column: value; ...".

Every path is capped (size, pages, rows, columns, characters) so a visitor cannot push
unbounded data through the serverless function.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from fastapi import HTTPException

MAX_UPLOAD = 4 * 1024 * 1024    # Vercel request bodies are capped at ~4.5 MB
MAX_CHARS = 150_000             # prose characters kept
MAX_SENTENCES = 1_200           # prose units kept
MAX_PDF_PAGES = 60
MAX_ROWS = 500                  # table rows kept
MAX_COLUMNS = 40
MAX_ROW_CHARS = 600             # characters kept per table row
MAX_UNIT_CHARS = 400            # long sentences are split to this size

PROSE_EXT = {"pdf", "docx", "txt", "md"}
TABLE_EXT = {"csv", "xlsx"}
SUPPORTED = sorted(PROSE_EXT | TABLE_EXT)

_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _ext(name: str) -> str:
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return " ".join(str(v).split())


# ---------- prose ----------

def _pdf_text(content: bytes) -> tuple[str, bool]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise HTTPException(400, "This PDF is password protected.")
        pages = reader.pages[:MAX_PDF_PAGES]
        text = "\n\n".join((p.extract_text() or "") for p in pages)
        return text, len(reader.pages) > MAX_PDF_PAGES
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Could not read this PDF.") from exc


def _docx_text(content: bytes) -> str:
    from docx import Document
    try:
        doc = Document(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, "Could not read this Word file.") from exc
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells if c.text.strip()))
    return "\n\n".join(parts)


def split_sentences(text: str) -> list[str]:
    """Paragraph -> sentence split; over-long sentences are cut on word boundaries."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        for sent in _SENT.split(para):
            sent = sent.strip()
            while len(sent) > MAX_UNIT_CHARS:
                cut = sent.rfind(" ", 0, MAX_UNIT_CHARS)
                cut = cut if cut > 100 else MAX_UNIT_CHARS
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if len(sent) > 1:
                out.append(sent)
    return out


def _parse_prose(name: str, content: bytes) -> dict:
    ext = _ext(name)
    truncated = False
    if ext == "pdf":
        text, truncated = _pdf_text(content)
    elif ext == "docx":
        text = _docx_text(content)
    else:
        text = _decode(content)
    if len(text) > MAX_CHARS:
        text, truncated = text[:MAX_CHARS], True
    units = split_sentences(text)
    if len(units) > MAX_SENTENCES:
        units, truncated = units[:MAX_SENTENCES], True
    if not units:
        raise HTTPException(400, "No readable text found. Scanned (image-only) PDFs are not supported.")
    return {"kind": "prose", "name": name, "units": units, "truncated": truncated,
            "limits": f"first {MAX_CHARS // 1000}k characters"}


# ---------- tables ----------

def _read_table(name: str, content: bytes) -> list[list[Any]]:
    ext = _ext(name)
    if ext == "csv":
        return list(csv.reader(io.StringIO(_decode(content))))
    from openpyxl import load_workbook
    try:
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        return [list(r) for r in wb.active.iter_rows(values_only=True)]
    except Exception as exc:
        raise HTTPException(400, "Could not read this Excel file.") from exc


def _parse_table(name: str, content: bytes) -> dict:
    rows = _read_table(name, content)
    rows = [r for r in rows if any(_cell(c) for c in r)]
    if len(rows) < 2:
        raise HTTPException(400, "Need a header row and at least one data row.")
    header = [_cell(c) or f"Column {i + 1}" for i, c in enumerate(rows[0][:MAX_COLUMNS])]
    body = rows[1:]
    truncated = len(body) > MAX_ROWS or len(rows[0]) > MAX_COLUMNS
    units = []
    for r in body[:MAX_ROWS]:
        cells = [f"{h}: {_cell(v)}" for h, v in zip(header, r) if _cell(v)]
        if cells:
            units.append("; ".join(cells)[:MAX_ROW_CHARS])
    if not units:
        raise HTTPException(400, "No data rows found.")
    return {"kind": "table", "name": name, "units": units, "columns": header, "truncated": truncated,
            "limits": f"first {MAX_ROWS} rows and {MAX_COLUMNS} columns"}


def parse(name: str, content: bytes) -> dict:
    ext = _ext(name)
    if ext in TABLE_EXT:
        return _parse_table(name, content)
    if ext in PROSE_EXT:
        return _parse_prose(name, content)
    raise HTTPException(400, f"Unsupported file type. Upload one of: {', '.join(SUPPORTED)}.")
