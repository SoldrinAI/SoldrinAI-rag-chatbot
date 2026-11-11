import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import files  # noqa: E402
from index import app  # noqa: E402

client = TestClient(app)
CSV = b"Name,Role,Notes\nAna,Veterinarian,Treats dogs and cats\nRaj,Mechanic,Repairs car engines\nLena,Chef,Cooks Italian pasta\n"
TEXT = (b"Dogs need daily walks. They also need fresh water.\n\n"
        b"Invoices are due in 30 days. Late fees apply after that date.")


def _tiny_pdf(text: str) -> bytes:
    """Smallest valid one-page PDF with a line of text (no extra libraries needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offsets:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return bytes(out)


def test_csv_one_row_per_unit():
    out = files.parse("x.csv", CSV)
    assert out["kind"] == "table" and len(out["units"]) == 3
    assert out["units"][0] == "Name: Ana; Role: Veterinarian; Notes: Treats dogs and cats"


def test_xlsx_one_row_per_unit():
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Item", "Price"])
    ws.append(["Apple", 3])
    ws.append(["Pear", 4.5])
    buf = io.BytesIO()
    wb.save(buf)
    out = files.parse("prices.xlsx", buf.getvalue())
    assert out["kind"] == "table" and out["units"] == ["Item: Apple; Price: 3", "Item: Pear; Price: 4.5"]


def test_txt_is_split_into_sentences():
    out = files.parse("notes.txt", TEXT)
    assert out["kind"] == "prose" and len(out["units"]) == 4
    assert out["units"][2].startswith("Invoices")


def test_docx_and_pdf():
    from docx import Document
    d = Document()
    d.add_paragraph("Refunds take five days. Contact support for help.")
    buf = io.BytesIO()
    d.save(buf)
    assert len(files.parse("a.docx", buf.getvalue())["units"]) == 2

    assert "Shipping" in files.parse("a.pdf", _tiny_pdf("Shipping is free over fifty dollars."))["units"][0]


def test_row_and_size_caps():
    big_rows = "id,v\n" + "\n".join(f"{i},value {i}" for i in range(files.MAX_ROWS + 50))
    out = files.parse("big.csv", big_rows.encode())
    assert len(out["units"]) == files.MAX_ROWS and out["truncated"] is True

    long_text = ("Word " * 60 + ". ") * 2000
    out = files.parse("long.txt", long_text.encode())
    assert len(out["units"]) <= files.MAX_SENTENCES and out["truncated"] is True
    assert all(len(u) <= files.MAX_UNIT_CHARS for u in out["units"])


def test_rejects_bad_files():
    with pytest.raises(Exception):
        files.parse("x.exe", b"hi")
    with pytest.raises(Exception):
        files.parse("x.csv", b"Name,Role\n")
    with pytest.raises(Exception):
        files.parse("empty.txt", b"   ")


def test_upload_limit_enforced_by_api():
    big = b"a" * (files.MAX_UPLOAD + 10)
    r = client.post("/api/parse", files={"file": ("big.txt", big)})
    assert r.status_code == 413


def test_embeddings_are_normalised_384d():
    v = engine.embed(["hello world", "another sentence"])
    assert v.shape == (2, 384)
    assert abs(float((v[0] ** 2).sum()) - 1.0) < 1e-3


def test_embed_endpoint_and_limits():
    ok = client.post("/api/embed", json={"texts": ["a dog", "a cat"]})
    assert ok.status_code == 200 and ok.json()["count"] == 2
    assert client.post("/api/embed", json={"texts": ["x"] * 129}).status_code == 422
    assert client.get("/api/limits").json()["maxUploadBytes"] == files.MAX_UPLOAD
