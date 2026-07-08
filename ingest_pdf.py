"""
ingest_pdf.py
=============
Ingest a folder of local PDFs (scanned, text-based, or mixed) into the
literary_documents ChromaDB collection under an explicit, isolated genre —
following the same pattern used to fix the Deasy Labs / JAIT contamination:
a distinct genre tag and per-document source slugs, so new non-literary
content never gets force-fit into the literary taxonomy or bleeds into
unrelated retrieval.

Per PDF:
  1. Extract text page-by-page with pdfplumber (text-based PDFs).
  2. Any page whose extracted text is too short (likely a scanned image) is
     rasterized and re-extracted with Tesseract OCR instead.
  3. The concatenated text is run through ocr_cleaner.py's cleanup pipeline
     (ligatures, hyphenation, page headers/footers, OCR artifact fixes).
  4. Cleaned text is chunked (~800 chars, 120 overlap) and embedded with
     text-embedding-3-large (3072-dim), then inserted into 'literary_documents'
     with source=f"{collection_slug}/{pdf-stem}__chunkN", genre=<required>,
     title=<pdf filename>, themes=<optional>.

Usage:
  python ingest_pdf.py --dir pdf_inbox --genre technical_reference --themes "engineering,reference"
  python ingest_pdf.py --dir pdf_inbox --genre technical_reference --collection-slug mymanuals
  python ingest_pdf.py --file some.pdf --genre technical_reference --dry-run
"""

import argparse
import io
import os
import sys
from pathlib import Path

import chromadb
import pdfplumber
import pytesseract
from openai import OpenAI
from pdf2image import convert_from_path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from ocr_cleaner import run_pipeline  # noqa: E402

load_dotenv()

CHROMA_PATH       = "chroma_db"
COLLECTION_NAME   = "literary_documents"
EMBEDDING_MODEL   = "text-embedding-3-large"
EMBEDDING_DIMS    = 3072
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 120
BATCH_SIZE        = 20
MIN_TEXT_LEN_OK   = 40  # below this, treat page as scanned image and OCR it


def extract_pdf_text(pdf_path: Path) -> str:
    """Text-first extraction with per-page OCR fallback for scanned pages."""
    pages_text = []
    ocr_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_TEXT_LEN_OK:
                pages_text.append(None)  # mark for OCR
                ocr_pages.append(i)
            else:
                pages_text.append(text)

    if ocr_pages:
        print(f"    {len(ocr_pages)}/{n_pages} pages need OCR (scanned/image content)")
        images = convert_from_path(str(pdf_path), dpi=300)
        for i in ocr_pages:
            if i < len(images):
                pages_text[i] = pytesseract.image_to_string(images[i]).strip()

    return "\n\n".join(t for t in pages_text if t)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 80]


def ingest_pdf(pdf_path: Path, collection, client, genre: str, themes: str,
               collection_slug: str, dry_run: bool):
    stem = pdf_path.stem
    print(f"\n{pdf_path.name}")

    if pdf_path.stat().st_size == 0:
        print("    0-byte file — skipping")
        return 0

    try:
        raw_text = extract_pdf_text(pdf_path)
    except Exception as e:
        print(f"    ERROR extracting ({e}) — skipping")
        return 0

    if not raw_text.strip():
        print("    no extractable text — skipping")
        return 0

    clean_text, footnotes, change_log = run_pipeline(raw_text)
    print(f"    extracted {len(raw_text)} chars -> cleaned {len(clean_text)} chars")

    chunks = chunk_text(clean_text)
    source_label = f"{collection_slug}/{stem}"
    print(f"    {len(chunks)} chunks -> source={source_label}")

    if dry_run:
        print("    [dry-run] not writing to ChromaDB")
        if chunks:
            print(f"    sample chunk[0]: {chunks[0][:200]!r}")
        return len(chunks)

    added = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"{collection_slug}__{stem}__chunk{i + j}" for j in range(len(batch))]

        existing = set(collection.get(ids=ids)["ids"])
        new_idx = [j for j, id_ in enumerate(ids) if id_ not in existing]
        if not new_idx:
            continue

        new_texts = [batch[j] for j in new_idx]
        new_ids = [ids[j] for j in new_idx]
        try:
            resp = client.embeddings.create(input=new_texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS)
        except Exception as e:
            print(f"    ERROR embedding batch ({e}) — skipping batch")
            continue
        embeddings = [d.embedding for d in resp.data]

        metas = [{
            "source": source_label,
            "title": pdf_path.stem.replace("_", " "),
            "genre": genre,
            "themes": themes,
            "tones": "",
        } for _ in new_texts]

        collection.add(documents=new_texts, embeddings=embeddings, metadatas=metas, ids=new_ids)
        added += len(new_ids)

    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory of PDFs to ingest")
    parser.add_argument("--file", help="Single PDF file to ingest")
    parser.add_argument("--genre", required=True, help="Genre tag to isolate this content (required — do not reuse literary genres)")
    parser.add_argument("--themes", default="", help="Comma-separated themes, e.g. 'engineering,reference'")
    parser.add_argument("--collection-slug", default=None, help="Prefix for source labels (default: derived from --genre)")
    parser.add_argument("--dry-run", action="store_true", help="Extract/chunk only, don't write to ChromaDB")
    args = parser.parse_args()

    if not args.dir and not args.file:
        parser.error("provide --dir or --file")

    pdf_paths = []
    if args.file:
        pdf_paths.append(Path(args.file))
    if args.dir:
        pdf_paths.extend(sorted(Path(args.dir).glob("*.pdf")))

    if not pdf_paths:
        print("No PDFs found.")
        return

    collection_slug = args.collection_slug or args.genre

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for p in pdf_paths:
        try:
            total += ingest_pdf(p, collection, client, args.genre, args.themes, collection_slug, args.dry_run)
        except Exception as e:
            print(f"    ERROR processing {p.name} ({e}) — skipping file")

    print(f"\nDone — {total} chunks {'would be ' if args.dry_run else ''}added.")
    if not args.dry_run:
        print(f"Collection now has {collection.count()} total documents")


if __name__ == "__main__":
    main()
