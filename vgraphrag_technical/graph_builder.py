"""
graph_builder.py (technical)
=============================
Stage 1: PDF corpus -> Rich Knowledge Graph (RKG)

Two phases:
  Phase A (ingest_pdfs_to_chunks) — reads every PDF from templates/pdfs/
    (domain="nautical") and templates/pdfs/ai/ (domain="ai_research"),
    extracts text (pdfplumber + per-page OCR fallback), cleans it with the
    same ocr_cleaner pipeline used for ChromaDB ingestion, chunks it, and
    writes each chunk to clean_text_technical/ as a flat .txt file — mirroring
    the clean_text/ convention so the retriever can glob-read chunks the
    same way. Idempotent per-PDF (skips a PDF whose chunks already exist).

  Phase B (build_graph) — reads every .txt chunk from clean_text_technical/,
    calls Claude to extract entities and relationships using a technical
    ontology (CONCEPT/METHOD/SYSTEM/COMPONENT/...), and merges them into a
    NetworkX DiGraph. Checkpointed to disk so it can be interrupted/resumed,
    structurally identical to vgraphrag/graph_builder.py's build_graph.

Generic graph-assembly logic (node merging, JSON repair, name
normalization, serialization) is imported from vgraphrag.graph_builder
rather than duplicated.
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
import networkx as nx
from dotenv import load_dotenv

from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER

# Repo root on sys.path so we can reuse the PDF extraction / OCR cleanup
# already built for ChromaDB ingestion (ingest_pdf.py, ocr_cleaner.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest_pdf import extract_pdf_text, unwrap_isolated_corrupt  # noqa: E402
from ocr_cleaner import run_pipeline  # noqa: E402

from vgraphrag.graph_builder import (  # noqa: E402
    _normalize_name,
    add_extraction_to_graph,
    save_graph,
    load_graph,
    _repair_truncated_json,
    looks_garbled,
    graph_stats,
)

load_dotenv()

# ─── source corpus ──────────────────────────────────────────────
NAUTICAL_DIR    = Path("templates/pdfs")
AI_RESEARCH_DIR = Path("templates/pdfs/ai")

# ─── paths ────────────────────────────────────────────────────
CHUNK_DIR       = Path("clean_text_technical")
DB_DIR          = Path("vgraphrag_technical_db")
GRAPH_PATH      = DB_DIR / "rkg.json"
PROGRESS_PATH   = DB_DIR / "extraction_progress.json"

# ─── corpus exclusions ─────────────────────────────────────────
# Large fully-scanned service manuals whose page-by-page OCR volume
# (270 and 204 pages) would dominate the chunk count relative to every
# other source PDF without adding proportionate graph value.
SKIP_PDFS = {
    "DETROIT_DIESEL_TUNE-UP_TROUBLESHOOTING_MTCE",
    "Detroit Diesel Engine Series 71 Service Manual",
}

# ─── model ────────────────────────────────────────────────────
# Cheaper than the literary pipeline's claude-opus-4-8 — structured
# technical extraction needs less interpretive nuance than literary
# analysis, and this corpus's per-chunk volume is comparable.
MODEL        = "claude-sonnet-4-6"
MAX_TOKENS   = 4096
RATE_PAUSE   = 0.6
BATCH_SAVE   = 25

# ─── chunking ─────────────────────────────────────────────────
# Larger than the literary per-file chunks since these are long single
# documents rather than pre-curated excerpts.
CHUNK_SIZE    = 4000
CHUNK_OVERLAP = 400
MIN_CHUNK_LEN = 200


# ─────────────────────────────────────────────────────────────
# PHASE A — PDF -> cleaned, chunked .txt files
# ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > MIN_CHUNK_LEN]


def _iter_source_pdfs():
    """Yield (pdf_path, domain) for every source PDF in the technical corpus."""
    for pdf_path in sorted(NAUTICAL_DIR.glob("*.pdf")):
        if pdf_path.stem in SKIP_PDFS:
            continue
        yield pdf_path, "nautical"
    for pdf_path in sorted(AI_RESEARCH_DIR.glob("*.pdf")):
        if pdf_path.stem in SKIP_PDFS:
            continue
        yield pdf_path, "ai_research"


def ingest_pdfs_to_chunks(chunk_dir: Path = CHUNK_DIR, force: bool = False) -> int:
    """
    Phase A: extract, clean, and chunk every source PDF into chunk_dir.
    Idempotent per-PDF — skips a PDF whose chunk files already exist unless
    force=True. Returns the number of PDFs (re)processed.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for pdf_path, domain in _iter_source_pdfs():
        stem = pdf_path.stem
        existing = list(chunk_dir.glob(f"{domain}__{stem}__chunk*.txt"))
        if existing and not force:
            continue

        print(f"[ingest] {domain}/{pdf_path.name}")
        try:
            raw_text = extract_pdf_text(pdf_path)
        except Exception as e:
            print(f"  ERROR extracting ({e}) — skipping")
            continue

        if not raw_text.strip():
            print("  no extractable text — skipping")
            continue

        clean_text, _footnotes, _log = run_pipeline(raw_text)
        clean_text = unwrap_isolated_corrupt(clean_text)

        chunks = _chunk_text(clean_text)
        if not chunks:
            print("  no chunks after cleaning — skipping")
            continue

        for f in existing:
            f.unlink()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{domain}__{stem}__chunk{i:03d}"
            (chunk_dir / f"{chunk_id}.txt").write_text(chunk, encoding="utf-8")

        print(f"  {len(chunks)} chunks written")
        processed += 1

    return processed


# ─────────────────────────────────────────────────────────────
# PHASE B — chunk .txt files -> RKG
# ─────────────────────────────────────────────────────────────

def parse_chunk_filename(filename: str) -> dict:
    """
    Chunk filenames look like: {domain}__{pdf_stem}__chunk{NNN}.txt
    """
    stem = Path(filename).stem
    parts = stem.split("__")
    domain   = parts[0] if len(parts) > 0 else "unknown"
    doc_stem = parts[1] if len(parts) > 1 else stem
    section  = parts[2] if len(parts) > 2 else ""
    return {
        "domain":    domain,
        "doc_title": doc_stem.replace("_", " ").strip(),
        "section":   section.replace("chunk", "chunk ").strip(),
    }


def _call_claude(client: anthropic.Anthropic, system: str, user: str,
                 retries: int = 3) -> Optional[str]:
    """Call Claude with exponential backoff on errors."""
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limit, waiting {wait}s...")
            import time
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            import time
            time.sleep(2 ** attempt)
    return None


def extract_entities_and_relations(
    client: anthropic.Anthropic,
    text: str,
    file_info: dict,
) -> dict:
    """
    Call Claude to extract entities and relationships from one chunk.
    Returns {"entities": [...], "relationships": [...]} or empty dicts on failure.
    """
    user_prompt = EXTRACTION_USER.format(
        doc_title=file_info["doc_title"],
        domain=file_info["domain"],
        section=file_info["section"],
        text=text[:6000],
    )

    raw = _call_claude(client, EXTRACTION_SYSTEM, user_prompt)
    if not raw:
        return {"entities": [], "relationships": []}

    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(raw)
        return {
            "entities":      parsed.get("entities", []),
            "relationships": parsed.get("relationships", []),
        }
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}, attempting repair...")
        repaired = _repair_truncated_json(raw)
        if repaired:
            return repaired
        return {"entities": [], "relationships": []}


def _load_progress() -> set:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def _save_progress(done: set) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done), f)


def build_graph(
    chunk_dir: Path = CHUNK_DIR,
    limit: Optional[int] = None,
    resume: bool = True,
) -> nx.DiGraph:
    """
    Build or resume building the technical RKG from clean_text_technical/.

    Args:
        chunk_dir: Path to folder containing PDF-derived chunk .txt files
        limit:     Process at most this many chunks (None = all)
        resume:    If True, skip chunks already processed (default)

    Returns:
        A NetworkX DiGraph — the technical Rich Knowledge Graph.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    G = load_graph(GRAPH_PATH) if (resume and GRAPH_PATH.exists()) else nx.DiGraph()
    done = _load_progress() if resume else set()

    files = sorted(chunk_dir.glob("*.txt"))
    if limit:
        files = files[:limit]

    pending = [f for f in files if f.name not in done]
    total   = len(files)
    print(f"Corpus: {total} chunks | Already done: {len(done)} | To process: {len(pending)}")

    WORKERS = 8
    lock    = threading.Lock()

    def process_chunk(filepath: Path):
        text      = filepath.read_text(encoding="utf-8", errors="ignore")
        file_info = parse_chunk_filename(filepath.name)

        if looks_garbled(text):
            return filepath, None, "skipped (garbled/short)"

        extraction = extract_entities_and_relations(client, text, file_info)
        return filepath, extraction, file_info

    counter = [len(done)]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_chunk, fp): fp for fp in pending}

        for future in as_completed(futures):
            filepath, extraction, info = future.result()

            with lock:
                counter[0] += 1
                label = f"[{counter[0]}/{total}] {filepath.name}"

                if extraction is None:
                    print(f"{label} — {info}")
                else:
                    n_ents = len(extraction["entities"])
                    n_rels = len(extraction["relationships"])
                    print(f"{label} — {n_ents} entities, {n_rels} relations")
                    add_extraction_to_graph(
                        G,
                        extraction,
                        chunk_id=filepath.stem,
                        work_title=info["doc_title"],
                    )

                done.add(filepath.name)

                if counter[0] % BATCH_SAVE == 0:
                    save_graph(G, GRAPH_PATH)
                    _save_progress(done)
                    print(f"  Checkpoint — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    save_graph(G, GRAPH_PATH)
    _save_progress(done)

    print(f"\n{'='*50}")
    print(f"Graph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Chunks processed: {len(done)}/{total}")
    return G
