"""
ingest_url.py
=============
Scrape one or more URLs, chunk the text, embed with the same OpenAI model
used by the main corpus, and insert into the existing ChromaDB collection.

Usage:
  python ingest_url.py https://www.deasylabs.com/
  python ingest_url.py https://www.deasylabs.com/ https://www.deasylabs.com/platform

The script crawls each seed URL and follows internal links up to --depth levels.
"""

import argparse
import hashlib
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "literary_documents"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS  = 3072
CHUNK_SIZE      = 800   # characters per chunk
CHUNK_OVERLAP   = 120
BATCH_SIZE      = 20
REQUEST_DELAY   = 0.5   # seconds between requests

client        = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def fetch_text(url: str) -> tuple[str, str]:
    """Return (title, cleaned_body_text) for a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; raglib-ingest/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type and content_type != "text/html":
        raise ValueError(f"unsupported content-type '{content_type}' (not HTML) — skipping")

    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.string.strip() if soup.title else urlparse(url).path

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return title, text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 80]


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS)
    return [d.embedding for d in resp.data]


def doc_id(url: str, chunk_idx: int) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"web__{h}__chunk{chunk_idx}"


def page_slug(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.replace("/", "-") if path else "home"


def ingest_url(url: str, source_label: str) -> int:
    print(f"  Fetching {url}")
    try:
        title, text = fetch_text(url)
    except Exception as e:
        print(f"  ✗ {e}")
        return 0

    source_label = f"{source_label}/{page_slug(url)}"

    chunks = chunk_text(text)
    if not chunks:
        print(f"  ✗ no usable text")
        return 0

    added = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids   = [doc_id(url, i + j) for j in range(len(batch))]

        # Skip chunks already in the collection
        existing = set(collection.get(ids=ids)["ids"])
        new_idx  = [j for j, id_ in enumerate(ids) if id_ not in existing]
        if not new_idx:
            continue

        new_texts = [batch[j] for j in new_idx]
        new_ids   = [ids[j]   for j in new_idx]
        embeddings = embed_batch(new_texts)

        metas = [{
            "source":  source_label,
            "url":     url,
            "title":   title,
            "genre":   "reference",
            "themes":  "metadata,ai,retrieval",
            "tones":   "",
        } for _ in new_texts]

        collection.add(documents=new_texts, embeddings=embeddings, metadatas=metas, ids=new_ids)
        added += len(new_ids)

    print(f"  ✓ {title} — {added} chunks added ({len(chunks)} total)")
    return added


def crawl(seed: str, depth: int = 1) -> list[str]:
    """Return all internal URLs reachable from seed within depth hops."""
    base = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"
    visited, queue = set(), [(seed, 0)]
    urls = []
    while queue:
        url, d = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        urls.append(url)
        if d >= depth:
            continue
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; raglib-ingest/1.0)"}
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0].rstrip("/")
                if href.startswith(base) and href not in visited:
                    queue.append((href, d + 1))
            time.sleep(REQUEST_DELAY)
        except Exception:
            pass
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--depth", type=int, default=1,
                        help="How many link-hops to follow (default 1)")
    parser.add_argument("--label", default="",
                        help="Source label stored in metadata (default: domain name)")
    args = parser.parse_args()

    total = 0
    for seed in args.urls:
        label = args.label or urlparse(seed).netloc
        print(f"\nCrawling {seed} (depth={args.depth}, label={label})")
        urls = crawl(seed, depth=args.depth)
        print(f"  Found {len(urls)} pages")
        for url in urls:
            total += ingest_url(url, label)
            time.sleep(REQUEST_DELAY)

    print(f"\n✓ Done — {total} new chunks added to '{COLLECTION_NAME}'")
    print(f"  Collection now has {collection.count()} total documents")


if __name__ == "__main__":
    main()
