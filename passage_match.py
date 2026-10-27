"""
passage_match.py
================
Match any piece of text (a news story, a paragraph) to the library passages
and Leiden communities it most resembles. Shared by news_oracle.py and the
raglib MCP server. Reads the local chroma_db.
"""

import os
import re

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS  = 3072

# The literary side of the corpus — leaves out nautical manuals, AI papers,
# code and scraped reference pages, which would otherwise win on surface topic.
LITERARY_GENRES = ["prose", "consciousness_studies", "postmodern_fiction",
                   "creative_thinking", "mythology", "poetry", "drama"]

_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_chroma = chromadb.PersistentClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
_docs = _chroma.get_collection("literary_documents")


def embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 50):
        resp = _oai.embeddings.create(input=texts[i:i + 50], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS)
        vecs += [d.embedding for d in resp.data]
    m = np.asarray(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def work_label(meta: dict) -> tuple[str, str]:
    """(title, author) from chunk metadata; sources look like 'Title__Author_slug.txt'."""
    src = meta.get("source", "")
    title, _, rest = src.partition("__")
    author = rest.split("_")[0] if rest else ""
    title = meta.get("title") or re.sub(r"\s+\d{3}\b.*|\.txt$", "", title) or src
    if not rest and "_" in title:                    # 'Harold Pinter_contra' → 'Harold Pinter'
        title = title.split("_")[0]
    return title.strip(), (meta.get("authors") or author).strip()


def excerpt(text: str, n: int = 420) -> str:
    """Trim a chunk to whole sentences, ~n characters."""
    text = " ".join(text.split())
    start = re.search(r"(?<=[.!?”\"])\s+[A-Z“\"]", text[:200])
    text = text[start.end() - 1:] if start else text
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("” "))
    return cut[:end + 1] if end > n * 0.5 else cut.rsplit(" ", 1)[0] + "…"


def find_passages(vec: np.ndarray, k: int = 3, genres: list[str] | None = LITERARY_GENRES) -> list[dict]:
    """Top-k passages from k distinct works for an embedded query."""
    where = {"genre": {"$in": genres}} if genres else {"genre": {"$ne": "reference"}}
    r = _docs.query(query_embeddings=[vec.tolist()], n_results=k * 6, where=where,
                    include=["documents", "metadatas", "distances"])
    out, seen = [], set()
    for id_, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
        title, author = work_label(meta)
        if title in seen:
            continue
        seen.add(title)
        out.append({"id": id_, "title": title, "author": author, "genre": meta.get("genre"),
                    "similarity": round(1 - dist, 4), "passage": excerpt(doc)})
        if len(out) == k:
            break
    return out


def closest_clusters(vec: np.ndarray, taste: list[dict], k: int = 1) -> list[dict]:
    """Nearest Leiden communities, using discover.load_taste() vectors."""
    sims = np.stack([t["vec"] for t in taste]) @ vec
    return [{"id": taste[i]["id"], "title": taste[i]["title"], "works": taste[i]["works"],
             "similarity": round(float(sims[i]), 4)}
            for i in np.argsort(-sims)[:k]]
