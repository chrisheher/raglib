"""
raglib_mcp.py
=============
MCP server that exposes the live raglib library as a tool. Pair it with
Kernel's MCP server in the same client (Claude Code, Cursor) and one agent
can browse the live web with Kernel and cross-check against your library.

search_library and library_topics call the deployed API (Railway, via
onlychr.is/raglib). find_passages reads the local chroma_db, so it only
works on a machine with the databases.

Usage (Claude Code):
  claude mcp add raglib -- /Applications/MAMP/htdocs/libby/venv/bin/python /Applications/MAMP/htdocs/libby/raglib_mcp.py
  kernel mcp install --target <your-client>     # adds Kernel's browser tools

Override the endpoint with RAGLIB_URL (e.g. http://localhost:5000 for local).
"""

import json
import os

import requests
from mcp.server.mcpserver import MCPServer

# MCP clients launch this from anywhere; the local modules use paths
# relative to the project (chroma_db, taste_labels.json, .env).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAGLIB_URL = os.getenv("RAGLIB_URL", "https://onlychr.is/raglib").rstrip("/")
_taste = None

mcp = MCPServer("raglib")


def query_raglib(query: str, genre: str = "", n_final: int = 8) -> dict:
    """POST to /api/query and fold its event stream into {answer, sources}."""
    body = {"query": query, "n_final": n_final}
    if genre:
        body["genre_filter"] = genre
    answer, sources = [], []
    with requests.post(f"{RAGLIB_URL}/api/query", json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                answer.append(event["data"])
            elif event["type"] == "sources":
                sources = [
                    {"title": s["metadata"].get("title") or s["metadata"].get("source"),
                     "genre": s["metadata"].get("genre"),
                     "excerpt": s["text"][:400]}
                    for s in event["data"]
                ]
            elif event["type"] == "error":
                raise RuntimeError(event["data"])
    return {"answer": "".join(answer), "sources": sources}


@mcp.tool()
def search_library(query: str, genre: str = "") -> dict:
    """Ask Chris Heher's GraphRAG library (10K+ docs: literature, consciousness
    studies, nautical, creative thinking). Returns a cited answer plus the
    source excerpts it drew on. Optional genre, e.g. 'poetry', 'nautical'."""
    return query_raglib(query, genre)


@mcp.tool()
def library_topics() -> dict:
    """List the genres and themes in the library, with document counts."""
    r = requests.get(f"{RAGLIB_URL}/api/stats", timeout=30)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def find_passages(text: str, k: int = 3) -> dict:
    """Match any text — a news story, a paragraph, a claim — to the passages in
    Chris Heher's literary library that most resemble it (literature, philosophy,
    mythology, consciousness studies; not the nautical manuals or AI papers).
    Returns up to k passages from different books, with similarity scores, plus
    the Leiden community (a thematic cluster of the library) closest to the text.
    Use it to find what an older book anticipated or illuminates about the text."""
    global _taste
    import discover
    import passage_match as pm
    if _taste is None:
        _taste = discover.load_taste()
    vec = pm.embed([text[:6000]])[0]
    return {"passages": pm.find_passages(vec, k=max(1, min(k, 8))),
            "cluster": pm.closest_clusters(vec, _taste)[0]}


if __name__ == "__main__":
    mcp.run()
