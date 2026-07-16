"""
retriever.py (technical)
=========================
Stage 3: Operator Configuration — technical graph

Same operator vocabulary and pipeline shape as vgraphrag/retriever.py.
The pure graph-algorithm operators (Entity.PPR, Rel.Aggregator,
Chunk.Aggregator, Rel.Onehop) are imported directly since they take the
graph/scores as arguments and have no path dependencies. Only the operators
that touch ChromaDB collections or the on-disk chunk corpus are
reimplemented here, pointed at the technical DB/corpus.

SPECIFIC QA pipeline:
  Entity.Link    → extract entity mentions from query, match to graph nodes
  Entity.PPR     → Personalized PageRank seeded at linked entities
  Rel.Aggregator → edge_score = ppr[src] + ppr[tgt]
  Chunk.Aggregator → chunk_score = sum(edge_scores for edges from that chunk)

ABSTRACT QA pipeline:
  Community.VDB  → top-k communities by vector similarity to query
  Node.VDB       → fallback: top-k entity nodes by vector similarity
  Rel.Onehop     → all edges incident to VDB-retrieved nodes
"""

import json
import re
from pathlib import Path

import anthropic
import networkx as nx

from vgraphrag.retriever import (
    op_entity_ppr,
    op_relationship_aggregator,
    op_chunk_aggregator,
    op_relationship_onehop,
)
from vgraphrag.graph_builder import _normalize_name

from .index_builder import (
    embed_texts,
    get_node_collection,
    get_relationship_collection,
    get_community_collection,
)
from .prompts import LINKER_SYSTEM, LINKER_USER

from dotenv import load_dotenv
load_dotenv()

MODEL = "claude-sonnet-4-6"


def _embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# ─────────────────────────────────────────────────────────────
# OPERATOR 1 — Entity.Link
# ─────────────────────────────────────────────────────────────

def op_entity_link(
    query: str,
    G: nx.DiGraph,
    anthropic_client: anthropic.Anthropic,
) -> list[str]:
    user_prompt = LINKER_USER.format(query=query)
    try:
        msg = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=LINKER_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        mentions = json.loads(raw)
        if not isinstance(mentions, list):
            mentions = []
    except Exception:
        mentions = []

    query_lower = query.lower()
    direct_hits = [
        key for key, d in G.nodes(data=True)
        if len(key) > 4
        and (key in query_lower
             or d.get("name", "").lower() in query_lower)
    ]

    matched_keys = set(direct_hits)
    node_names = {
        _normalize_name(d.get("name", k)): k
        for k, d in G.nodes(data=True)
    }

    for mention in mentions:
        norm = _normalize_name(mention)
        if norm in node_names:
            matched_keys.add(node_names[norm])
        else:
            for nk, node_key in node_names.items():
                if norm in nk or nk in norm:
                    matched_keys.add(node_key)
                    break

    return list(matched_keys)


# ─────────────────────────────────────────────────────────────
# OPERATOR 5 — Community.VDB
# ─────────────────────────────────────────────────────────────

def op_community_vdb(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    col = get_community_collection()
    if col.count() == 0:
        return []

    q_emb = _embed_query(query)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "community_id": meta.get("community_id"),
            "summary":      doc,
            "size":         meta.get("size", 0),
            "works":        json.loads(meta.get("works", "[]")),
            "node_keys":    json.loads(meta.get("node_keys", "[]")),
            "score":        1 - dist,
            "operator":     "Community.VDB",
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ─────────────────────────────────────────────────────────────
# OPERATOR 6 — Node.VDB
# ─────────────────────────────────────────────────────────────

def op_node_vdb(
    query: str,
    top_k: int = 15,
) -> list[dict]:
    col = get_node_collection()
    if col.count() == 0:
        return []

    q_emb = _embed_query(query)
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "node_key":   rid,
            "name":       meta.get("name", rid),
            "type":       meta.get("type", "CONCEPT"),
            "description": meta.get("description", ""),
            "source_chunks": json.loads(meta.get("source_chunks", "[]")),
            "score":      1 - dist,
            "operator":   "Node.VDB",
        }
        for rid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ─────────────────────────────────────────────────────────────
# CHUNK FETCHER — reads text from clean_text_technical/
# ─────────────────────────────────────────────────────────────

CLEAN_TEXT_DIR = Path("clean_text_technical")

def fetch_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    texts = {}
    for cid in chunk_ids:
        candidates = list(CLEAN_TEXT_DIR.glob(f"{cid}.txt"))
        if candidates:
            texts[cid] = candidates[0].read_text(encoding="utf-8", errors="ignore").strip()
    return texts


# ─────────────────────────────────────────────────────────────
# FULL SPECIFIC QA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_specific_pipeline(
    query: str,
    G: nx.DiGraph,
    anthropic_client: anthropic.Anthropic,
    n_chunks: int = 6,
    n_rels: int = 50,
) -> dict:
    seed_keys = op_entity_link(query, G, anthropic_client)
    print(f"  Entity.Link: {len(seed_keys)} seeds: {seed_keys[:5]}")

    ppr = op_entity_ppr(seed_keys, G)

    top_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Entity.PPR: top nodes = {[(G.nodes[k].get('name', k), f'{s:.4f}') for k, s in top_ppr[:5]]}")

    relationships = op_relationship_aggregator(ppr, G, top_k=n_rels)
    print(f"  Rel.Aggregator: {len(relationships)} scored relationships")

    chunk_results = op_chunk_aggregator(relationships, top_k=n_chunks * 2)
    print(f"  Chunk.Aggregator: {len(chunk_results)} scored chunks")

    chunk_ids  = [c["chunk_id"] for c in chunk_results[:n_chunks]]
    chunk_texts = fetch_chunk_texts(chunk_ids)

    chunks = []
    for c in chunk_results:
        cid  = c["chunk_id"]
        text = chunk_texts.get(cid, "")
        if not text:
            continue
        chunks.append({
            "chunk_id": cid,
            "text":     text,
            "score":    c["score"],
            "rels":     c["rels"],
            "operator": "Chunk.Aggregator",
        })
        if len(chunks) >= n_chunks:
            break

    return {
        "mode":          "specific",
        "chunks":        chunks,
        "relationships": relationships[:20],
        "ppr_scores":    dict(top_ppr),
        "seed_entities": seed_keys,
    }


# ─────────────────────────────────────────────────────────────
# FULL ABSTRACT QA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_abstract_pipeline(
    query: str,
    G: nx.DiGraph,
    n_communities: int = 4,
    n_nodes: int = 10,
    n_chunks: int = 6,
) -> dict:
    communities = op_community_vdb(query, top_k=n_communities)
    print(f"  Community.VDB: {len(communities)} communities retrieved")

    vdb_nodes = op_node_vdb(query, top_k=n_nodes)
    print(f"  Node.VDB: {len(vdb_nodes)} nodes retrieved")

    node_keys = [n["node_key"] for n in vdb_nodes]
    for comm in communities:
        node_keys.extend(comm.get("node_keys", []))
    node_keys = list(dict.fromkeys(node_keys))

    relationships = op_relationship_onehop(node_keys, G, top_k=40)
    print(f"  Rel.Onehop: {len(relationships)} relationships")

    chunk_ids: list[str] = []
    seen_cids: set = set()

    for rel in relationships:
        cid = rel.get("source_chunk", "")
        if cid and cid not in seen_cids:
            chunk_ids.append(cid)
            seen_cids.add(cid)

    for node in vdb_nodes:
        for cid in node.get("source_chunks", []):
            if cid not in seen_cids:
                chunk_ids.append(cid)
                seen_cids.add(cid)

    rel_chunk_set = {rel.get("source_chunk") for rel in relationships}
    chunk_ids_scored = sorted(
        chunk_ids,
        key=lambda cid: (cid in rel_chunk_set, sum(
            r["score"] for r in relationships if r.get("source_chunk") == cid
        )),
        reverse=True,
    )

    chunk_texts = fetch_chunk_texts(chunk_ids_scored[:n_chunks * 2])

    chunks = []
    for cid in chunk_ids_scored:
        text = chunk_texts.get(cid, "")
        if not text:
            continue
        chunks.append({
            "chunk_id": cid,
            "text":     text,
            "score":    sum(r["score"] for r in relationships if r.get("source_chunk") == cid),
            "operator": "Rel.Onehop → Chunk",
        })
        if len(chunks) >= n_chunks:
            break

    return {
        "mode":          "abstract",
        "communities":   communities,
        "chunks":        chunks,
        "relationships": relationships[:20],
        "vdb_nodes":     vdb_nodes[:10],
    }
