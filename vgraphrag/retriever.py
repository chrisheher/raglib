"""
retriever.py
============
Stage 3: Operator Configuration

Implements the VGraphRAG operator vocabulary for two pipelines:

SPECIFIC QA pipeline (character, plot, intertextual fact):
  Entity.Link    → extract entity mentions from query, match to graph nodes
  Entity.PPR     → Personalized PageRank seeded at linked entities
  Rel.Aggregator → edge_score = ppr[src] + ppr[tgt]
  Chunk.Aggregator → chunk_score = sum(edge_scores for edges from that chunk)

ABSTRACT QA pipeline (thematic, cross-author, systemic):
  Community.VDB  → top-k communities by vector similarity to query
  Node.VDB       → fallback: top-k entity nodes by vector similarity
  Rel.Onehop     → all edges incident to VDB-retrieved nodes
"""

import json
from pathlib import Path
from typing import Optional

import anthropic
import networkx as nx

from .index_builder import (
    embed_texts,
    get_node_collection,
    get_relationship_collection,
    get_community_collection,
    load_communities,
)
from .graph_builder import _normalize_name, load_graph, GRAPH_PATH
from .prompts import LINKER_SYSTEM, LINKER_USER

import os
from dotenv import load_dotenv
load_dotenv()

MODEL = "claude-sonnet-4-6"

# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def _load_graph_cached() -> nx.DiGraph:
    """Load the RKG once; callers should pass G in directly when possible."""
    return load_graph(GRAPH_PATH)


# ─────────────────────────────────────────────────────────────
# OPERATOR 1 — Entity.Link
#
# Extract named entity mentions from the query string, then
# fuzzy-match them to canonical node keys in the RKG.
#
# Two-step:
#   a) Claude extracts surface-form mentions from the query
#   b) We match each mention against the node name index
#      (substring / normalised-name matching)
# ─────────────────────────────────────────────────────────────

def op_entity_link(
    query: str,
    G: nx.DiGraph,
    anthropic_client: anthropic.Anthropic,
) -> list[str]:
    """
    Entity.Link operator.
    Returns a list of RKG node keys that correspond to entities
    mentioned in the query.
    """
    # Step a: LLM extracts surface forms
    user_prompt = LINKER_USER.format(query=query)
    try:
        msg = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=LINKER_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences
        import re
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        mentions = json.loads(raw)
        if not isinstance(mentions, list):
            mentions = []
    except Exception:
        # Fall back to simple noun-phrase extraction from the query
        mentions = []

    # Also do direct substring scan on the query for short node names
    query_lower = query.lower()
    direct_hits = [
        key for key, d in G.nodes(data=True)
        if len(key) > 4
        and (key in query_lower
             or d.get("name", "").lower() in query_lower)
    ]

    # Step b: normalise and match mentions to node keys
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
            # Partial / substring match
            for nk, node_key in node_names.items():
                if norm in nk or nk in norm:
                    matched_keys.add(node_key)
                    break

    return list(matched_keys)


# ─────────────────────────────────────────────────────────────
# OPERATOR 2 — Entity.PPR
#
# Personalized PageRank seeded at the linked entity nodes.
# Propagates through the RKG to score all reachable nodes.
# Returns {node_key: ppr_score}.
# ─────────────────────────────────────────────────────────────

PPR_ALPHA      = 0.85
PPR_MAX_ITER   = 100

def op_entity_ppr(
    seed_keys: list[str],
    G: nx.DiGraph,
) -> dict[str, float]:
    """
    Entity.PPR operator.
    Returns {node_key: ppr_score} for all nodes in G,
    seeded uniformly at seed_keys.
    """
    if not seed_keys:
        return {}

    valid_seeds = [k for k in seed_keys if G.has_node(k)]
    if not valid_seeds:
        return {}

    weight = 1.0 / len(valid_seeds)
    personalization = {k: weight for k in valid_seeds}

    # nx.pagerank works on both directed and undirected;
    # use undirected view so PPR flows through all edges
    G_undirected = G.to_undirected()

    try:
        ppr = nx.pagerank(
            G_undirected,
            alpha=PPR_ALPHA,
            personalization=personalization,
            max_iter=PPR_MAX_ITER,
            weight="weight",
        )
    except nx.PowerIterationFailedConvergence:
        # Fallback: give equal score to all valid seeds
        ppr = {k: 1.0 / max(len(valid_seeds), 1) for k in valid_seeds}

    return ppr


# ─────────────────────────────────────────────────────────────
# OPERATOR 3 — Relationship.Aggregator
#
# Score each edge by the sum of PPR scores of its endpoints.
# Returns [(edge_id, score, edge_data), ...] sorted descending.
# ─────────────────────────────────────────────────────────────

def op_relationship_aggregator(
    ppr_scores: dict[str, float],
    G: nx.DiGraph,
    top_k: int = 50,
) -> list[dict]:
    """
    Relationship.Aggregator operator.
    edge_score = ppr[src] + ppr[tgt]
    Returns top-k relationship dicts sorted by score.
    """
    scored = []
    for src, tgt, data in G.edges(data=True):
        score = ppr_scores.get(src, 0.0) + ppr_scores.get(tgt, 0.0)
        if score > 0:
            scored.append({
                "src_key":     src,
                "tgt_key":     tgt,
                "src_name":    G.nodes[src].get("name", src),
                "tgt_name":    G.nodes[tgt].get("name", tgt),
                "rel_name":    data.get("name", "related_to"),
                "description": data.get("description", ""),
                "keywords":    data.get("keywords", []),
                "weight":      data.get("weight", 0.5),
                "source_chunk": data.get("source_chunk", ""),
                "score":       score,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────
# OPERATOR 4 — Chunk.Aggregator
#
# Score each source chunk by the sum of relationship scores for
# all relationships extracted from that chunk.
# Returns top-k chunk ids with their aggregated scores.
# ─────────────────────────────────────────────────────────────

def op_chunk_aggregator(
    relationships: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Chunk.Aggregator operator.
    chunk_score = Σ rel.score for all rels from that chunk.
    Returns sorted list of {chunk_id, score, contributing_rels}.
    """
    chunk_scores: dict[str, float] = {}
    chunk_rels:   dict[str, list]  = {}

    for rel in relationships:
        cid = rel.get("source_chunk", "")
        if not cid:
            continue
        chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rel["score"]
        chunk_rels.setdefault(cid, []).append(rel)

    results = [
        {
            "chunk_id":   cid,
            "score":      score,
            "rels":       chunk_rels[cid][:5],  # sample relations for context
        }
        for cid, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return results[:top_k]


# ─────────────────────────────────────────────────────────────
# OPERATOR 5 — Community.VDB
#
# Embed the query with BGE-M3 and retrieve top-k communities
# from the Community Index by cosine similarity.
# ─────────────────────────────────────────────────────────────

def op_community_vdb(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Community.VDB operator.
    Returns top-k community report dicts sorted by similarity to query.
    """
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
#
# Embed query and retrieve top-k entity nodes from Node Index.
# Used as fallback/complement in abstract QA pipeline.
# ─────────────────────────────────────────────────────────────

def op_node_vdb(
    query: str,
    top_k: int = 15,
) -> list[dict]:
    """
    Node.VDB operator.
    Returns top-k entity node dicts sorted by similarity to query.
    """
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
# OPERATOR 7 — Relationship.Onehop
#
# Given a set of entity node keys, return all edges incident
# to those nodes in the RKG.
# ─────────────────────────────────────────────────────────────

def op_relationship_onehop(
    node_keys: list[str],
    G: nx.DiGraph,
    top_k: int = 40,
) -> list[dict]:
    """
    Relationship.Onehop operator.
    Returns all edges incident to the given node keys,
    sorted by edge weight descending.
    """
    key_set = set(node_keys)
    relations = []
    seen = set()

    for key in node_keys:
        if not G.has_node(key):
            continue
        # Undirected: check both predecessors and successors
        for nbr in list(G.predecessors(key)) + list(G.successors(key)):
            edge_pair = (min(key, nbr), max(key, nbr))
            if edge_pair in seen:
                continue
            seen.add(edge_pair)

            # Get edge data (handle both directions)
            if G.has_edge(key, nbr):
                data = G[key][nbr]
                src_key, tgt_key = key, nbr
            else:
                data = G[nbr][key]
                src_key, tgt_key = nbr, key

            relations.append({
                "src_key":     src_key,
                "tgt_key":     tgt_key,
                "src_name":    G.nodes[src_key].get("name", src_key),
                "tgt_name":    G.nodes[tgt_key].get("name", tgt_key),
                "rel_name":    data.get("name", "related_to"),
                "description": data.get("description", ""),
                "keywords":    data.get("keywords", []),
                "weight":      data.get("weight", 0.5),
                "source_chunk": data.get("source_chunk", ""),
                "score":       data.get("weight", 0.5),
                "operator":    "Relationship.Onehop",
            })

    relations.sort(key=lambda x: x["weight"], reverse=True)
    return relations[:top_k]


# ─────────────────────────────────────────────────────────────
# CHUNK FETCHER
#
# Given chunk IDs, reads the actual text from clean_text/.
# This is where we resolve chunk_ids → passage text for the LLM.
# ─────────────────────────────────────────────────────────────

CLEAN_TEXT_DIR = Path("clean_text")

def fetch_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    """
    Read text content for a list of chunk IDs (= file stems).
    Returns {chunk_id: text}.
    """
    texts = {}
    for cid in chunk_ids:
        # Try exact stem match first
        candidates = list(CLEAN_TEXT_DIR.glob(f"{cid}.txt"))
        if not candidates:
            # Try with .clean.txt suffix (alternate extension)
            candidates = list(CLEAN_TEXT_DIR.glob(f"{cid}*.txt"))

        if candidates:
            raw = candidates[0].read_text(encoding="utf-8", errors="ignore")
            # Strip metadata header
            if "### END METADATA ###" in raw:
                text = raw.split("### END METADATA ###", 1)[1].strip()
            else:
                text = raw.strip()
            texts[cid] = text

    return texts


# ─────────────────────────────────────────────────────────────
# FULL SPECIFIC QA PIPELINE
#
# Entity.Link → Entity.PPR → Rel.Aggregator → Chunk.Aggregator
# ─────────────────────────────────────────────────────────────

def run_specific_pipeline(
    query: str,
    G: nx.DiGraph,
    anthropic_client: anthropic.Anthropic,
    n_chunks: int = 6,
    n_rels: int = 50,
) -> dict:
    """
    VGraphRAG Specific QA operator chain.

    Returns:
      chunks:        list of {chunk_id, text, score, rels}
      relationships: top scored relationships
      ppr_scores:    top PPR-scored nodes
      seed_entities: linked entity keys
    """
    # 1. Entity.Link
    seed_keys = op_entity_link(query, G, anthropic_client)
    print(f"  Entity.Link: {len(seed_keys)} seeds: {seed_keys[:5]}")

    # 2. Entity.PPR
    ppr = op_entity_ppr(seed_keys, G)

    top_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Entity.PPR: top nodes = {[(G.nodes[k].get('name', k), f'{s:.4f}') for k, s in top_ppr[:5]]}")

    # 3. Relationship.Aggregator
    relationships = op_relationship_aggregator(ppr, G, top_k=n_rels)
    print(f"  Rel.Aggregator: {len(relationships)} scored relationships")

    # 4. Chunk.Aggregator
    chunk_results = op_chunk_aggregator(relationships, top_k=n_chunks * 2)
    print(f"  Chunk.Aggregator: {len(chunk_results)} scored chunks")

    # 5. Fetch texts
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
#
# Community.VDB → Node.VDB → Relationship.Onehop
# ─────────────────────────────────────────────────────────────

def run_abstract_pipeline(
    query: str,
    G: nx.DiGraph,
    n_communities: int = 4,
    n_nodes: int = 10,
    n_chunks: int = 6,
) -> dict:
    """
    VGraphRAG Abstract QA operator chain.

    Returns:
      communities: top community reports
      chunks:      passage chunks derived from community nodes + onehop relations
      relationships: onehop relationships from VDB-retrieved nodes
    """
    # 1. Community.VDB
    communities = op_community_vdb(query, top_k=n_communities)
    print(f"  Community.VDB: {len(communities)} communities retrieved")

    # 2. Node.VDB (fallback / complement)
    vdb_nodes = op_node_vdb(query, top_k=n_nodes)
    print(f"  Node.VDB: {len(vdb_nodes)} nodes retrieved")

    # Collect all node keys: from VDB + from community membership
    node_keys = [n["node_key"] for n in vdb_nodes]
    for comm in communities:
        node_keys.extend(comm.get("node_keys", []))
    node_keys = list(dict.fromkeys(node_keys))  # deduplicate, preserve order

    # 3. Relationship.Onehop
    relationships = op_relationship_onehop(node_keys, G, top_k=40)
    print(f"  Rel.Onehop: {len(relationships)} relationships")

    # Collect chunk IDs: from onehop rels + from VDB node source_chunks
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

    # Score chunks: rels provide richer signal; VDB node chunks as fallback
    rel_chunk_set = {rel.get("source_chunk") for rel in relationships}
    chunk_ids_scored = sorted(
        chunk_ids,
        key=lambda cid: (cid in rel_chunk_set, sum(
            r["score"] for r in relationships if r.get("source_chunk") == cid
        )),
        reverse=True,
    )

    # Fetch texts
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
