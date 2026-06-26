"""
index_builder.py
================
Stage 2: RKG → Three Indexes in ChromaDB

Index 1 — Node Index
  Embed: "{entity_name} ({type}): {description}"
  Collection: vgraphrag_nodes

Index 2 — Relationship Index
  Embed: "{source_name} {relation_name} {target_name}: {description} [{keywords}]"
  Collection: vgraphrag_relationships

Index 3 — Community Index
  Algorithm: Leiden (via leidenalg + igraph)
  Summary:   LLM-generated per community (via Claude)
  Embed:     community summary text
  Collection: vgraphrag_communities

All collections use BGE-M3 embeddings (1024 dims) from sentence-transformers.
Indexes are stored in vgraphrag_db/chroma/.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import anthropic
import chromadb
import igraph as ig
import leidenalg
import networkx as nx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .graph_builder import load_graph, GRAPH_PATH
from .prompts import COMMUNITY_SYSTEM, COMMUNITY_USER

load_dotenv()

# ─── paths ────────────────────────────────────────────────────
DB_DIR          = Path("vgraphrag_db")
CHROMA_DIR      = DB_DIR / "chroma"
COMMUNITIES_PATH = DB_DIR / "communities.json"

# ─── collections ──────────────────────────────────────────────
NODE_COLLECTION         = "vgraphrag_nodes"
RELATIONSHIP_COLLECTION = "vgraphrag_relationships"
COMMUNITY_COLLECTION    = "vgraphrag_communities"

# ─── embedding model ──────────────────────────────────────────
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_DIMS       = 1024

# ─── LLM ──────────────────────────────────────────────────────
MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 1024

# ─── leiden ───────────────────────────────────────────────────
LEIDEN_RESOLUTION  = 1.0   # lower = larger communities
MIN_COMMUNITY_SIZE = 5     # discard tiny communities


# ─────────────────────────────────────────────────────────────
# BGE-M3 EMBEDDER  (loaded once, shared across calls)
# ─────────────────────────────────────────────────────────────

_bge_model: Optional[SentenceTransformer] = None

def _get_bge() -> SentenceTransformer:
    global _bge_model
    if _bge_model is None:
        print(f"Loading {BGE_MODEL_NAME}...")
        _bge_model = SentenceTransformer(BGE_MODEL_NAME)
    return _bge_model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """BGE-M3 batch embedding. Returns list of 1024-dim float vectors."""
    model  = _get_bge()
    result = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        result.extend(vecs.tolist())
    return result


# ─────────────────────────────────────────────────────────────
# CHROMADB CLIENT
# ─────────────────────────────────────────────────────────────

def _get_chroma() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_or_create(client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
    )


def _clear_collection(col: chromadb.Collection) -> None:
    if col.count() > 0:
        existing = col.get(limit=col.count())
        col.delete(ids=existing["ids"])


# ─────────────────────────────────────────────────────────────
# INDEX 1 — NODE INDEX
# ─────────────────────────────────────────────────────────────

def build_node_index(G: nx.DiGraph, force: bool = False) -> chromadb.Collection:
    """
    Embed each entity node's description into the Node Index.
    Text format: "{name} ({type}): {description}"
    """
    client = _get_chroma()
    col    = _get_or_create(client, NODE_COLLECTION)

    if not force and col.count() > 0:
        print(f"  Node Index: {col.count()} entries (skipped — use force=True to rebuild)")
        return col

    _clear_collection(col)

    nodes = [(key, data) for key, data in G.nodes(data=True)]
    print(f"  Embedding {len(nodes)} entity nodes...")

    texts, ids, metadatas = [], [], []
    for key, data in nodes:
        name = data.get("name", key)
        typ  = data.get("type", "CONCEPT")
        desc = data.get("description", "")
        text = f"{name} ({typ}): {desc}" if desc else f"{name} ({typ})"

        texts.append(text)
        ids.append(key)
        metadatas.append({
            "name":          name,
            "type":          typ,
            "description":   desc[:500],
            "source_chunks": json.dumps(data.get("source_chunks", [])[:20]),
            "work_titles":   json.dumps(data.get("work_titles", [])[:10]),
            "degree":        G.degree(key),
        })

    # Embed in batches and upsert
    embeddings = embed_texts(texts)

    BATCH = 200
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    print(f"  ✓ Node Index: {col.count()} entities")
    return col


# ─────────────────────────────────────────────────────────────
# INDEX 2 — RELATIONSHIP INDEX
# ─────────────────────────────────────────────────────────────

def build_relationship_index(G: nx.DiGraph, force: bool = False) -> chromadb.Collection:
    """
    Embed each relationship edge into the Relationship Index.
    Text format: "{src_name} {rel_name} {tgt_name}: {description} [{keywords}]"
    Each edge gets a composite ID: "{src_key}|{tgt_key}"
    """
    client = _get_chroma()
    col    = _get_or_create(client, RELATIONSHIP_COLLECTION)

    if not force and col.count() > 0:
        print(f"  Relationship Index: {col.count()} entries (skipped)")
        return col

    _clear_collection(col)

    edges = list(G.edges(data=True))
    print(f"  Embedding {len(edges)} relationships...")

    texts, ids, metadatas = [], [], []
    seen_ids = set()

    for src_key, tgt_key, data in edges:
        src_name = G.nodes[src_key].get("name", src_key)
        tgt_name = G.nodes[tgt_key].get("name", tgt_key)
        rel_name = data.get("name", "related_to")
        desc     = data.get("description", "")
        keywords = data.get("keywords", [])
        kw_str   = ", ".join(keywords) if keywords else ""

        text = f"{src_name} {rel_name} {tgt_name}"
        if desc:
            text += f": {desc}"
        if kw_str:
            text += f" [{kw_str}]"

        edge_id = f"{src_key}|{tgt_key}"
        # Guard against duplicate IDs (shouldn't happen, but just in case)
        if edge_id in seen_ids:
            edge_id = f"{edge_id}_{len(seen_ids)}"
        seen_ids.add(edge_id)

        texts.append(text)
        ids.append(edge_id)
        metadatas.append({
            "source_entity":  src_key,
            "target_entity":  tgt_key,
            "source_name":    src_name,
            "target_name":    tgt_name,
            "relation_name":  rel_name,
            "keywords":       kw_str,
            "description":    desc[:500],
            "weight":         data.get("weight", 0.5),
            "source_chunk":   data.get("source_chunk", ""),
            "work_title":     data.get("work_title", ""),
        })

    embeddings = embed_texts(texts)

    BATCH = 200
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    print(f"  ✓ Relationship Index: {col.count()} relationships")
    return col


# ─────────────────────────────────────────────────────────────
# LEIDEN COMMUNITY DETECTION
# Converts the NetworkX graph to igraph, runs Leiden,
# returns a list of community dicts.
# ─────────────────────────────────────────────────────────────

def _nx_to_igraph(G: nx.DiGraph) -> tuple[ig.Graph, list[str]]:
    """
    Convert NetworkX DiGraph to igraph undirected Graph for Leiden.
    Returns (igraph_graph, node_keys_in_order).
    """
    node_keys = list(G.nodes())
    key_to_idx = {k: i for i, k in enumerate(node_keys)}

    edges    = [(key_to_idx[u], key_to_idx[v]) for u, v in G.edges()]
    weights  = [G[u][v].get("weight", 1.0) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(node_keys), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    return ig_graph, node_keys


def detect_leiden_communities(G: nx.DiGraph) -> list[dict]:
    """
    Run Leiden algorithm with modularity optimization.
    Returns list of community dicts:
      {id, node_keys, size, top_types, works}
    """
    ig_graph, node_keys = _nx_to_igraph(G)

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        n_iterations=-1,  # run until stable
        seed=42,
    )

    communities = []
    for comm_idx, member_indices in enumerate(partition):
        if len(member_indices) < MIN_COMMUNITY_SIZE:
            continue

        members = [node_keys[i] for i in member_indices]

        # Collect type distribution and works
        type_counts: dict = {}
        works: set = set()
        for key in members:
            t = G.nodes[key].get("type", "CONCEPT")
            type_counts[t] = type_counts.get(t, 0) + 1
            for w in G.nodes[key].get("work_titles", []):
                works.add(w)

        communities.append({
            "id":        comm_idx,
            "node_keys": members,
            "size":      len(members),
            "top_types": sorted(type_counts, key=type_counts.get, reverse=True)[:3],
            "works":     sorted(works),
        })

    communities.sort(key=lambda c: c["size"], reverse=True)
    return communities


# ─────────────────────────────────────────────────────────────
# LLM COMMUNITY SUMMARIES
# ─────────────────────────────────────────────────────────────

def _generate_community_summary(
    client: anthropic.Anthropic,
    G: nx.DiGraph,
    community: dict,
    max_entities: int = 30,
    max_rels: int = 40,
) -> str:
    """Generate a thematic LLM summary for one community."""
    members = community["node_keys"][:max_entities]

    entity_lines = []
    for key in members:
        d = G.nodes[key]
        entity_lines.append(
            f"- {d.get('name', key)} ({d.get('type', '?')}): {d.get('description', '')[:120]}"
        )

    # Gather relationships within this community
    member_set = set(members)
    rel_lines  = []
    for u, v, d in G.edges(data=True):
        if u in member_set and v in member_set and len(rel_lines) < max_rels:
            u_name = G.nodes[u].get("name", u)
            v_name = G.nodes[v].get("name", v)
            rel_lines.append(
                f"- {u_name} → {d.get('name', 'related_to')} → {v_name}: "
                f"{d.get('description', '')[:100]}"
            )

    user_prompt = COMMUNITY_USER.format(
        entity_list="\n".join(entity_lines),
        relationship_list="\n".join(rel_lines) if rel_lines else "(none within community)",
        works_list=", ".join(community["works"][:20]),
    )

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=COMMUNITY_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  Community summary failed: {e}")
        # Fallback: structured summary without LLM
        return (
            f"Community of {community['size']} entities. "
            f"Types: {', '.join(community['top_types'])}. "
            f"Works: {', '.join(community['works'][:10])}."
        )


# ─────────────────────────────────────────────────────────────
# INDEX 3 — COMMUNITY INDEX
# ─────────────────────────────────────────────────────────────

def build_community_index(G: nx.DiGraph, force: bool = False) -> chromadb.Collection:
    """
    Detect Leiden communities, generate LLM summaries, embed them.
    Also saves communities.json to disk for use by the retriever.
    """
    client_chroma = _get_chroma()
    col           = _get_or_create(client_chroma, COMMUNITY_COLLECTION)

    if not force and col.count() > 0 and COMMUNITIES_PATH.exists():
        print(f"  Community Index: {col.count()} communities (skipped)")
        return col

    _clear_collection(col)

    print("  Detecting Leiden communities...")
    communities = detect_leiden_communities(G)
    print(f"  {len(communities)} communities (≥{MIN_COMMUNITY_SIZE} members)")

    client_llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    texts, ids, metadatas = [], [], []
    enriched_communities  = []

    for i, comm in enumerate(communities):
        print(f"  Summarising community {i+1}/{len(communities)} (size={comm['size']})...")
        summary = _generate_community_summary(client_llm, G, comm)
        time.sleep(0.5)  # rate limit buffer

        comm["summary"] = summary
        enriched_communities.append(comm)

        comm_id = f"comm_{comm['id']}"
        texts.append(summary)
        ids.append(comm_id)
        metadatas.append({
            "community_id": comm["id"],
            "size":         comm["size"],
            "top_types":    json.dumps(comm["top_types"]),
            "works":        json.dumps(comm["works"][:30]),
            "node_keys":    json.dumps(comm["node_keys"][:50]),
        })

    # Save to disk (retriever reads this to look up node membership)
    COMMUNITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMUNITIES_PATH, "w") as f:
        json.dump(enriched_communities, f, indent=2, ensure_ascii=False)

    # Embed and store
    embeddings = embed_texts(texts)
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    print(f"  ✓ Community Index: {col.count()} communities")
    return col


# ─────────────────────────────────────────────────────────────
# BUILD ALL INDEXES  (called from build_vgraphrag.py)
# ─────────────────────────────────────────────────────────────

def build_all_indexes(graph_path: Path = GRAPH_PATH, force: bool = False) -> dict:
    """
    Load the RKG from disk and build all three indexes.
    Returns a dict with collection references.
    """
    print(f"\nLoading RKG from {graph_path}...")
    G = load_graph(graph_path)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[1/3] Building Node Index...")
    node_col = build_node_index(G, force=force)

    print("\n[2/3] Building Relationship Index...")
    rel_col = build_relationship_index(G, force=force)

    print("\n[3/3] Building Community Index (Leiden + LLM summaries)...")
    comm_col = build_community_index(G, force=force)

    return {
        "graph":         G,
        "node_col":      node_col,
        "rel_col":       rel_col,
        "community_col": comm_col,
    }


# ─────────────────────────────────────────────────────────────
# ACCESSORS (used by retriever.py)
# ─────────────────────────────────────────────────────────────

def get_node_collection() -> chromadb.Collection:
    return _get_or_create(_get_chroma(), NODE_COLLECTION)

def get_relationship_collection() -> chromadb.Collection:
    return _get_or_create(_get_chroma(), RELATIONSHIP_COLLECTION)

def get_community_collection() -> chromadb.Collection:
    return _get_or_create(_get_chroma(), COMMUNITY_COLLECTION)

def load_communities() -> list[dict]:
    if not COMMUNITIES_PATH.exists():
        return []
    with open(COMMUNITIES_PATH) as f:
        return json.load(f)
