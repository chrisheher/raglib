"""
Hybrid GraphRAG Retrieval Pipeline
===================================
Designed using the vgraphRAG operator vocabulary (arXiv:2503.04338 / DIGIMON framework).

Four-stage architecture:
  Stage 1  Graph Building      — TKG from metadata + entity co-occurrence
  Stage 2  Index Construction  — Node Index (existing) + Community Index (built here)
  Stage 3  Operator Config     — 8 modular operators wired into two pipelines
  Stage 4  Retrieval & Gen     — Query-routed dispatch + RRF fusion

Query routing:
  thematic / abstract  →  Community pipeline
                             Entity.VDB → Entity.PPR → Community.Entity + Chunk.Occurrence
  character / intertextual → Entity-Relation pipeline
                             Entity.VDB + Entity.Link → Relationship.Onehop → Chunk.FromRel
  hybrid (ambiguous)   →  Both pipelines, fused with Reciprocal Rank Fusion

Usage:
  from hybrid_pipeline import build_indexes, classify_query, run_pipeline
"""

import re
import json
import math
import time
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Literal, Optional

import networkx as nx
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHROMA_PATH        = "chroma_db"
CHUNK_COLLECTION   = "literary_documents"
ENTITY_COLLECTION  = "literary_entities"       # Node Index over works/themes
COMMUNITY_COLLECTION = "literary_communities"  # Community Index
EMBEDDING_MODEL    = "text-embedding-3-large"
EMBEDDING_DIMS     = 3072
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")

# Operator defaults
DEFAULT_K_ENTITY     = 12   # Entity.VDB top-k
DEFAULT_K_COMMUNITY  = 4    # Community.Entity top-k
DEFAULT_K_CHUNKS     = 8    # Chunk.Occurrence / Chunk.FromRel pool size
DEFAULT_N_FINAL      = 6    # chunks sent to LLM
PPR_ALPHA            = 0.85  # PageRank damping
PPR_ITERATIONS       = 30


# ─────────────────────────────────────────────
# CLIENTS (lazy-initialized)
# ─────────────────────────────────────────────
_openai_client    = None
_chroma_client    = None
_anthropic_client = None


def _oai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _claude():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def _collection(name: str, create=False):
    c = _chroma()
    if create:
        return c.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    try:
        return c.get_collection(name)
    except Exception:
        return None


# ─────────────────────────────────────────────
# EMBEDDING HELPER
# ─────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed up to 100 texts."""
    results = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        resp = _oai().embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS
        )
        results.extend([r.embedding for r in resp.data])
    return results


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — GRAPH BUILDING
#
# Builds a Textual Knowledge Graph (TKG) over the corpus.
# Nodes:  works (source titles), themes, genres, myth traditions,
#         consciousness techniques, nautical contexts
# Edges:  labeled semantic relations derived from chunk metadata
# ═══════════════════════════════════════════════════════════════

def _extract_title(source: str) -> str:
    """Normalize a chunk source field to a canonical work title."""
    s = re.sub(r'_\d+$', '', source.strip())
    s = s.split('__')[0]
    return s.replace('_', ' ').strip()


def build_tkg(all_metadatas: list[dict]) -> nx.Graph:
    """
    Stage 1: Graph Building
    Returns a TKG where:
      - work nodes  carry: genre, chunk_count
      - theme nodes carry: type='theme'
      - tradition / technique / nautical_context nodes similarly typed
    Edges carry: relation label + weight (frequency-normalized)
    """
    G = nx.Graph()
    edge_counts: dict[tuple, int] = defaultdict(int)

    for meta in all_metadatas:
        source = meta.get("source", "")
        if not source:
            continue

        work = _extract_title(source)
        genre = meta.get("genre", "prose")

        if not G.has_node(work):
            G.add_node(work, type="work", genre=genre, chunk_count=0)
        G.nodes[work]["chunk_count"] = G.nodes[work].get("chunk_count", 0) + 1

        # theme edges
        for theme in meta.get("themes", "").split(", "):
            theme = theme.strip()
            if not theme:
                continue
            if not G.has_node(theme):
                G.add_node(theme, type="theme")
            edge_counts[(work, theme, "has_theme")] += 1

        # myth tradition edge
        trad = meta.get("myth_tradition", "")
        if trad:
            if not G.has_node(trad):
                G.add_node(trad, type="tradition")
            edge_counts[(work, trad, "draws_from")] += 1

        # consciousness technique edge
        tech = meta.get("consciousness_technique", "")
        if tech:
            if not G.has_node(tech):
                G.add_node(tech, type="technique")
            edge_counts[(work, tech, "uses_technique")] += 1

        # nautical context edge
        naut = meta.get("nautical_context", "")
        if naut:
            if not G.has_node(naut):
                G.add_node(naut, type="nautical_context")
            edge_counts[(work, naut, "set_in")] += 1

        # genre node (connects works sharing a genre)
        if not G.has_node(genre):
            G.add_node(genre, type="genre")
        edge_counts[(work, genre, "belongs_to_genre")] += 1

    # Add all edges with normalized weights
    for (u, v, rel), count in edge_counts.items():
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(G[u][v]["weight"], count / 10.0)
            G[u][v]["relations"] = G[u][v].get("relations", []) + [rel]
        else:
            G.add_edge(u, v, relation=rel, weight=min(count / 10.0, 1.0), relations=[rel])

    return G


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — INDEX CONSTRUCTION
#
# Community detection on the TKG → community summary reports →
# embedded into a Community Index in ChromaDB.
#
# Run build_indexes() once (or when corpus changes).
# ═══════════════════════════════════════════════════════════════

def detect_communities(G: nx.Graph, n_clusters: int = 30) -> dict[int, list[str]]:
    """
    Community detection via KMeans clustering on embedded work titles.

    Embeds each canonical work title string directly using the OpenAI embedding
    model, then clusters the resulting vectors. Avoids ChromaDB embedding
    retrieval issues while still producing semantically grounded communities.
    """
    from collections import defaultdict
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        import numpy as np
    except ImportError:
        work_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "work"]
        return {0: work_nodes}

    # Collect work titles from the TKG
    titles = [n for n, d in G.nodes(data=True) if d.get("type") == "work"]
    print(f"  Community detection: {len(titles)} work titles to cluster")

    if len(titles) < n_clusters:
        return {0: titles}

    # Embed the title strings (cheap — ~315 short strings)
    try:
        vecs = embed(titles)
    except Exception as exc:
        print(f"  Community detection embedding failed ({exc}), falling back to 1 community")
        return {0: titles}

    X = normalize(np.array(vecs))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    communities: dict[int, list[str]] = defaultdict(list)
    for title, label in zip(titles, labels):
        communities[int(label)].append(title)

    return dict(communities)


_COMMUNITY_INSIGHT_PROMPT = """\
You are analyzing a cluster of literary works that were grouped together by \
semantic similarity. Based on the titles and representative passages below, \
write 3–4 sentences describing the conceptual thread that unites this cluster.

Focus on the actual ideas at stake — the shared obsessions, tensions, or ways \
of seeing — not surface genre labels or theme keywords. Speak as a literary \
critic, not a cataloguer. Do not list themes by name. Do not use bullet points.

Go deep: 6–8 sentences. Explore the underlying tension or contradiction that \
animates the cluster, how different works approach it from different angles, \
and what this grouping reveals that no single work could reveal alone.

Works in this cluster ({n} total): {titles}

Representative passages:
{passages}

Write the insight now:"""


def _community_summary(comm_id: int, nodes: list[str], G: nx.Graph,
                       common_themes: set[str] | None = None) -> dict:
    """Generate an LLM-written insight for a community cluster."""
    if common_themes is None:
        common_themes = set()

    # Collect lightweight metadata for return value
    genres: dict[str, int] = defaultdict(int)
    themes: dict[str, int] = defaultdict(int)
    traditions: set[str] = set()
    techniques: set[str] = set()
    for node in nodes:
        nd = G.nodes[node]
        g = nd.get("genre", "")
        if g:
            genres[g] += nd.get("chunk_count", 1)
        for nbr in G[node]:
            nbr_type = G.nodes[nbr].get("type", "")
            if nbr_type == "theme" and nbr not in common_themes:
                themes[nbr] += 1
            elif nbr_type == "tradition":
                traditions.add(nbr)
            elif nbr_type == "technique":
                techniques.add(nbr)

    top_genres = sorted(genres, key=genres.get, reverse=True)[:3]
    top_themes = sorted(themes, key=themes.get, reverse=True)[:6]

    # Pick the 5 most central works (highest chunk_count in TKG)
    ranked = sorted(
        nodes,
        key=lambda n: G.nodes[n].get("chunk_count", 0),
        reverse=True
    )[:5]

    # Pull one representative passage per selected work from ChromaDB
    col = _collection(CHUNK_COLLECTION)
    passages = []
    if col:
        for title in ranked:
            try:
                src_key = title.replace(" ", "_") + ".txt"
                res = col.get(
                    where={"source": {"$eq": src_key}},
                    limit=1,
                    include=["documents"],
                )
                if res["documents"]:
                    snippet = " ".join(res["documents"][0].split()[:120])
                    passages.append(f"[{title}]\n{snippet}")
            except Exception:
                pass

    # Fall back to titles only if ChromaDB lookup fails
    if not passages:
        passages = [f"[{t}]" for t in ranked]

    titles_str   = ", ".join(nodes[:15]) + (f" … and {len(nodes)-15} more" if len(nodes) > 15 else "")
    passages_str = "\n\n".join(passages)

    prompt = _COMMUNITY_INSIGHT_PROMPT.format(
        n=len(nodes),
        titles=titles_str,
        passages=passages_str,
    )

    try:
        resp = _claude().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        report = resp.content[0].text.strip()
    except Exception as exc:
        # Fallback: plain title list
        report = f"Cluster of {len(nodes)} works: {titles_str}."

    return {
        "community_id": comm_id,
        "nodes":        nodes,
        "genres":       top_genres,
        "themes":       top_themes,
        "traditions":   list(traditions),
        "techniques":   list(techniques),
        "works":        nodes,
        "report":       report,
    }


def build_indexes(force_rebuild=False):
    """
    Stage 2: Index Construction
    Loads all chunk metadata from ChromaDB, builds the TKG,
    detects communities, generates reports, embeds them into
    the Community Index.

    Also persists the TKG graph to disk as tkg.gpickle.

    Call this once after loading your corpus, or when it changes.
    """
    chunk_col = _collection(CHUNK_COLLECTION)
    if chunk_col is None:
        raise RuntimeError("Chunk collection not found. Run embed.py first.")

    community_col = _collection(COMMUNITY_COLLECTION, create=True)
    tkg_path = Path(CHROMA_PATH) / "tkg.gpickle"

    if not force_rebuild and tkg_path.exists() and community_col.count() > 0:
        print("✓ Indexes already built. Use force_rebuild=True to regenerate.")
        import pickle
        with open(str(tkg_path), 'rb') as f:
            G = pickle.load(f)
        return G

    print(f"Building TKG from {chunk_col.count()} chunks...")
    all_meta = chunk_col.get(limit=chunk_col.count(), include=["metadatas"])
    metadatas = all_meta["metadatas"]

    G = build_tkg(metadatas)
    print(f"  TKG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    import pickle
    with open(str(tkg_path), 'wb') as f:
        pickle.dump(G, f)
    print(f"  TKG saved to {tkg_path}")

    print("Detecting communities...")
    communities = detect_communities(G)
    print(f"  {len(communities)} communities detected")

    # Compute global theme frequency so summaries can suppress ubiquitous themes
    total_works = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "work")
    global_theme_freq: dict[str, int] = defaultdict(int)
    for n, d in G.nodes(data=True):
        if d.get("type") == "work":
            for nbr in G[n]:
                if G.nodes[nbr].get("type") == "theme":
                    global_theme_freq[nbr] += 1
    # Themes present in >40% of works are corpus-wide noise — suppress them
    common_themes = {t for t, cnt in global_theme_freq.items() if cnt / max(total_works, 1) > 0.40}
    if common_themes:
        print(f"  Suppressing {len(common_themes)} ubiquitous themes from summaries: "
              f"{', '.join(sorted(common_themes))}")

    reports = [_community_summary(cid, nodes, G, common_themes) for cid, nodes in communities.items()]

    # Embed community reports
    print("Embedding community reports...")
    texts = [r["report"] for r in reports]
    embeddings = embed(texts)

    # Clear and repopulate community collection
    if community_col.count() > 0:
        existing = community_col.get(limit=community_col.count())
        community_col.delete(ids=existing["ids"])

    community_col.add(
        ids=[f"comm_{r['community_id']}" for r in reports],
        embeddings=embeddings,
        documents=[r["report"] for r in reports],
        metadatas=[{
            "community_id": r["community_id"],
            "genres": ", ".join(r["genres"]),
            "themes": ", ".join(r["themes"]),
            "works": json.dumps(r["works"][:20]),  # store sample
            "traditions": ", ".join(r["traditions"]),
            "techniques": ", ".join(r["techniques"]),
        } for r in reports]
    )
    print(f"  ✓ Community Index: {community_col.count()} reports")

    return G


def _load_tkg() -> nx.Graph:
    tkg_path = Path(CHROMA_PATH) / "tkg.gpickle"
    if not tkg_path.exists():
        raise RuntimeError("TKG not found. Run build_indexes() first.")
    import pickle
    with open(str(tkg_path), 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — OPERATORS
#
# Each operator has a clear signature and single responsibility.
# They are designed to be composed into pipeline chains.
# ═══════════════════════════════════════════════════════════════

# ── Entity Operators ─────────────────────────────────────────

def op_entity_vdb(query: str, k: int = DEFAULT_K_ENTITY) -> list[dict]:
    """
    Entity.VDB — semantic ANN search over the Node (chunk) Index.
    Seed: query string
    Returns: list of {id, text, metadata, score}
    Maps to: Entity.VDB in the vgraphRAG vocabulary
    """
    col = _collection(CHUNK_COLLECTION)
    if col is None:
        return []

    q_emb = embed([query])[0]
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "id": rid,
            "text": doc,
            "metadata": meta,
            "score": 1 - dist,
            "retrieval_path": "Entity.VDB"
        }
        for rid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def op_entity_link(query: str, G: nx.Graph) -> list[str]:
    """
    Entity.Link — map query surface-form mentions to canonical TKG nodes.
    Checks for work titles, theme names, tradition names.
    Seed: query string
    Returns: list of matched node names in G
    Maps to: Entity.Link in the vgraphRAG vocabulary
    """
    query_lower = query.lower()
    linked = []
    for node in G.nodes():
        if len(node) > 3 and node.lower() in query_lower:
            linked.append(node)
    return linked


def op_entity_ppr(seed_works: list[str], G: nx.Graph) -> dict[str, float]:
    """
    Entity.PPR — Personalized PageRank seeded at the retrieved work nodes.
    Propagates relevance through the TKG to find thematically related nodes.
    Seed: list of work/theme/tradition node names
    Returns: {node_name: ppr_score}
    Maps to: Entity.PPR in the vgraphRAG vocabulary
    """
    if not seed_works or G.number_of_nodes() == 0:
        return {}

    personalization = {}
    nodes_in_G = [n for n in seed_works if G.has_node(n)]
    if not nodes_in_G:
        return {}

    seed_weight = 1.0 / len(nodes_in_G)
    for n in nodes_in_G:
        personalization[n] = seed_weight

    try:
        ppr = nx.pagerank(
            G,
            alpha=PPR_ALPHA,
            personalization=personalization,
            max_iter=PPR_ITERATIONS,
            weight="weight"
        )
    except nx.PowerIterationFailedConvergence:
        ppr = {n: 1.0 for n in nodes_in_G}

    return ppr


# ── Relationship Operators ────────────────────────────────────

def op_relationship_onehop(seed_entities: list[dict], G: nx.Graph) -> list[dict]:
    """
    Relationship.Onehop — all edges incident to seed entity nodes.
    Seed: list of entity dicts (with 'metadata.source' field)
    Returns: list of {src, tgt, relation, weight}
    Maps to: Relationship.Onehop in the vgraphRAG vocabulary
    """
    relations = []
    seen = set()

    for ent in seed_entities:
        work = _extract_title(ent["metadata"].get("source", ""))
        if not G.has_node(work):
            continue
        for neighbor, edata in G[work].items():
            edge_key = (min(work, neighbor), max(work, neighbor))
            if edge_key not in seen:
                seen.add(edge_key)
                relations.append({
                    "src": work,
                    "tgt": neighbor,
                    "relation": edata.get("relation", "related_to"),
                    "weight": edata.get("weight", 0.5),
                    "tgt_type": G.nodes[neighbor].get("type", "unknown")
                })
    return relations


def op_relationship_vdb(query: str, k: int = 8) -> list[dict]:
    """
    Relationship.VDB — semantic search over the Community Index for
    community-level structural relations. (Reuses community embeddings as
    a proxy for the Relationship Index — captures thematic linkages.)
    Seed: query string
    Returns: list of community report dicts with score
    Maps to: Relationship.VDB (approximate — community-level)
    """
    col = _collection(COMMUNITY_COLLECTION)
    if col is None or col.count() == 0:
        return []

    q_emb = embed([query])[0]
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "id": rid,
            "report": doc,
            "metadata": meta,
            "score": 1 - dist,
            "retrieval_path": "Relationship.VDB"
        }
        for rid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


# ── Chunk Operators ───────────────────────────────────────────

def op_chunk_from_rel(relations: list[dict], seed_entities: list[dict]) -> list[dict]:
    """
    Chunk.FromRel — return source chunks that were retrieved alongside
    the seed entities AND share work/theme nodes with the relation endpoints.
    Seed: relations (from Relationship.Onehop) + seed entity chunks
    Returns: filtered and scored list of chunk dicts
    Maps to: Chunk.FromRel in the vgraphRAG vocabulary
    """
    # Build set of works connected via relations
    connected_works = set()
    for rel in relations:
        connected_works.add(rel["src"])
        connected_works.add(rel["tgt"])

    out = []
    for ent in seed_entities:
        work = _extract_title(ent["metadata"].get("source", ""))
        if work in connected_works:
            chunk = ent.copy()
            chunk["retrieval_path"] = "Chunk.FromRel"
            # Boost score for chunks that sit on a relation endpoint
            chunk["score"] = ent.get("score", 0.5) * 1.15
            out.append(chunk)

    return out


def op_chunk_occurrence(seed_entities: list[dict], ppr_scores: dict[str, float],
                        k: int = DEFAULT_K_CHUNKS) -> list[dict]:
    """
    Chunk.Occurrence — rank seed chunks by co-occurrence with high-PPR nodes.
    Each chunk's score is boosted by the PPR score of its source work.
    Seed: seed entity chunks + PPR score map
    Returns: top-k chunks sorted by co-occurrence score
    Maps to: Chunk.Occurrence in the vgraphRAG vocabulary
    """
    scored = []
    for ent in seed_entities:
        work = _extract_title(ent["metadata"].get("source", ""))
        ppr_boost = ppr_scores.get(work, 0.0)
        combined_score = ent.get("score", 0.5) + ppr_boost
        chunk = ent.copy()
        chunk["score"] = combined_score
        chunk["ppr_boost"] = ppr_boost
        chunk["retrieval_path"] = "Chunk.Occurrence"
        scored.append(chunk)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ── Community Operators ───────────────────────────────────────

def op_community_entity(seed_entities: list[dict], query: str,
                        k: int = DEFAULT_K_COMMUNITY) -> list[dict]:
    """
    Community.Entity — retrieve community reports for communities that
    contain the seed entity works. Falls back to VDB search if no match.
    Seed: seed entity chunks (from Entity.VDB)
    Returns: list of community report dicts
    Maps to: Community.Entity in the vgraphRAG vocabulary
    """
    col = _collection(COMMUNITY_COLLECTION)
    if col is None or col.count() == 0:
        return []

    # Get the works from seed entities
    seed_works = set()
    for ent in seed_entities:
        work = _extract_title(ent["metadata"].get("source", ""))
        seed_works.add(work)

    if not seed_works:
        return op_relationship_vdb(query, k=k)

    # Fetch all community reports and find ones containing seed works
    all_comms = col.get(limit=col.count(), include=["documents", "metadatas"])
    matched = []
    for i, (cid, doc, meta) in enumerate(zip(
        all_comms["ids"], all_comms["documents"], all_comms["metadatas"]
    )):
        works_in_comm = set(json.loads(meta.get("works", "[]")))
        overlap = seed_works & works_in_comm
        if overlap:
            matched.append({
                "id": cid,
                "report": doc,
                "metadata": meta,
                "score": len(overlap) / max(len(seed_works), 1),
                "overlap_works": list(overlap),
                "retrieval_path": "Community.Entity"
            })

    matched.sort(key=lambda x: x["score"], reverse=True)

    # If no structural match, fall back to semantic VDB search
    if not matched:
        return op_relationship_vdb(query, k=k)

    return matched[:k]


def op_community_layer(level_threshold: float = 0.3,
                       k: int = DEFAULT_K_COMMUNITY) -> list[dict]:
    """
    Community.Layer — retrieve the top-level (most coherent) communities
    regardless of query. Used for very abstract/broad thematic queries.
    Returns communities sorted by internal coherence (approximated by size).
    Maps to: Community.Layer in the vgraphRAG vocabulary
    """
    col = _collection(COMMUNITY_COLLECTION)
    if col is None or col.count() == 0:
        return []

    all_comms = col.get(limit=col.count(), include=["documents", "metadatas"])
    reports = [
        {
            "id": cid,
            "report": doc,
            "metadata": meta,
            "score": len(json.loads(meta.get("works", "[]"))),  # size as proxy for coherence
            "retrieval_path": "Community.Layer"
        }
        for cid, doc, meta in zip(
            all_comms["ids"], all_comms["documents"], all_comms["metadatas"]
        )
    ]
    reports.sort(key=lambda x: x["score"], reverse=True)
    return reports[:k]


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — QUERY ROUTER
#
# Classifies a query as 'thematic', 'entity', or 'hybrid'
# using lightweight lexical heuristics.
# ═══════════════════════════════════════════════════════════════

# Keywords suggesting community/thematic retrieval
_THEMATIC_SIGNALS = [
    "theme", "motif", "throughout", "across", "recurring", "in general",
    "broadly", "pattern", "how does", "what is the role of", "tradition",
    "genre", "compare across", "generally", "all", "most", "common",
    "why does", "what drives", "what connects", "overall"
]

# Keywords suggesting entity/relationship retrieval
_ENTITY_SIGNALS = [
    "character", "who is", "between", "and", "relationship", "compare",
    "versus", "vs", "intertextual", "allusion", "reference", "quotes",
    "influence", "parallel", "like", "similar to", "contrast",
    "specifically", "in the scene where", "when does", "example of"
]


def classify_query(query: str) -> Literal["thematic", "entity", "hybrid"]:
    """
    Stage 3: Query Router
    Returns the retrieval mode for this query.

    thematic  → community pipeline (abstract/thematic questions)
    entity    → entity-relation pipeline (character/intertextual questions)
    hybrid    → both pipelines + RRF fusion
    """
    q = query.lower()
    thematic_score = sum(1 for sig in _THEMATIC_SIGNALS if sig in q)
    entity_score   = sum(1 for sig in _ENTITY_SIGNALS   if sig in q)

    if thematic_score > entity_score:
        return "thematic"
    elif entity_score > thematic_score:
        return "entity"
    else:
        return "hybrid"


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — PIPELINES
# ═══════════════════════════════════════════════════════════════

def _pipeline_thematic(query: str, G: nx.Graph, n_final: int) -> dict:
    """
    Community Pipeline (for abstract / thematic queries):

      Entity.VDB(query)
        → Entity.PPR(seed_works)
        → Community.Entity(seed_entities)  [community-level context]
        → Chunk.Occurrence(entities, ppr)  [supporting passages]

    Optimized for: "How does consciousness manifest in voyages?"
                   "What themes recur across postmodern fiction?"
    """
    # 1. Entity.VDB — get seed chunks
    seed_entities = op_entity_vdb(query, k=DEFAULT_K_ENTITY)

    # 2. Entity.Link — check for explicit work/theme mentions
    linked_nodes  = op_entity_link(query, G)

    # 3. Entity.PPR — propagate relevance through TKG
    seed_works = [_extract_title(e["metadata"].get("source", "")) for e in seed_entities]
    seed_works += linked_nodes
    ppr_scores = op_entity_ppr(seed_works, G)

    # 4. Community.Entity — retrieve community summaries
    communities = op_community_entity(seed_entities, query, k=DEFAULT_K_COMMUNITY)

    # 5. Chunk.Occurrence — rank passages by PPR-boosted co-occurrence
    chunks = op_chunk_occurrence(seed_entities, ppr_scores, k=n_final)

    return {
        "mode": "thematic",
        "chunks": chunks,
        "communities": communities,
        "ppr_scores": dict(sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


def _pipeline_entity(query: str, G: nx.Graph, n_final: int) -> dict:
    """
    Entity-Relation Pipeline (for character / intertextual queries):

      Entity.VDB(query)
        → Entity.Link(query)
        → Relationship.Onehop(entities)
        → Chunk.FromRel(relations)

    Optimized for: "How does Bloom compare to Odysseus?"
                   "What is the intertextual relationship between X and Y?"
    """
    # 1. Entity.VDB — semantic seed retrieval
    seed_entities = op_entity_vdb(query, k=DEFAULT_K_ENTITY)

    # 2. Entity.Link — surface-form linking to TKG nodes
    linked_nodes  = op_entity_link(query, G)

    # 3. Relationship.Onehop — expand to one-hop neighbor edges
    relations = op_relationship_onehop(seed_entities, G)

    # 4. Chunk.FromRel — filter to chunks sitting on relation endpoints
    rel_chunks = op_chunk_from_rel(relations, seed_entities)

    # Merge: prefer FromRel chunks, pad with VDB seed if needed
    seen_ids = {c["id"] for c in rel_chunks}
    for ent in seed_entities:
        if ent["id"] not in seen_ids:
            rel_chunks.append(ent)
            seen_ids.add(ent["id"])

    rel_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "mode": "entity",
        "chunks": rel_chunks[:n_final],
        "communities": [],
        "relations": relations[:20],
    }


def _reciprocal_rank_fusion(
    list_a: list[dict], list_b: list[dict], k: int = 60
) -> list[dict]:
    """
    RRF fusion of two ranked chunk lists.
    Returns merged list sorted by fused score.
    """
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for rank, item in enumerate(list_a):
        scores[item["id"]] += 1 / (k + rank + 1)
        items[item["id"]] = item

    for rank, item in enumerate(list_b):
        scores[item["id"]] += 1 / (k + rank + 1)
        if item["id"] not in items:
            items[item["id"]] = item

    merged = sorted(items.values(), key=lambda x: scores[x["id"]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item["id"]]
    return merged


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    query: str,
    mode: Optional[Literal["thematic", "entity", "hybrid", "auto"]] = "auto",
    n_final: int = DEFAULT_N_FINAL,
) -> dict:
    """
    Stage 4: Retrieval & Generation (retrieval side)

    Args:
        query:   the user question
        mode:    'thematic' | 'entity' | 'hybrid' | 'auto' (default)
                 'auto' runs the query router to decide
        n_final: number of chunks to return for LLM context

    Returns dict with:
        mode:        detected or specified mode
        chunks:      list of chunk dicts for LLM context (n_final items)
        communities: list of community report dicts (thematic mode only)
        graph_info:  diagnostic metadata about the retrieval run
    """
    G = _load_tkg()

    if mode == "auto":
        mode = classify_query(query)

    if mode == "thematic":
        result = _pipeline_thematic(query, G, n_final)

    elif mode == "entity":
        result = _pipeline_entity(query, G, n_final)

    else:  # hybrid
        thematic_r = _pipeline_thematic(query, G, n_final * 2)
        entity_r   = _pipeline_entity(query, G, n_final * 2)
        fused      = _reciprocal_rank_fusion(thematic_r["chunks"], entity_r["chunks"])

        result = {
            "mode": "hybrid",
            "chunks": fused[:n_final],
            "communities": thematic_r.get("communities", []),
            "ppr_scores": thematic_r.get("ppr_scores", {}),
            "relations": entity_r.get("relations", []),
        }

    # ── Build graph_info for UI display ──────────────────────
    result["graph_info"] = {
        "query_mode": result["mode"],
        "tkg_nodes": G.number_of_nodes(),
        "tkg_edges": G.number_of_edges(),
        "chunks_returned": len(result.get("chunks", [])),
        "communities_returned": len(result.get("communities", [])),
        "top_ppr_nodes": list(result.get("ppr_scores", {}).keys())[:5],
    }

    return result


# ═══════════════════════════════════════════════════════════════
# CONTEXT BUILDER — formats retrieval results for LLM
# ═══════════════════════════════════════════════════════════════

def build_context(result: dict) -> str:
    """
    Converts a run_pipeline() result into a formatted context string
    suitable for inclusion in the LLM prompt.
    """
    parts = []

    # Community summaries first (thematic orientation)
    for i, comm in enumerate(result.get("communities", [])):
        genres   = comm["metadata"].get("genres", "")
        themes   = comm["metadata"].get("themes", "")
        score    = f'{comm["score"]:.3f}'
        parts.append(
            f"[Community {i+1} | Genres: {genres} | Themes: {themes} | Relevance: {score}]\n"
            f"{comm['report']}"
        )

    # Passage chunks
    for i, chunk in enumerate(result.get("chunks", [])):
        meta   = chunk["metadata"]
        source = _extract_title(meta.get("source", "unknown"))
        genre  = meta.get("genre", "")
        themes = meta.get("themes", "")
        score  = f'{chunk.get("rrf_score", chunk.get("score", 0)):.3f}'
        path   = chunk.get("retrieval_path", "")
        parts.append(
            f"[Passage {i+1} | Source: {source} | Genre: {genre} | "
            f"Themes: {themes} | Score: {score} | via: {path}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# CLI SMOKE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        G = build_indexes(force_rebuild=True)
        print(f"\nTKG summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print("Work nodes:",  sum(1 for _, d in G.nodes(data=True) if d.get("type") == "work"))
        print("Theme nodes:", sum(1 for _, d in G.nodes(data=True) if d.get("type") == "theme"))
    else:
        query = sys.argv[1] if len(sys.argv) > 1 else "How does consciousness shape the experience of the sea voyage?"
        print(f"Query: {query}")
        mode = classify_query(query)
        print(f"Detected mode: {mode}")

        result = run_pipeline(query)
        print(f"\nGraph info: {json.dumps(result['graph_info'], indent=2)}")
        print(f"\nTop chunks:")
        for c in result["chunks"][:3]:
            print(f"  [{c.get('retrieval_path','')}] {_extract_title(c['metadata'].get('source',''))} — score {c.get('score',0):.3f}")
        if result.get("communities"):
            print(f"\nTop community:")
            print(f"  {result['communities'][0]['report'][:200]}...")
