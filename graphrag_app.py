"""
GraphRAG Web Application
Literary RAG engine with graph-based retrieval, reranking, and Claude integration.

Install dependencies:
    pip install flask chromadb openai anthropic cohere networkx python-dotenv sentence-transformers
"""

import os
import re
import json
import time
from pathlib import Path
import numpy as np
import networkx as nx
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from hybrid_pipeline import run_pipeline, build_context, build_indexes

# VGraphRAG engine (loaded lazily on first request)
_vgraphrag_engine = None

def _get_vgraphrag_engine():
    global _vgraphrag_engine
    if _vgraphrag_engine is None:
        try:
            from vgraphrag.engine import VGraphRAGEngine
            _vgraphrag_engine = VGraphRAGEngine()
        except Exception as e:
            print(f"VGraphRAG engine not ready: {e}")
            return None
    return _vgraphrag_engine

# VGraphRAG technical graph — nautical manuals + AI/RAG research papers.
# Loaded read-only (no Anthropic client needed) since the sidebar/graph-modal
# UI only browses the existing graph, it doesn't run queries against it.
_vgtech_graph = None
_vgtech_communities = None

def _get_vgtech_graph():
    global _vgtech_graph
    if _vgtech_graph is None:
        try:
            from vgraphrag_technical.graph_builder import load_graph, GRAPH_PATH
            _vgtech_graph = load_graph(GRAPH_PATH)
        except Exception as e:
            print(f"VGraphRAG technical graph not ready: {e}")
            return None
    return _vgtech_graph

def _get_vgtech_communities():
    global _vgtech_communities
    if _vgtech_communities is None:
        from vgraphrag_technical.index_builder import load_communities
        _vgtech_communities = load_communities()
    return _vgtech_communities

# Cross-graph "bridges" — entities shared between the literary graph and
# the technical graph, curated by build_bridges.py. Static JSON, loaded
# once and served as-is (no per-request graph traversal needed).
_bridges = None

def _get_bridges():
    global _bridges
    if _bridges is None:
        path = Path("vgraphrag_technical_db/bridges.json")
        _bridges = json.loads(path.read_text()) if path.exists() else []
    return _bridges

# Within-corpus "document connections" — entities used as a genuine
# metaphorical vehicle across multiple, genre-distant clean_text works,
# curated by build_document_connections.py. Includes resolved passage
# text per work so the UI can show the actual referenced excerpts.
_document_connections = None

def _get_document_connections():
    global _document_connections
    if _document_connections is None:
        path = Path("vgraphrag_db/document_connections.json")
        _document_connections = json.loads(path.read_text()) if path.exists() else []
    return _document_connections

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────
CHROMA_PATH       = "chroma_db"
COLLECTION_NAME   = "literary_documents"
EMBEDDING_MODEL   = "text-embedding-3-large"
EMBEDDING_DIMS    = 3072
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")
APP_BASE_PATH     = os.getenv("APP_BASE_PATH", "")
CLAUDE_MODEL      = "claude-sonnet-4-6"

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "literary_scholar": {
        "name": "Cordoroyed elbows",
        "description": "Deep textual analysis with symbolic depth expressed casually",
        "prompt": """You are an unctuous goofball with deep curiosity.

When answering questions:
- Draw connections between documents, noting intertextual relationships and thematic resonances
- Attend to narrative technique, prose style, and structural choices
- Reference specific passages and details from the retrieved documents
- Consider the consciousness techniques at play — stream of consciousness, unreliable narrators, fragmented perspectives
- Note mythological archetypes and symbolic dimensions when present
- Maintain scholarly precision while remaining intellectually alive and generative
- If documents contradict or complicate each other, surface that tension rather than resolving it prematurely

Speak with casual authority but remain open to ambiguity. Great literature resists easy interpretation.
- Identify archetypal patterns: the hero's journey, the trickster, the descent and return, the sacred marriage, the world tree
- Draw parallels across mythological traditions — show how the same deep story appears in different cultural clothing
- Connect mythology to consciousness: myths as maps of inner experience, not just outer narrative
- Treat the nautical voyage as archetypal — the sea as the unconscious, the storm as initiation, the harbor as threshold
- Honor both the literal and symbolic registers simultaneously
- Use evocative, imagistic language that mirrors the mythic mode
- Ground your responses in the actual retrieved documents while opening toward their deeper symbolic dimensions

You speak from within the mythic imagination, not merely about it.
- Treat consciousness as the central subject — how does awareness shape, distort, and illuminate experience?
- Draw on phenomenological concepts: perception, intentionality, embodiment, temporality, the lived body
- Connect nautical metaphors to cognitive and creative processes — dead reckoning as intuition, navigation as thought, storm as creative crisis
- Notice how postmodern fragmentation mirrors the actual non-linearity of conscious experience
- Attend to the creative thinking dimensions — where does insight come from? How does imagination navigate uncertainty?
- Synthesize across documents to build a map of related ideas rather than treating each in isolation
- Use precise but evocative language that honors both the analytical and experiential dimensions

Your responses navigate between rigorous thinking and lived, embodied knowing."""
    },

}

# ─────────────────────────────────────────────
# RERANKER OPTIONS
# ─────────────────────────────────────────────
RERANKERS = {
    "none": {
        "name": "No Reranking",
        "description": "Use raw ChromaDB cosine similarity scores"
    },
    "cohere": {
        "name": "Cohere Rerank",
        "description": "Best accuracy — requires Cohere API key"
    },
    "cross_encoder": {
        "name": "Cross-Encoder (Local)",
        "description": "Free, runs locally — good accuracy, slower"
    },
    "reciprocal_rank": {
        "name": "Reciprocal Rank Fusion",
        "description": "Combines vector + keyword scores, no API needed"
    }
}

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"✓ Connected to ChromaDB: {collection.count()} documents")
    except Exception as e:
        print(f"✗ ChromaDB collection not found: {e}")
        collection = None
except Exception as e:
    print(f"✗ ChromaDB client init failed: {e}")
    chroma_client = None
    collection = None

# ingest_pdf.py's nautical_pdfs batch reused genre="nautical" (against its
# own CLI guidance not to reuse literary genres), so a genre-scoped query
# for "nautical" is ~99% technical PDF manuals (diesel/electrical/nav) and
# only ~1% actual clean_text sea-voyage literature. Falling prompts that
# filter by genre are meant to explore the literary corpus, so precompute
# this batch's source labels once at boot and exclude them from those
# genre-scoped queries below.
NAUTICAL_PDF_SOURCES = set()
if collection:
    _all_meta = collection.get(limit=collection.count(), include=["metadatas"])["metadatas"]
    NAUTICAL_PDF_SOURCES = {
        m["source"] for m in _all_meta if (m.get("source") or "").startswith("nautical_pdfs/")
    }
    del _all_meta


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def embed_query(text):
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────
def _domain_key(meta):
    """The 'domain' a chunk belongs to for cross-domain edge scoring — the
    taxonomy's top-level category (NAUTICAL/STORIES/AI/HUMANITY) when a chunk
    has been tagged, else its raw genre for untagged chunks."""
    leaf = meta.get("taxonomy_leaf", "")
    if leaf:
        return TAXONOMY_TOP.get(leaf[0], leaf)
    return meta.get("genre", "")


# How many of a node's nearest-by-embedding neighbors (within the retrieved
# candidate pool) become graph edges. Rank-based rather than a fixed cosine
# cutoff: on real queries the candidate pool is already query-relevant, so
# same-domain and cross-domain pairwise similarities overlap heavily (means
# ~0.5-0.65 either way) — an absolute threshold connects almost everything.
# Per-node top-K neighbors stays sparse and meaningful regardless of where
# that similarity band happens to sit for a given query.
EMBEDDING_EDGE_TOP_K = 5
# Added to a candidate's similarity score, before ranking, when it sits in a
# different domain than the seed node — enough to pull a genuinely close
# cross-domain passage into the top-K even when a same-domain passage is
# marginally closer, without being large enough to promote a weak match.
CROSS_DOMAIN_BONUS = 0.06


def build_document_graph(docs, metadatas, ids, distances, embeddings=None):
    """Build a graph connecting documents by shared metadata attributes and,
    when embeddings are available, by embedding similarity — biased toward
    surfacing cross-domain pairs (e.g. a nautical passage and a consciousness
    passage) rather than only ever linking documents already in the same
    genre/tone/theme bucket."""
    G = nx.Graph()

    for i, doc_id in enumerate(ids):
        G.add_node(doc_id,
                   text=docs[i],
                   metadata=metadatas[i],
                   distance=distances[i],
                   score=1 - distances[i])

    # Connect nodes that share genre, themes, myth_tradition, or consciousness_technique
    shared_attributes = ["genre", "myth_tradition", "consciousness_technique", "nautical_context", "tone"]

    edge_shared = {}   # (i, j) -> shared reasons list
    edge_weight = {}   # (i, j) -> weight

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = []
            for attr in shared_attributes:
                val_i = metadatas[i].get(attr, "")
                val_j = metadatas[j].get(attr, "")
                if val_i and val_j and val_i == val_j:
                    shared.append(attr)

            # Also check theme overlap
            themes_i = set(metadatas[i].get("themes", "").split(", "))
            themes_j = set(metadatas[j].get("themes", "").split(", "))
            theme_overlap = themes_i & themes_j - {""}
            if theme_overlap:
                shared.append(f"themes:{','.join(theme_overlap)}")

            if shared:
                edge_shared[(i, j)] = shared
                edge_weight[(i, j)] = len(shared) / len(shared_attributes)

    # Embedding-similarity edges: for each node, connect to its top-K nearest
    # neighbors by cosine similarity within the candidate pool, ranking
    # cross-domain candidates with a bonus so they can outrank a marginally
    # closer same-domain one.
    if embeddings is not None and len(embeddings) == len(ids):
        emb = np.array(embeddings, dtype=np.float64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = emb / norms
        sim = unit @ unit.T
        domains = [_domain_key(m) for m in metadatas]

        n = len(ids)
        for i in range(n):
            ranked = sorted(
                (j for j in range(n) if j != i),
                key=lambda j: sim[i][j] + (CROSS_DOMAIN_BONUS if domains[j] != domains[i] else 0.0),
                reverse=True,
            )
            for j in ranked[:EMBEDDING_EDGE_TOP_K]:
                key = (i, j) if i < j else (j, i)
                raw_sim = float(sim[i][j])
                cross = domains[i] != domains[j]
                reason = "cross_domain_affinity" if cross else "semantic_similarity"

                existing_shared = edge_shared.get(key, [])
                if reason not in existing_shared:
                    edge_shared[key] = existing_shared + [reason]
                edge_weight[key] = max(edge_weight.get(key, 0.0), raw_sim)

    for (i, j), weight in edge_weight.items():
        G.add_edge(ids[i], ids[j], weight=weight, shared=edge_shared[(i, j)])

    return G


def expand_via_graph(G, seed_ids, expansion_depth=1):
    """Expand retrieved documents by traversing graph neighbors."""
    expanded = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(expansion_depth):
        new_frontier = set()
        for node in frontier:
            if node in G:
                neighbors = dict(G[node])
                # Only include strongly connected neighbors (weight > 0.3)
                strong = {n for n, d in neighbors.items() if d.get("weight", 0) > 0.3}
                new_frontier |= strong - expanded
        expanded |= new_frontier
        frontier = new_frontier

    return list(expanded)


# ─────────────────────────────────────────────
# RERANKERS
# ─────────────────────────────────────────────
def rerank_cohere(query, docs_with_ids):
    """Rerank using Cohere's rerank API."""
    try:
        import cohere
        co = cohere.Client(COHERE_API_KEY)
        texts = [d["text"][:2000] for d in docs_with_ids]
        results = co.rerank(
            query=query,
            documents=texts,
            model="rerank-english-v3.0",
            top_n=len(texts)
        )
        reranked = []
        for r in results.results:
            doc = docs_with_ids[r.index].copy()
            doc["rerank_score"] = r.relevance_score
            reranked.append(doc)
        return reranked
    except Exception as e:
        print(f"Cohere rerank failed: {e}, falling back to original order")
        return docs_with_ids


def rerank_cross_encoder(query, docs_with_ids):
    """Rerank using local cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, d["text"][:512]] for d in docs_with_ids]
        scores = model.predict(pairs)
        for i, doc in enumerate(docs_with_ids):
            doc["rerank_score"] = float(scores[i])
        return sorted(docs_with_ids, key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        print(f"Cross-encoder rerank failed: {e}, falling back")
        return docs_with_ids


def rerank_reciprocal_rank_fusion(query, docs_with_ids):
    """Combine vector similarity + simple keyword matching via RRF."""
    query_words = set(re.findall(r'\w+', query.lower()))
    k = 60  # RRF constant

    for rank, doc in enumerate(docs_with_ids):
        # Vector rank score
        vector_score = 1 / (k + rank + 1)

        # Keyword overlap score
        doc_words = set(re.findall(r'\w+', doc["text"].lower()))
        overlap = len(query_words & doc_words)
        keyword_rank = -overlap  # negative so we can sort ascending
        keyword_score = 1 / (k + overlap + 1) if overlap > 0 else 0

        doc["rerank_score"] = vector_score + keyword_score

    return sorted(docs_with_ids, key=lambda x: x["rerank_score"], reverse=True)


def apply_reranker(reranker_name, query, docs_with_ids):
    if reranker_name == "cohere":
        return rerank_cohere(query, docs_with_ids)
    elif reranker_name == "cross_encoder":
        return rerank_cross_encoder(query, docs_with_ids)
    elif reranker_name == "reciprocal_rank":
        return rerank_reciprocal_rank_fusion(query, docs_with_ids)
    else:
        # No reranking — add placeholder score
        for i, doc in enumerate(docs_with_ids):
            doc["rerank_score"] = 1 - doc.get("distance", 0)
        return docs_with_ids


def _work_key(doc):
    """
    Identify the source "work" (book/paper) a chunk belongs to, so retrieval
    can diversify across distinct works instead of returning several chunks
    from the same book. Mirrors the title-extraction convention used by
    /api/genre and /api/sources: prefer an explicit metadata.title (set by
    the PDF ingestion pipelines), else derive it from the source string
    (clean_text/ books, coalescing "<Title>_<NNN>_<topic>.txt" excerpt files
    back to their shared title).
    """
    meta = doc["metadata"]
    explicit = (meta.get("title") or "").strip()
    if explicit:
        return explicit
    s = (meta.get("source") or doc["id"]).strip()
    if s.endswith(".txt"):
        s = s[:-4]
    if "__" in s:
        s = s.split("__")[0]
    else:
        m = re.match(r'^(.*?)_\d{1,4}_', s)
        if m:
            s = m.group(1)
    return s.replace("_", " ").strip()


def _diversify_by_work(ranked_docs, n_final):
    """Take the best-scoring chunk from each distinct work, in rank order,
    until n_final distinct works are collected (or the pool is exhausted)."""
    picked = []
    seen_works = set()
    for doc in ranked_docs:
        work = _work_key(doc)
        if work in seen_works:
            continue
        seen_works.add(work)
        picked.append(doc)
        if len(picked) >= n_final:
            break
    return picked


def _diversify_by_work_and_genre(ranked_docs, n_final, balance_genres):
    """Like _diversify_by_work, but when multiple genres were explicitly
    requested (a cross-domain comparison prompt), reserve a fair share of
    the final slots for each one first. Otherwise pure relevance score hands
    every slot to whichever genre's embedding happens to sit closest to a
    compound query — e.g. "diesel engine" + "AI" lands overwhelmingly in
    AI-space, so a plain top-N-by-score never lets diesel content through
    even when it's sitting right there in the candidate pool."""
    if not balance_genres or len(balance_genres) <= 1:
        return _diversify_by_work(ranked_docs, n_final)

    quota = max(1, n_final // len(balance_genres))
    picked = []
    seen_works = set()

    for g in balance_genres:
        count = 0
        for doc in ranked_docs:
            if count >= quota or len(picked) >= n_final:
                break
            if doc["metadata"].get("genre") != g:
                continue
            work = _work_key(doc)
            if work in seen_works:
                continue
            seen_works.add(work)
            picked.append(doc)
            count += 1

    if len(picked) < n_final:
        for doc in ranked_docs:
            if len(picked) >= n_final:
                break
            work = _work_key(doc)
            if work in seen_works:
                continue
            seen_works.add(work)
            picked.append(doc)

    picked.sort(key=lambda d: -d.get("rerank_score", 0))
    return picked


# ─────────────────────────────────────────────
# GRAPHRAG PIPELINE
# ─────────────────────────────────────────────
def graphrag_query(
    query,
    system_prompt_key="literary_scholar",
    reranker="none",
    n_initial=120,
    n_final=8,
    graph_expansion_depth=1,
    min_distance_threshold=0.8,
    metadata_filter=None,
    balance_genres=None,
    custom_system_prompt=None
):
    if not collection:
        return {"error": "ChromaDB not connected"}, []

    # 1. Embed the query
    query_embedding = embed_query(query)

    # 2. Initial retrieval from ChromaDB
    if balance_genres and len(balance_genres) > 1:
        # A single pooled nearest-neighbor query lets whichever genre's
        # vocabulary the query text leans toward crowd out the others
        # entirely — e.g. "diesel engine" + "artificial intelligence" in
        # one query comes back 100% one domain or the other, never both,
        # because ranking has no notion of "balance across topics," only
        # "closest overall." Querying each listed genre separately and
        # merging guarantees every genre gets a fair shot at the pool.
        per_genre_n = max(5, n_initial // len(balance_genres))
        ids, docs, metadatas, distances, cand_embeddings = [], [], [], [], []
        seen_ids = set()
        for g in balance_genres:
            # A handful of chunks in the corpus have corrupted embedding
            # records that crash ChromaDB's query plan if they land in the
            # result set (a known pre-existing data issue, not caused by
            # this query). Shrink n_results a couple of times before giving
            # up on this genre entirely — one genre missing from one query
            # beats a 500 for the whole request.
            genre_cond = {"genre": {"$eq": g}}
            if g == "nautical" and NAUTICAL_PDF_SOURCES:
                genre_cond = {"$and": [genre_cond, {"source": {"$nin": list(NAUTICAL_PDF_SOURCES)}}]}

            sub = None
            attempt_n = min(per_genre_n, collection.count())
            for _ in range(3):
                try:
                    sub = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=attempt_n,
                        where=genre_cond,
                        include=["documents", "metadatas", "distances", "embeddings"],
                    )
                    break
                except Exception:
                    attempt_n = attempt_n // 2
                    if attempt_n < 1:
                        break
            if not sub:
                print(f"  balance_genres: skipping genre '{g}' after repeated query errors")
                continue
            for id_, doc, meta, dist, emb in zip(
                sub["ids"][0], sub["documents"][0], sub["metadatas"][0],
                sub["distances"][0], sub["embeddings"][0]
            ):
                if id_ in seen_ids:
                    continue
                seen_ids.add(id_)
                ids.append(id_)
                docs.append(doc)
                metadatas.append(meta)
                distances.append(dist)
                cand_embeddings.append(emb)
    else:
        query_params = dict(
            query_embeddings=[query_embedding],
            n_results=min(n_initial, collection.count()),
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        if metadata_filter:
            query_params["where"] = metadata_filter

        results = collection.query(**query_params)

        ids        = results["ids"][0]
        docs       = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]
        cand_embeddings = results["embeddings"][0]

    # Filter by distance threshold — skipped in balance_genres mode. A single
    # embedding for a compound cross-domain query ("diesel engine" + "AI")
    # doesn't sit equally close to both topics, so a fixed absolute cutoff
    # tuned for single-topic pooled search can wipe out an entire genre's
    # contribution even though genre membership (not raw distance) is what
    # the caller explicitly asked to guarantee here.
    if balance_genres and len(balance_genres) > 1:
        filtered = list(zip(ids, docs, metadatas, distances, cand_embeddings))
    else:
        filtered = [(id_, doc, meta, dist, emb) for id_, doc, meta, dist, emb in
                    zip(ids, docs, metadatas, distances, cand_embeddings) if dist < min_distance_threshold]

    if not filtered:
        return {"error": "No relevant documents found. Try a different query or adjust the distance threshold."}, []

    ids, docs, metadatas, distances, cand_embeddings = zip(*filtered)

    # 3. Build document graph (embedding-similarity edges biased toward
    # cross-domain pairs, alongside the existing metadata-overlap edges)
    G = build_document_graph(list(docs), list(metadatas), list(ids), list(distances), list(cand_embeddings))

    # 4. Expand via graph neighbors
    seed_ids = list(ids)
    expanded_ids = expand_via_graph(G, seed_ids, expansion_depth=graph_expansion_depth)

    # Fetch any newly expanded documents from ChromaDB
    new_ids = [id_ for id_ in expanded_ids if id_ not in ids]
    all_docs_map = {id_: {"id": id_, "text": doc, "metadata": meta, "distance": dist}
                    for id_, doc, meta, dist in zip(ids, docs, metadatas, distances)}

    if new_ids:
        try:
            extra = collection.get(ids=new_ids, include=["documents", "metadatas"])
            for i, eid in enumerate(extra["ids"]):
                if eid not in all_docs_map:
                    all_docs_map[eid] = {
                        "id": eid,
                        "text": extra["documents"][i],
                        "metadata": extra["metadatas"][i],
                        "distance": 0.5,  # estimated distance for graph-expanded docs
                    }
        except Exception as e:
            print(f"Graph expansion fetch error: {e}")

    all_docs = list(all_docs_map.values())

    # 5. Rerank
    reranked = apply_reranker(reranker, query, all_docs)

    # 6. Take top N for context, one chunk per distinct work at most
    top_docs = _diversify_by_work_and_genre(reranked, n_final, balance_genres)

    # 7. Build graph summary for UI
    cross_domain_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if "cross_domain_affinity" in d.get("shared", [])
    ]

    graph_info = {
        "initial_retrieved": len(ids),
        "graph_expanded": len(new_ids),
        "total_considered": len(all_docs),
        "final_used": len(top_docs),
        "cross_domain_links": len(cross_domain_edges),
        "graph_edges": list(G.edges(data=True))[:10],  # sample edges for display
        "cross_domain_edges": sorted(cross_domain_edges, key=lambda e: -e[2].get("weight", 0))[:5],
    }

    # 8. Build context
    context_parts = []
    for i, doc in enumerate(top_docs):
        meta = doc["metadata"]
        source = meta.get("source", "unknown")
        genre = meta.get("genre", "")
        themes = meta.get("themes", "")
        score = doc.get("rerank_score", 0)
        context_parts.append(
            f"[Document {i+1} | Source: {source} | Genre: {genre} | Themes: {themes} | Score: {score:.3f}]\n{doc['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # 9. Build system prompt
    if custom_system_prompt and custom_system_prompt.strip():
        system = custom_system_prompt.strip()
    else:
        system = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["literary_scholar"])["prompt"]

    system += "\n\nUse ONLY the documents provided below to answer. If the answer isn't in the documents, say so clearly. Always reference which documents informed your answer."

    messages = [
        {"role": "user", "content": f"Context documents:\n\n{context}\n\n---\n\nQuestion: {query}"}
    ]

    return {"system": system, "messages": messages, "graph_info": graph_info, "top_docs": top_docs}, top_docs


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           system_prompts=SYSTEM_PROMPTS,
                           rerankers=RERANKERS,
                           doc_count=collection.count() if collection else 0,
                           base_path=APP_BASE_PATH)


# Maps a chunk's metadata["source"] collection-slug prefix (e.g. "nautical_pdfs/anchor_bend")
# to the directory the original PDF lives in, so citations can link back to it.
PDF_SOURCE_DIRS = {
    "nautical_pdfs": "templates/pdfs",
    "ai_pdfs": "templates/pdfs/ai",
}


# A few clean_text/ collections split into per-excerpt files with a topic
# slug but no numeric separator (e.g. "This is Story of Happy
# Marriage_lapd.txt"), so the "_<NNN>_" coalescing regex below can't detect
# where the title ends. Listed explicitly as they're found.
# Maps a matched source prefix to its canonical display title — usually
# identical, but sometimes a normalization (e.g. a hyphen-inconsistent
# variant folded into the same title as the rest of that book's chunks).
KNOWN_COLLECTION_TITLES = {
    "This is Story of Happy Marriage": "This is Story of Happy Marriage",
    "Making Shapely Fiction": "Making Shapely Fiction",
    "Writers Art": "Writers Art",
    "Boating Skills and Seamanship": "Boating Skills and Seamanship",
    "Cinematography": "Cinematography",
    "Conversations of Goethe": "Conversations of Goethe",
    "Creators on Creating": "Creators on Creating",
    "Global Hesychasm": "Global Hesychasm",
    "IF Handbook": "IF Handbook",
    "In Recognition of William Gaddis": "In Recognition of William Gaddis",
    "Perl": "Perl",
    "Philokalia": "Philokalia",
    "Scientific American Creativity": "Scientific American Creativity",
    "Texas Handbook": "Texas Handbook",
    "The Legacy of DFW": "The Legacy of DFW",
    "The Letters of Vincent Van Gogh": "The Letters of Vincent Van Gogh",
    "The Mystical Theology of Saint Bernard": "The Mystical Theology of Saint Bernard",
    "Plot and Structure": "Plot and Structure",
    "The Self-Conscious Novel": "The Self Conscious Novel",  # normalize hyphen variant into the Stonehill group
    "Vector Narratives": "Vector Narratives",
    "The Rhetoric of Fiction": "The Rhetoric of Fiction",
}


def _extract_title_from_source(s):
    s = (s or "").strip()
    if s.endswith('.txt'):
        s = s[:-4]
    for known, canonical in KNOWN_COLLECTION_TITLES.items():
        if s == known or s.startswith(known + "_"):
            return canonical
    if '__' in s:
        s = s.split('__')[0]
    else:
        # Many clean_text/ books are split across many small per-excerpt
        # files named "<Title>_<NNN>_<topic slug>.txt" (e.g. "The Letters
        # of Vincent Van Gogh_094_win_nature_over.txt"). Coalesce these back
        # to their shared book title instead of showing each excerpt as its
        # own entry.
        m = re.match(r'^(.*?)_\d{1,4}_', s)
        if m:
            s = m.group(1)
    return s.replace('_', ' ').strip()


def _display_title(meta):
    explicit = (meta.get("title") or "").strip()
    if explicit:
        return explicit
    return _extract_title_from_source(meta.get("source", ""))


def _author_from_source(s):
    """Best-effort last-name extraction for clean_text/ books using the
    "<Title>__<Author>_<slug>.txt" convention (author names use a literal
    space for multi-word surnames, e.g. "Van Dusen", so splitting on the
    first underscore after "__" isolates just the name). Returns None for
    PDF-backed/known-collection sources that carry no author segment.
    """
    s = (s or "").strip()
    if s.endswith('.txt'):
        s = s[:-4]
    if '__' not in s:
        return None
    rest = s.split('__', 1)[1]
    author = rest.split('_')[0].strip()
    return author or None


# The taxonomy the user hand-designed to replace genre/topic browsing in the
# sidebar. Leaf codes are written onto chunk metadata as "taxonomy_leaf" by
# the one-off scratchpad classification pass (build_taxonomy.py) — chunks
# with no confident leaf are left untagged and simply don't appear here.
TAXONOMY_TOP = {"1": "NAUTICAL", "2": "STORIES", "3": "AI", "4": "HUMANITY"}
TAXONOMY_LEAF_LABELS = {
    "1a": "Diesel Maintenance", "1b": "Electrical", "1c": "Navigation",
    "2a": "Postmodern Fiction", "2b": "American Classics", "2c": "Historical Classics",
    "2d": "Craft", "2e": "Theory",
    "3a": "Conversational Design", "3b": "RAG / GraphRAG", "3c": "AI General Studies",
    "3d": "Pre-AI Computer Science",
    "4a": "Art", "4b": "Consciousness", "4c": "Myth", "4d": "Creativity",
}


@app.route("/pdfs/<slug>/<path:filename>")
def serve_source_pdf(slug, filename):
    pdf_dir = PDF_SOURCE_DIRS.get(slug)
    if not pdf_dir:
        return jsonify({"error": "unknown source collection"}), 404
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".pdf"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(pdf_dir, safe_name)


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    query_text          = data.get("query", "").strip()
    system_prompt_key   = data.get("system_prompt", "literary_scholar")
    reranker            = data.get("reranker", "none")
    n_initial           = int(data.get("n_initial", 120))
    n_final             = int(data.get("n_final", 8))
    graph_depth         = int(data.get("graph_depth", 1))
    distance_threshold  = float(data.get("distance_threshold", 0.8))
    custom_prompt       = data.get("custom_prompt", "")
    genre_filter        = data.get("genre_filter", "")

    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Deasy Labs content (genre="reference") should never surface as a
    # source/citation for any prompt or search — it's AI-infrastructure
    # marketing copy, not corpus material meant to be quoted back to users.
    exclude_reference = {"genre": {"$ne": "reference"}}
    balance_genres = None
    if genre_filter:
        if isinstance(genre_filter, list):
            genre_cond = {"genre": {"$in": genre_filter}}
            if len(genre_filter) > 1:
                balance_genres = genre_filter
        else:
            genre_cond = {"genre": {"$eq": genre_filter}}
        conditions = [genre_cond, exclude_reference]
        genres_requested = genre_filter if isinstance(genre_filter, list) else [genre_filter]
        if "nautical" in genres_requested and NAUTICAL_PDF_SOURCES:
            conditions.append({"source": {"$nin": list(NAUTICAL_PDF_SOURCES)}})
        metadata_filter = {"$and": conditions}
    else:
        metadata_filter = exclude_reference

    payload, top_docs = graphrag_query(
        query=query_text,
        system_prompt_key=system_prompt_key,
        reranker=reranker,
        n_initial=n_initial,
        n_final=n_final,
        graph_expansion_depth=graph_depth,
        min_distance_threshold=distance_threshold,
        metadata_filter=metadata_filter,
        balance_genres=balance_genres,
        custom_system_prompt=custom_prompt
    )

    if "error" in payload:
        return jsonify(payload), 400

    # Stream response from Claude
    def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            # First yield graph info and sources
            yield f"data: {json.dumps({'type': 'graph_info', 'data': payload['graph_info']})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'data': [{'id': d['id'], 'metadata': d['metadata'], 'score': d.get('rerank_score', 0), 'text': d['text']} for d in top_docs]})}\n\n"

            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=payload["system"],
                messages=payload["messages"]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'token', 'data': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "system_prompts": SYSTEM_PROMPTS,
        "rerankers": RERANKERS,
        "doc_count": collection.count() if collection else 0,
        "genres": ["postmodern_fiction", "consciousness_studies", "mythology",
                   "nautical", "creative_thinking", "poetry", "drama", "prose"]
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    genre_counts = {}
    theme_counts = {}

    for meta in sample["metadatas"]:
        g = meta.get("genre", "unknown")
        genre_counts[g] = genre_counts.get(g, 0) + 1
        for theme in meta.get("themes", "").split(", "):
            if theme:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

    return jsonify({
        "total": collection.count(),
        "genres": genre_counts,
        "themes": dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    })


@app.route("/api/topics", methods=["GET"])
def get_topics():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    tree = {}
    for meta in sample["metadatas"]:
        genre = meta.get("genre", "unknown")
        topic = meta.get("topic", "")
        if not topic:
            continue
        tree.setdefault(genre, {})
        tree[genre][topic] = tree[genre].get(topic, 0) + 1

    # sort topics within each genre by count desc
    for genre in tree:
        tree[genre] = dict(sorted(tree[genre].items(), key=lambda x: -x[1]))

    return jsonify({"genres": tree})


@app.route("/api/genre/<genre>", methods=["GET"])
def get_genre_docs(genre):
    if not collection:
        return jsonify({"error": "No collection"})

    topic = request.args.get("topic", "")
    sample = collection.get(limit=collection.count(), include=["metadatas"])

    def matches(meta):
        if meta.get("genre", "") != genre:
            return False
        if topic and meta.get("topic", "") != topic:
            return False
        return bool(meta.get("source", "").strip())

    # Group chunks by their display title (metadata.title when the ingestion
    # pipeline set one — nautical_pdfs/ai_pdfs/jait2010/deasylabs all do —
    # else derived from the raw source string), tracking every distinct raw
    # source that maps to it. A title is "PDF-backed" only when it comes from
    # exactly one raw source and that source belongs to a known PDF-backed
    # collection (see PDF_SOURCE_DIRS) — clean_text books span many
    # chunks/sources under one title and should keep opening the text carousel.
    pdf_prefixes = tuple(f"{slug}/" for slug in PDF_SOURCE_DIRS)
    title_to_sources = {}
    for meta in sample["metadatas"]:
        if not matches(meta):
            continue
        src = meta.get("source", "").strip()
        title = _display_title(meta)
        title_to_sources.setdefault(title, set()).add(src)

    docs = []
    for title, sources in title_to_sources.items():
        pdf_source = None
        if len(sources) == 1:
            only = next(iter(sources))
            if only.startswith(pdf_prefixes):
                pdf_source = only
        docs.append({"title": title, "pdf_source": pdf_source})
    docs.sort(key=lambda d: d["title"])

    return jsonify({"genre": genre, "topic": topic, "titles": docs, "count": len(docs)})


@app.route("/api/taxonomy", methods=["GET"])
def get_taxonomy():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    tree = {}
    for meta in sample["metadatas"]:
        leaf = meta.get("taxonomy_leaf", "")
        if not leaf:
            continue
        top = TAXONOMY_TOP.get(leaf[0], "OTHER")
        label = TAXONOMY_LEAF_LABELS.get(leaf, leaf)
        key = f"{leaf} {label}"
        tree.setdefault(top, {})
        tree[top][key] = tree[top].get(key, 0) + 1

    for top in tree:
        tree[top] = dict(sorted(tree[top].items(), key=lambda x: x[0]))

    return jsonify({"tree": tree, "top_order": ["NAUTICAL", "STORIES", "AI", "HUMANITY"]})


# clean_text/ chunks (tagged by embed.py's Claude Haiku tagger, real theme
# vocabulary) are identifiable by their source string having no "/" —
# unlike PDF-batch chunks (nautical_pdfs/..., ai_pdfs/..., jait2010/...,
# deasylabs/...), which reuse this same "themes" field for their own
# unrelated free-text CLI tags ("boating", "ai", "retrieval", "wimax", ...).
# Genre alone can't make this distinction since ingest_pdf.py can also
# assign genre="nautical" to a PDF-batch chunk, colliding with embed.py's
# own genre=nautical clean_text chunks. Discovering theme values
# dynamically this way (rather than a hardcoded word list) also surfaces
# every literary sub-theme actually present in the data — the corpus
# carries ~76 distinct values, not just the original 10-word canonical set.
def _is_clean_text_source(meta):
    return "/" not in (meta.get("source") or "")


@app.route("/api/themes", methods=["GET"])
def get_themes():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    counts = {}
    for meta in sample["metadatas"]:
        if not _is_clean_text_source(meta):
            continue
        for word in (meta.get("themes") or "").split(","):
            word = word.strip()
            if word:
                counts[word] = counts.get(word, 0) + 1

    ordered = sorted(counts.items(), key=lambda x: -x[1])
    return jsonify({"themes": ordered})


@app.route("/api/themes/<theme>", methods=["GET"])
def get_theme_docs(theme):
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    cooccur = {}
    n_chunks = 0
    found = False
    for meta in sample["metadatas"]:
        if not _is_clean_text_source(meta):
            continue
        words = {w.strip() for w in (meta.get("themes") or "").split(",") if w.strip()}
        if theme not in words:
            continue
        found = True
        n_chunks += 1
        for other in words:
            if other == theme:
                continue
            cooccur[other] = cooccur.get(other, 0) + 1

    if not found:
        return jsonify({"error": "unknown theme"}), 404

    ordered = sorted(cooccur.items(), key=lambda x: -x[1])
    return jsonify({"theme": theme, "n_chunks": n_chunks, "cooccurring_themes": ordered})


@app.route("/api/taxonomy/<leaf>", methods=["GET"])
def get_taxonomy_docs(leaf):
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])
    pdf_prefixes = tuple(f"{slug}/" for slug in PDF_SOURCE_DIRS)
    title_to_sources = {}
    for meta in sample["metadatas"]:
        if meta.get("taxonomy_leaf", "") != leaf:
            continue
        src = meta.get("source", "").strip()
        if not src:
            continue
        title = _display_title(meta)
        title_to_sources.setdefault(title, set()).add(src)

    docs = []
    for title, sources in title_to_sources.items():
        pdf_source = None
        if len(sources) == 1:
            only = next(iter(sources))
            if only.startswith(pdf_prefixes):
                pdf_source = only
        author = _author_from_source(next(iter(sources)))
        docs.append({"title": title, "pdf_source": pdf_source, "author": author})
    docs.sort(key=lambda d: d["title"])

    label = TAXONOMY_LEAF_LABELS.get(leaf, leaf)
    return jsonify({"leaf": leaf, "label": label, "titles": docs, "count": len(docs)})


@app.route("/api/sources", methods=["GET"])
def get_sources():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])

    title_to_source = {}
    for meta in sample["metadatas"]:
        src = meta.get("source", "").strip()
        if not src or meta.get("genre", "") == "reference":
            continue
        title_to_source.setdefault(_display_title(meta), src)

    entries = []
    for title, src in title_to_source.items():
        author = _author_from_source(src)
        entries.append((title, f"{title} | {author}" if author else title))
    entries.sort(key=lambda x: x[0])

    return jsonify({"sources": [label for _, label in entries]})


@app.route("/api/title_chunks", methods=["GET"])
def get_title_chunks():
    if not collection:
        return jsonify({"error": "No collection"})

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400

    all_meta = collection.get(limit=collection.count(), include=["metadatas"])
    matching_ids = [
        all_meta["ids"][i]
        for i, meta in enumerate(all_meta["metadatas"])
        if _display_title(meta) == title
    ]

    if not matching_ids:
        return jsonify({"title": title, "chunks": [], "count": 0})

    result = collection.get(ids=matching_ids, include=["documents", "metadatas"])

    chunks = [
        {"id": doc_id, "text": result["documents"][i], "metadata": result["metadatas"][i]}
        for i, doc_id in enumerate(result["ids"])
    ]
    chunks.sort(key=lambda x: x["id"])

    return jsonify({"title": title, "chunks": chunks, "count": len(chunks)})


@app.route("/api/query_hybrid", methods=["POST"])
def query_hybrid():
    """
    Hybrid vgraphRAG endpoint.

    Uses the four-stage operator pipeline from hybrid_pipeline.py:
      auto   → query router decides (thematic / entity / hybrid)
      thematic → Community pipeline: Entity.VDB → PPR → Community.Entity + Chunk.Occurrence
      entity   → Relation pipeline:  Entity.VDB → Entity.Link → Relationship.Onehop → Chunk.FromRel
      hybrid   → Both pipelines fused with Reciprocal Rank Fusion
    """
    data = request.json
    query_text        = data.get("query", "").strip()
    pipeline_mode     = data.get("pipeline_mode", "auto")   # auto|thematic|entity|hybrid
    n_final           = int(data.get("n_final", 6))
    system_prompt_key = data.get("system_prompt", "literary_scholar")
    custom_prompt     = data.get("custom_prompt", "")

    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        result = run_pipeline(query_text, mode=pipeline_mode, n_final=n_final)
    except RuntimeError as e:
        return jsonify({"error": str(e) + " — run python hybrid_pipeline.py build first"}), 500

    context = build_context(result)

    if custom_prompt and custom_prompt.strip():
        system = custom_prompt.strip()
    else:
        system = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["literary_scholar"])["prompt"]

    system += (
        "\n\nUse ONLY the documents provided below to answer. "
        "Community summaries give thematic orientation; passages give textual evidence. "
        "Synthesize across both. Always reference which sources informed your answer."
    )

    messages = [
        {"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query_text}"}
    ]

    def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            yield f"data: {json.dumps({'type': 'graph_info', 'data': result['graph_info']})}\n\n"

            sources = [
                {"id": c["id"], "metadata": c["metadata"],
                 "score": c.get("rrf_score", c.get("score", 0)),
                 "text": c["text"], "retrieval_path": c.get("retrieval_path", "")}
                for c in result.get("chunks", [])
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            if result.get("communities"):
                yield f"data: {json.dumps({'type': 'communities', 'data': result['communities']})}\n\n"

            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'token', 'data': text})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/query_vgraphrag", methods=["POST"])
def query_vgraphrag():
    """
    VGraphRAG endpoint — full RKG pipeline.
    Specific QA:  Entity.Link → Entity.PPR → Rel.Aggregator → Chunk.Aggregator
    Abstract QA:  Community.VDB → Node.VDB → Rel.Onehop
    Requires: python build_vgraphrag.py build
    """
    engine = _get_vgraphrag_engine()
    if engine is None:
        return jsonify({"error": "VGraphRAG indexes not built. Run: python build_vgraphrag.py build"}), 503

    data          = request.json
    query_text    = data.get("query", "").strip()
    mode          = data.get("mode") or None          # None → auto-route
    n_chunks      = int(data.get("n_chunks", 6))
    n_communities = int(data.get("n_communities", 4))
    custom_prompt = data.get("custom_prompt", "")

    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    if custom_prompt:
        engine.system = custom_prompt

    def generate():
        try:
            for event in engine.stream(
                query_text,
                mode=mode,
                n_chunks=n_chunks,
                n_communities=n_communities,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/vgraphrag_technical/ontology", methods=["GET"])
def get_vgtech_ontology():
    """Entity-type and relationship-type counts for the technical graph."""
    G = _get_vgtech_graph()
    if G is None:
        return jsonify({"nodes": 0, "edges": 0, "node_types": {}, "rel_types": {}})

    from vgraphrag_technical.graph_builder import graph_stats
    return jsonify(graph_stats(G))


@app.route("/api/vgraphrag_technical/entity_type/<etype>", methods=["GET"])
def get_vgtech_entity_type(etype):
    """Top entities of one type, by degree, for the sidebar drill-down."""
    G = _get_vgtech_graph()
    if G is None:
        return jsonify({"entities": []})

    matches = [
        {"name": d.get("name", k), "description": d.get("description", ""), "degree": G.degree(k)}
        for k, d in G.nodes(data=True)
        if d.get("type", "CONCEPT") == etype
    ]
    matches.sort(key=lambda e: e["degree"], reverse=True)
    return jsonify({"entities": matches[:25]})


@app.route("/api/vgraphrag_technical/communities", methods=["GET"])
def get_vgtech_communities():
    """All Leiden communities in the technical graph, largest first."""
    comms = _get_vgtech_communities()
    out = [{
        "id":        c["id"],
        "size":      c["size"],
        "top_types": c["top_types"],
        "works":     c["works"][:8],
        "summary":   c["summary"],
    } for c in comms]
    out.sort(key=lambda c: c["size"], reverse=True)
    return jsonify({"communities": out})


@app.route("/api/vgraphrag_technical/community/<int:comm_id>/graph", methods=["GET"])
def get_vgtech_community_graph(comm_id):
    """
    Node-link JSON for one community, capped to the top entities by
    in-community degree — a full community (up to ~700 entities) is
    unreadable as a force layout, so the modal shows the most connected
    subset and reports the true size alongside it.
    """
    G = _get_vgtech_graph()
    comms = _get_vgtech_communities()
    comm = next((c for c in comms if c["id"] == comm_id), None)
    if G is None or comm is None:
        return jsonify({"nodes": [], "links": [], "total_size": 0}), 404

    LIMIT = 70
    node_keys = [k for k in comm["node_keys"] if G.has_node(k)]
    sub = G.subgraph(node_keys)

    kept = sorted(sub.nodes(), key=lambda k: sub.degree(k), reverse=True)[:LIMIT]
    kept_set = set(kept)
    view = sub.subgraph(kept)

    nodes = [{
        "id":     k,
        "name":   view.nodes[k].get("name", k),
        "type":   view.nodes[k].get("type", "CONCEPT"),
        "degree": view.degree(k),
    } for k in view.nodes()]

    links = [{
        "source": u,
        "target": v,
        "name":   d.get("name", "related_to"),
        "weight": d.get("weight", 0.5),
    } for u, v, d in view.edges(data=True)]

    return jsonify({
        "nodes": nodes,
        "links": links,
        "total_size": comm["size"],
        "shown": len(nodes),
    })


@app.route("/api/bridges", methods=["GET"])
def get_bridges():
    """
    Entities shared between the literary graph and the technical graph —
    curated by build_bridges.py to surface genuine cross-corpus connections
    (same real-world referent, treated literally in one and thematically
    in the other) rather than coincidental name collisions.
    """
    return jsonify({"bridges": _get_bridges()})


@app.route("/api/document_connections", methods=["GET"])
def get_document_connections():
    """
    Entities used as a genuine metaphorical vehicle across multiple,
    genre-distant clean_text works — curated by build_document_connections.py.
    Each entry includes resolved passage text for every referenced work so
    the UI can show the actual excerpts, not just a paraphrase.
    """
    return jsonify({"connections": _get_document_connections()})


@app.route("/api/build_indexes", methods=["POST"])
def api_build_indexes():
    """Build or rebuild the TKG and Community Index from loaded chunks."""
    force = request.json.get("force", False) if request.json else False
    try:
        G = build_indexes(force_rebuild=force)
        return jsonify({
            "status": "ok",
            "tkg_nodes": G.number_of_nodes(),
            "tkg_edges": G.number_of_edges(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


FEEDBACK_PATH = Path("vgraphrag_db/feedback.jsonl")

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Store a user highlight as a labeled pair for future training."""
    data = request.get_json(silent=True) or {}
    query          = data.get("query", "").strip()
    highlighted    = data.get("highlighted_text", "").strip()
    sources        = data.get("sources", [])

    if not query or not highlighted:
        return jsonify({"error": "missing query or highlighted_text"}), 400

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "highlighted_text": highlighted,
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)