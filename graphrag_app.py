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
import networkx as nx
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
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
        "prompt": """You are a goofball with deep expertise in postmodern fiction, consciousness studies, mythology, creative thinking, and nautical literature. You approach texts with humble intellectual rigor and goofy curiosity.

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
def build_document_graph(docs, metadatas, ids, distances):
    """Build a graph connecting documents by shared metadata attributes."""
    G = nx.Graph()

    for i, doc_id in enumerate(ids):
        G.add_node(doc_id, 
                   text=docs[i],
                   metadata=metadatas[i],
                   distance=distances[i],
                   score=1 - distances[i])

    # Connect nodes that share genre, themes, myth_tradition, or consciousness_technique
    shared_attributes = ["genre", "myth_tradition", "consciousness_technique", "nautical_context", "tone"]

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
                weight = len(shared) / len(shared_attributes)
                G.add_edge(ids[i], ids[j], weight=weight, shared=shared)

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


# ─────────────────────────────────────────────
# GRAPHRAG PIPELINE
# ─────────────────────────────────────────────
def graphrag_query(
    query,
    system_prompt_key="literary_scholar",
    reranker="none",
    n_initial=15,
    n_final=6,
    graph_expansion_depth=1,
    min_distance_threshold=0.8,
    metadata_filter=None,
    custom_system_prompt=None
):
    if not collection:
        return {"error": "ChromaDB not connected"}, []

    # 1. Embed the query
    query_embedding = embed_query(query)

    # 2. Initial retrieval from ChromaDB
    query_params = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_initial, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    if metadata_filter:
        query_params["where"] = metadata_filter

    results = collection.query(**query_params)

    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Filter by distance threshold
    filtered = [(id_, doc, meta, dist) for id_, doc, meta, dist in
                zip(ids, docs, metadatas, distances) if dist < min_distance_threshold]

    if not filtered:
        return {"error": "No relevant documents found. Try a different query or adjust the distance threshold."}, []

    ids, docs, metadatas, distances = zip(*filtered)

    # 3. Build document graph
    G = build_document_graph(list(docs), list(metadatas), list(ids), list(distances))

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

    # 6. Take top N for context
    top_docs = reranked[:n_final]

    # 7. Build graph summary for UI
    graph_info = {
        "initial_retrieved": len(ids),
        "graph_expanded": len(new_ids),
        "total_considered": len(all_docs),
        "final_used": len(top_docs),
        "graph_edges": list(G.edges(data=True))[:10]  # sample edges for display
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


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    query_text          = data.get("query", "").strip()
    system_prompt_key   = data.get("system_prompt", "literary_scholar")
    reranker            = data.get("reranker", "none")
    n_initial           = int(data.get("n_initial", 15))
    n_final             = int(data.get("n_final", 6))
    graph_depth         = int(data.get("graph_depth", 1))
    distance_threshold  = float(data.get("distance_threshold", 0.8))
    custom_prompt       = data.get("custom_prompt", "")
    genre_filter        = data.get("genre_filter", "")

    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    metadata_filter = None
    if genre_filter:
        metadata_filter = {"genre": {"$eq": genre_filter}}

    payload, top_docs = graphrag_query(
        query=query_text,
        system_prompt_key=system_prompt_key,
        reranker=reranker,
        n_initial=n_initial,
        n_final=n_final,
        graph_expansion_depth=graph_depth,
        min_distance_threshold=distance_threshold,
        metadata_filter=metadata_filter,
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


@app.route("/api/genre/<genre>", methods=["GET"])
def get_genre_docs(genre):
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])

    def extract_title(s):
        s = re.sub(r'_\d+$', '', s.strip())
        s = s.split('__')[0]
        return s.replace('_', ' ').strip()

    titles = sorted({
        extract_title(meta.get("source", ""))
        for meta in sample["metadatas"]
        if meta.get("genre", "") == genre and meta.get("source", "").strip()
    })
    return jsonify({"genre": genre, "titles": titles, "count": len(titles)})


@app.route("/api/sources", methods=["GET"])
def get_sources():
    if not collection:
        return jsonify({"error": "No collection"})

    sample = collection.get(limit=collection.count(), include=["metadatas"])

    def extract_title(s):
        s = re.sub(r'_\d+$', '', s.strip())  # strip trailing chunk number
        s = s.split('__')[0]                  # take only the part before __
        return s.replace('_', ' ').strip()

    book_titles = sorted({
        extract_title(meta.get("source", ""))
        for meta in sample["metadatas"]
        if meta.get("source", "").strip()
    })
    return jsonify({"sources": book_titles})


@app.route("/api/title_chunks", methods=["GET"])
def get_title_chunks():
    if not collection:
        return jsonify({"error": "No collection"})

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400

    def extract_title(s):
        s = re.sub(r'_\d+$', '', s.strip())
        s = s.split('__')[0]
        return s.replace('_', ' ').strip()

    all_meta = collection.get(limit=collection.count(), include=["metadatas"])
    matching_ids = [
        all_meta["ids"][i]
        for i, meta in enumerate(all_meta["metadatas"])
        if extract_title(meta.get("source", "")) == title
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