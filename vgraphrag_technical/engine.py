"""
engine.py (technical)
=======================
Stage 4: Retrieval & Generation — Orchestration for the technical graph

Mirrors vgraphrag/engine.py. Query classification (SPECIFIC vs ABSTRACT) is
reused directly from vgraphrag.query_router — it's a generic binary
classifier with no path dependencies.

Usage:
  from vgraphrag_technical.engine import VGraphRAGTechnicalEngine

  engine = VGraphRAGTechnicalEngine()
  result = engine.query("What does the raw water cooling system require?")
  print(result["answer"])
"""

import re
from pathlib import Path
from typing import Generator, Optional

import anthropic
import networkx as nx
from dotenv import load_dotenv
import os

from vgraphrag.query_router import classify_query, QueryMode

from .graph_builder import load_graph, GRAPH_PATH
from .retriever import run_specific_pipeline, run_abstract_pipeline
from .prompts import GENERATION_SYSTEM

load_dotenv()

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 2048


def _extract_work_title(chunk_id: str) -> str:
    """Derive a human-readable document title from a chunk id."""
    parts = chunk_id.split("__")
    stem = parts[1] if len(parts) > 1 else chunk_id
    return stem.replace("_", " ").strip()


def build_context(retrieval: dict) -> str:
    parts = []
    mode  = retrieval.get("mode", "specific")

    for i, comm in enumerate(retrieval.get("communities", [])):
        score = comm.get("score", 0)
        works = ", ".join(comm.get("works", [])[:8])
        summary = comm.get("summary", "")
        parts.append(
            f"[Community {i+1} | Relevance: {score:.3f} | Docs: {works}]\n"
            f"{summary}"
        )

    rels = retrieval.get("relationships", [])[:12]
    if rels:
        rel_lines = []
        for r in rels:
            kw  = ", ".join(r.get("keywords", []))
            rel_lines.append(
                f"  • {r['src_name']} —[{r['rel_name']}]→ {r['tgt_name']}"
                + (f" ({kw})" if kw else "")
                + (f": {r['description']}" if r.get("description") else "")
            )
        parts.append(
            f"[Key Relationships — retrieved via {rels[0].get('operator', 'aggregator')}]\n"
            + "\n".join(rel_lines)
        )

    for i, chunk in enumerate(retrieval.get("chunks", [])):
        cid   = chunk["chunk_id"]
        work  = _extract_work_title(cid)
        score = chunk.get("score", 0)
        op    = chunk.get("operator", "")

        rel_note = ""
        if mode == "specific" and chunk.get("rels"):
            rel_snippets = [
                f"{r['src_name']}→{r['rel_name']}→{r['tgt_name']}"
                for r in chunk["rels"][:3]
            ]
            rel_note = f" | via: {'; '.join(rel_snippets)}"

        parts.append(
            f"[Passage {i+1} | Source: {work} | Score: {score:.4f} | {op}{rel_note}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


class VGraphRAGTechnicalEngine:
    """
    Main entry point for the technical VGraphRAG pipeline (nautical
    engineering manuals + AI/RAG research papers).

    Initialization loads the technical RKG from disk
    (vgraphrag_technical_db/rkg.json). The three ChromaDB indexes are
    accessed via vgraphrag_technical.index_builder accessors.
    """

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        model: str = MODEL,
        custom_system_prompt: Optional[str] = None,
    ):
        print("Loading technical RKG...")
        self.G: nx.DiGraph = load_graph(graph_path)
        print(f"  {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model  = model
        self.system = custom_system_prompt or GENERATION_SYSTEM

    def retrieve(
        self,
        query: str,
        mode: Optional[QueryMode] = None,
        n_chunks: int = 6,
        n_communities: int = 4,
        use_llm_router: bool = True,
    ) -> dict:
        if mode is None:
            mode = classify_query(query, client=self.client, use_llm=use_llm_router)
        print(f"\nQuery mode: {mode.upper()}")

        if mode == "specific":
            result = run_specific_pipeline(
                query,
                self.G,
                self.client,
                n_chunks=n_chunks,
            )
        else:
            result = run_abstract_pipeline(
                query,
                self.G,
                n_communities=n_communities,
                n_chunks=n_chunks,
            )

        result["graph_info"] = {
            "query":             query,
            "mode":              mode,
            "rkg_nodes":         self.G.number_of_nodes(),
            "rkg_edges":         self.G.number_of_edges(),
            "chunks_returned":   len(result.get("chunks", [])),
            "communities_returned": len(result.get("communities", [])),
            "rels_returned":     len(result.get("relationships", [])),
            "seed_entities":     result.get("seed_entities", []),
            "top_ppr":           list(result.get("ppr_scores", {}).keys())[:8],
        }

        return result

    def query(
        self,
        query: str,
        mode: Optional[QueryMode] = None,
        n_chunks: int = 6,
        n_communities: int = 4,
        use_llm_router: bool = True,
    ) -> dict:
        retrieval = self.retrieve(
            query,
            mode=mode,
            n_chunks=n_chunks,
            n_communities=n_communities,
            use_llm_router=use_llm_router,
        )

        context = build_context(retrieval)

        if not context.strip():
            return {
                "answer":    "No relevant content found in the knowledge graph.",
                "mode":      retrieval.get("mode"),
                "retrieval": retrieval,
                "context":   "",
                "graph_info": retrieval.get("graph_info", {}),
            }

        system = self.system + (
            "\n\nUse ONLY the context documents below to answer. "
            "Community summaries give thematic orientation; "
            "Relationships show structural connections; "
            "Passages provide textual evidence. "
            "Synthesise across all three. Always cite which sources informed your answer."
        )

        messages = [{
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"
        }]

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        )
        answer = msg.content[0].text

        return {
            "answer":     answer,
            "mode":       retrieval.get("mode"),
            "retrieval":  retrieval,
            "context":    context,
            "graph_info": retrieval.get("graph_info", {}),
        }

    def stream(
        self,
        query: str,
        mode: Optional[QueryMode] = None,
        n_chunks: int = 6,
        n_communities: int = 4,
        use_llm_router: bool = True,
    ) -> Generator[dict, None, None]:
        try:
            retrieval = self.retrieve(
                query,
                mode=mode,
                n_chunks=n_chunks,
                n_communities=n_communities,
                use_llm_router=use_llm_router,
            )

            yield {"type": "graph_info", "data": retrieval["graph_info"]}

            yield {
                "type": "sources",
                "data": [
                    {
                        "chunk_id": c["chunk_id"],
                        "work":     _extract_work_title(c["chunk_id"]),
                        "score":    c.get("score", 0),
                        "operator": c.get("operator", ""),
                        "text":     c["text"][:400],
                    }
                    for c in retrieval.get("chunks", [])
                ],
            }

            if retrieval.get("communities"):
                yield {"type": "communities", "data": retrieval["communities"]}

            context = build_context(retrieval)

            if not context.strip():
                yield {"type": "error", "data": "No relevant content found."}
                yield {"type": "done"}
                return

            yield {"type": "context", "data": context[:2000]}

            system = self.system + (
                "\n\nUse ONLY the context documents below to answer. "
                "Community summaries give thematic orientation; "
                "Relationships show structural connections; "
                "Passages provide textual evidence. "
                "Synthesise across all three. Always cite which sources informed your answer."
            )

            messages = [{
                "role": "user",
                "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"
            }]

            with self.client.messages.stream(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield {"type": "token", "data": text}

            yield {"type": "done"}

        except Exception as e:
            yield {"type": "error", "data": str(e)}
