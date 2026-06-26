"""
engine.py
=========
Stage 4: Retrieval & Generation — Orchestration

VGraphRAGEngine ties everything together:
  1. Classify query (query_router)
  2. Run the appropriate operator pipeline (retriever)
  3. Build a structured context string
  4. Stream or return a Claude generation

Usage:
  from vgraphrag.engine import VGraphRAGEngine

  engine = VGraphRAGEngine()
  for chunk in engine.stream("How does Bloom parallel Odysseus?"):
      print(chunk, end="", flush=True)

  # Or non-streaming:
  result = engine.query("What themes connect Gaddis and Wallace?")
  print(result["answer"])
"""

import json
import re
from pathlib import Path
from typing import Generator, Optional

import anthropic
import networkx as nx
from dotenv import load_dotenv
import os

from .graph_builder import load_graph, GRAPH_PATH
from .query_router import classify_query, QueryMode
from .retriever import run_specific_pipeline, run_abstract_pipeline
from .prompts import GENERATION_SYSTEM

load_dotenv()

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 2048


# ─────────────────────────────────────────────────────────────
# CONTEXT BUILDER
#
# Formats retrieval results into a single context string for
# the LLM.  Structure differs by pipeline mode.
# ─────────────────────────────────────────────────────────────

def _extract_work_title(chunk_id: str) -> str:
    """Derive a human-readable work title from a chunk file stem."""
    s = chunk_id
    if "__" in s:
        s = s.split("__")[0]
    s = re.sub(r'_\d+$', '', s)
    return s.replace("_", " ").strip()


def build_context(retrieval: dict) -> str:
    """
    Build the LLM context string from a retrieval pipeline result.

    For SPECIFIC mode: relationships + passage chunks
    For ABSTRACT mode: community summaries + passage chunks
    """
    parts = []
    mode  = retrieval.get("mode", "specific")

    # ── Community summaries (abstract mode) ─────────────────
    for i, comm in enumerate(retrieval.get("communities", [])):
        score = comm.get("score", 0)
        works = ", ".join(comm.get("works", [])[:8])
        summary = comm.get("summary", "")
        parts.append(
            f"[Community {i+1} | Relevance: {score:.3f} | Works: {works}]\n"
            f"{summary}"
        )

    # ── Key relationships (both modes) ──────────────────────
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

    # ── Passage chunks ───────────────────────────────────────
    for i, chunk in enumerate(retrieval.get("chunks", [])):
        cid   = chunk["chunk_id"]
        work  = _extract_work_title(cid)
        score = chunk.get("score", 0)
        op    = chunk.get("operator", "")

        # Summarise contributing relations for specific mode
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


# ─────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────

class VGraphRAGEngine:
    """
    Main entry point for VGraphRAG query execution.

    Initialization loads the RKG from disk (vgraphrag_db/rkg.json).
    The three ChromaDB indexes are accessed via the index_builder
    accessors (they stay persistent on disk).
    """

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        model: str = MODEL,
        custom_system_prompt: Optional[str] = None,
    ):
        print("Loading RKG...")
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
        """
        Run the full retrieval pipeline without generating a response.

        Returns the raw retrieval dict (chunks, relationships, communities,
        ppr_scores, seed_entities, mode, graph_info).
        Useful for inspection, evaluation, or custom generation.
        """
        # Route
        if mode is None:
            mode = classify_query(query, client=self.client, use_llm=use_llm_router)
        print(f"\nQuery mode: {mode.upper()}")

        # Retrieve
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

        # Diagnostic info for UI / logging
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
        """
        Full pipeline: retrieve + generate (blocking, returns complete answer).

        Returns:
          answer:       LLM response string
          mode:         'specific' or 'abstract'
          retrieval:    raw retrieval dict
          context:      the context string passed to the LLM
          graph_info:   diagnostic metadata
        """
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
        """
        Full pipeline: retrieve + stream generation.

        Yields SSE-style dicts with 'type' key:
          {"type": "graph_info",  "data": {...}}
          {"type": "sources",     "data": [...]}
          {"type": "communities", "data": [...]}
          {"type": "context",     "data": "..."}
          {"type": "token",       "data": "..."}   ← streamed tokens
          {"type": "done"}
          {"type": "error",       "data": "..."}
        """
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

            yield {"type": "context", "data": context[:2000]}  # preview for UI

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
