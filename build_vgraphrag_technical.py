"""
build_vgraphrag_technical.py
=============================
CLI script to build the technical VGraphRAG pipeline (nautical engineering
manuals in templates/pdfs/ + AI/RAG research papers in templates/pdfs/ai/)
from scratch. Mirrors build_vgraphrag.py, with an added PDF-ingestion stage.

Usage:
  # Full build (all stages):
  python build_vgraphrag_technical.py

  # Limit graph extraction to first N chunks (for testing):
  python build_vgraphrag_technical.py --limit 20

  # Skip PDF ingestion (chunks already on disk), just (re)build the graph:
  python build_vgraphrag_technical.py --skip-ingest

  # Resume interrupted graph build, then rebuild indexes:
  python build_vgraphrag_technical.py --skip-ingest --skip-graph --rebuild-indexes

  # Force-rebuild everything:
  python build_vgraphrag_technical.py --force

  # Show graph stats for an existing graph:
  python build_vgraphrag_technical.py --stats

  # Test a query against a built index:
  python build_vgraphrag_technical.py --query "What does the raw water system require?"
  python build_vgraphrag_technical.py --query "How do retrieval-augmented systems evaluate against baselines?" --mode abstract

Stages:
  0. PDF Ingestion    → clean_text_technical/*.txt
  1. Graph Building   → vgraphrag_technical_db/rkg.json
  2. Node Index       → vgraphrag_technical_db/chroma/vgraphrag_tech_nodes
  3. Relationship Index → vgraphrag_technical_db/chroma/vgraphrag_tech_relationships
  4. Community Index  → vgraphrag_technical_db/chroma/vgraphrag_tech_communities
                      + vgraphrag_technical_db/communities.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vgraphrag_technical.graph_builder import (
    ingest_pdfs_to_chunks, build_graph, load_graph, graph_stats,
    CHUNK_DIR, GRAPH_PATH,
)
from vgraphrag_technical.index_builder import build_all_indexes
from vgraphrag_technical.engine import VGraphRAGTechnicalEngine


def cmd_build(args):
    if args.skip_ingest:
        print("Skipping PDF ingestion (--skip-ingest).")
    else:
        print("=" * 60)
        print("STAGE 0 — PDF Ingestion")
        print(f"  Output: {CHUNK_DIR}")
        print("=" * 60)
        n = ingest_pdfs_to_chunks(force=args.force)
        print(f"\n{n} PDF(s) (re)processed into chunks.")

    if args.skip_graph:
        print("Skipping graph build (--skip-graph).")
    else:
        print("\n" + "=" * 60)
        print("STAGE 1 — Graph Building")
        print(f"  Source: {CHUNK_DIR}")
        print(f"  Output: {GRAPH_PATH}")
        if args.limit:
            print(f"  Limit:  {args.limit} chunks")
        print("=" * 60)

        G = build_graph(
            chunk_dir=CHUNK_DIR,
            limit=args.limit,
            resume=not args.force,
        )
        print(f"\nGraph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n" + "=" * 60)
    print("STAGES 2–4 — Index Building")
    print("=" * 60)
    build_all_indexes(graph_path=GRAPH_PATH, force=args.force or args.rebuild_indexes)


def cmd_stats(args):
    if not GRAPH_PATH.exists():
        print(f"No graph found at {GRAPH_PATH}. Run build first.")
        sys.exit(1)

    G = load_graph(GRAPH_PATH)
    stats = graph_stats(G)

    print(f"\nGraph: {stats['nodes']} nodes, {stats['edges']} edges")
    print("\nNode types:")
    for t, count in stats["node_types"].items():
        print(f"  {t:<30} {count}")

    print("\nTop relationship types:")
    for r, count in list(stats["rel_types"].items())[:15]:
        print(f"  {r:<35} {count}")

    print("\nTop 20 entities by degree:")
    for e in stats["top_entities"]:
        print(f"  [{e['type']:<25}] deg={e['degree']:<4} {e['name']}")


def cmd_query(args):
    if not GRAPH_PATH.exists():
        print("No graph found. Run build first.")
        sys.exit(1)

    engine = VGraphRAGTechnicalEngine()
    result = engine.query(
        query=args.query,
        mode=args.mode,
        n_chunks=args.n_chunks,
        use_llm_router=True,
    )

    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print(f"Mode:  {result['mode'].upper()}")
    print("=" * 60)

    gi = result["graph_info"]
    print(f"\nGraph info:")
    print(f"  Chunks returned:      {gi['chunks_returned']}")
    print(f"  Communities returned: {gi['communities_returned']}")
    print(f"  Relationships:        {gi['rels_returned']}")
    if gi.get("seed_entities"):
        print(f"  Seed entities:        {gi['seed_entities'][:6]}")
    if gi.get("top_ppr"):
        print(f"  Top PPR nodes:        {gi['top_ppr'][:5]}")

    if result.get("retrieval", {}).get("communities"):
        print("\nCommunity summaries retrieved:")
        for i, c in enumerate(result["retrieval"]["communities"][:2]):
            print(f"\n  [{i+1}] Score={c['score']:.3f} | "
                  f"Docs: {', '.join(c.get('works', [])[:4])}")
            print(f"  {c['summary'][:300]}...")

    print("\nAnswer:")
    print("-" * 60)
    print(result["answer"])


def main():
    parser = argparse.ArgumentParser(
        description="Build and query the technical (nautical + AI-research) VGraphRAG pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")

    build_p = subparsers.add_parser("build", help="Ingest PDFs, build the RKG, and build all indexes")
    build_p.add_argument("--limit",   type=int,  default=None,
                         help="Process at most N chunks in the graph stage (for testing)")
    build_p.add_argument("--force",   action="store_true",
                         help="Force rebuild everything from scratch")
    build_p.add_argument("--skip-ingest", action="store_true",
                         help="Skip PDF ingestion, use existing chunks in clean_text_technical/")
    build_p.add_argument("--skip-graph", action="store_true",
                         help="Skip graph extraction, rebuild indexes only")
    build_p.add_argument("--rebuild-indexes", action="store_true",
                         help="Force-rebuild indexes without re-extracting the graph")

    stats_p = subparsers.add_parser("stats", help="Show graph statistics")

    query_p = subparsers.add_parser("query", help="Run a single query")
    query_p.add_argument("query", type=str, help="The question to ask")
    query_p.add_argument("--mode", choices=["specific", "abstract"], default=None,
                         help="Force a pipeline mode (default: auto-route)")
    query_p.add_argument("--n-chunks", type=int, default=6,
                         help="Number of passage chunks to retrieve (default: 6)")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        class DefaultArgs:
            limit = None
            force = False
            skip_ingest = False
            skip_graph = False
            rebuild_indexes = False
        print("No command specified — running full build.")
        cmd_build(DefaultArgs())


if __name__ == "__main__":
    main()
